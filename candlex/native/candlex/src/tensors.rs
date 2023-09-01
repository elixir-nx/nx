use crate::atoms;
use crate::error::CandlexError;
use crate::ops::{Acos, Asin, Atan, Cbrt, Ceil, Floor, Round, Tan};
use candle_core::{DType, Device, Tensor};
use half::{bf16, f16};
use rustler::{Atom, Binary, Env, NewBinary, NifStruct, ResourceArc, Term};
use std::ops::Deref;
use std::result::Result;
use std::str::FromStr;

pub(crate) struct TensorRef(Tensor);

#[derive(NifStruct)]
#[module = "Candlex.Backend"]
pub struct ExTensor {
    resource: ResourceArc<TensorRef>,
}

impl ExTensor {
    pub fn new(tensor: Tensor) -> Self {
        Self {
            resource: ResourceArc::new(TensorRef(tensor)),
        }
    }
}

// Implement Deref so we can call `Tensor` functions directly from an `ExTensor` struct.
impl Deref for ExTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.resource.0
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn from_binary(
    binary: Binary,
    dtype_str: &str,
    shape: Term,
    device: Atom,
) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(Tensor::from_raw_buffer(
        binary.as_slice(),
        // TODO: Handle DTypeParseError
        DType::from_str(dtype_str).unwrap(),
        // TODO: Handle rustler::Error
        &tuple_to_vec(shape).unwrap(),
        &device_from_atom(device)?,
    )?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn to_binary(env: Env, ex_tensor: ExTensor) -> Result<Binary, CandlexError> {
    let bytes = tensor_bytes(ex_tensor.flatten_all()?)?;
    let mut binary = NewBinary::new(env, bytes.len());
    binary.as_mut_slice().copy_from_slice(bytes.as_slice());

    Ok(binary.into())
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn narrow(
    t: ExTensor,
    dim: usize,
    start: usize,
    length: usize,
) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(t.narrow(dim, start, length)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn squeeze(t: ExTensor, dim: usize) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(t.squeeze(dim)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn arange(
    start: i64,
    end: i64,
    dtype_str: &str,
    shape: Term,
    device: Atom,
) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(
        Tensor::arange(start, end, &device_from_atom(device)?)?
            .to_dtype(DType::from_str(dtype_str).unwrap())?
            .reshape(tuple_to_vec(shape).unwrap())?,
    ))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn all(ex_tensor: ExTensor) -> Result<ExTensor, CandlexError> {
    let device = ex_tensor.device();
    let t = ex_tensor.flatten_all()?;
    let dims = t.shape().dims();
    let on_true = Tensor::ones(dims, DType::U8, device)?;
    let on_false = Tensor::zeros(dims, DType::U8, device)?;

    let bool_scalar = match t
        .where_cond(&on_true, &on_false)?
        .min(0)?
        .to_scalar::<u8>()?
    {
        0 => 0u8,
        _ => 1u8,
    };

    Ok(ExTensor::new(Tensor::new(bool_scalar, device)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn argmax(ex_tensor: ExTensor, dim: usize, keep_dim: bool) -> Result<ExTensor, CandlexError> {
    let t = if keep_dim {
        ex_tensor.argmax_keepdim(dim)?
    } else {
        ex_tensor.argmax(dim)?
    };

    Ok(ExTensor::new(t))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn argmin(ex_tensor: ExTensor, dim: usize, keep_dim: bool) -> Result<ExTensor, CandlexError> {
    let t = if keep_dim {
        ex_tensor.argmin_keepdim(dim)?
    } else {
        ex_tensor.argmin(dim)?
    };

    Ok(ExTensor::new(t))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn broadcast_to(t: ExTensor, shape: Term) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(t.broadcast_as(tuple_to_vec(shape).unwrap())?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn reshape(t: ExTensor, shape: Term) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(t.reshape(tuple_to_vec(shape).unwrap())?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn where_cond(
    t: ExTensor,
    on_true: ExTensor,
    on_false: ExTensor,
) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(t.where_cond(&on_true, &on_false)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn to_type(t: ExTensor, dtype_str: &str) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(
        t.to_dtype(DType::from_str(dtype_str).unwrap())?,
    ))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn concatenate(ex_tensors: Vec<ExTensor>, dim: usize) -> Result<ExTensor, CandlexError> {
    let tensors = ex_tensors
        .iter()
        .map(|t| t.deref())
        .collect::<Vec<&Tensor>>();
    Ok(ExTensor::new(Tensor::cat(&tensors[..], dim)?))
}

macro_rules! unary_nif {
    ($nif_name:ident, $native_fn_name:ident) => {
        #[rustler::nif(schedule = "DirtyCpu")]
        pub fn $nif_name(ex_tensor: ExTensor) -> Result<ExTensor, CandlexError> {
            Ok(ExTensor::new(ex_tensor.$native_fn_name()?))
        }
    };
    ($nif_name:ident) => {
        unary_nif!($nif_name, $nif_name);
    };
}

macro_rules! binary_nif {
    ($nif_name:ident, $native_fn_name:ident) => {
        #[rustler::nif(schedule = "DirtyCpu")]
        pub fn $nif_name(left: ExTensor, right: ExTensor) -> Result<ExTensor, CandlexError> {
            Ok(ExTensor::new(left.$native_fn_name(&right)?))
        }
    };
}

macro_rules! custom_unary_nif {
    ($nif_name:ident, $custom_op_name:ident) => {
        #[rustler::nif(schedule = "DirtyCpu")]
        pub fn $nif_name(ex_tensor: ExTensor) -> Result<ExTensor, CandlexError> {
            Ok(ExTensor::new(ex_tensor.apply_op1_no_bwd(&$custom_op_name)?))
        }
    };
}

unary_nif!(negate, neg);
unary_nif!(abs);
unary_nif!(cos);
unary_nif!(exp);
unary_nif!(sin);
unary_nif!(log);
unary_nif!(sqrt);
unary_nif!(tanh);

custom_unary_nif!(acos, Acos);
custom_unary_nif!(asin, Asin);
custom_unary_nif!(atan, Atan);
custom_unary_nif!(cbrt, Cbrt);
custom_unary_nif!(ceil, Ceil);
custom_unary_nif!(floor, Floor);
custom_unary_nif!(round, Round);
custom_unary_nif!(tan, Tan);

binary_nif!(add, broadcast_add);
binary_nif!(subtract, broadcast_sub);
binary_nif!(multiply, broadcast_mul);
binary_nif!(max, broadcast_maximum);
binary_nif!(min, broadcast_minimum);
binary_nif!(equal, eq);
binary_nif!(greater_equal, ge);
binary_nif!(less, lt);
binary_nif!(less_equal, le);
binary_nif!(matmul, broadcast_matmul);

fn tuple_to_vec(term: Term) -> Result<Vec<usize>, rustler::Error> {
    Ok(rustler::types::tuple::get_tuple(term)?
        .iter()
        .map(|elem| elem.decode())
        .collect::<Result<_, _>>()?)
}

fn device_from_atom(atom: Atom) -> Result<Device, CandlexError> {
    if atom == atoms::cpu() {
        Ok(Device::Cpu)
    // } else if atom == atoms::cuda() {
    //     Ok(Device::new_cuda(0)?)
    } else {
        Err(CandlexError::Other(format!(
            "unsupported device {:?}",
            atom
        )))
    }
}

fn tensor_bytes(tensor: Tensor) -> Result<Vec<u8>, CandlexError> {
    Ok(match tensor.dtype() {
        DType::I64 => tensor
            .to_vec1::<i64>()?
            .iter()
            .flat_map(|val| val.to_ne_bytes())
            .collect(),
        DType::U8 => tensor
            .to_vec1::<u8>()?
            .iter()
            .flat_map(|val| val.to_ne_bytes())
            .collect(),
        DType::U32 => tensor
            .to_vec1::<u32>()?
            .iter()
            .flat_map(|val| val.to_ne_bytes())
            .collect(),
        DType::F16 => tensor
            .to_vec1::<f16>()?
            .iter()
            .flat_map(|val| val.to_ne_bytes())
            .collect(),
        DType::F32 => tensor
            .to_vec1::<f32>()?
            .iter()
            .flat_map(|val| val.to_ne_bytes())
            .collect(),
        DType::F64 => tensor
            .to_vec1::<f64>()?
            .iter()
            .flat_map(|val| val.to_ne_bytes())
            .collect(),
        DType::BF16 => tensor
            .to_vec1::<bf16>()?
            .iter()
            .flat_map(|val| val.to_ne_bytes())
            .collect(),
    })
}
