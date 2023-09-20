use crate::atoms;
use crate::error::CandlexError;
use crate::ops::{
    Acos, Asin, Atan, BitAnd, BitNot, BitOr, BitXor, Cbrt, Ceil, ErfInv, Floor, IsInf, Log1p,
    LogicalOr, LogicalXor, Pow, Round, Shl, Shr, Sigmoid, Tan,
};
use candle_core::{DType, Device, Tensor};
use half::{bf16, f16};
use rustler::{Atom, Binary, Encoder, Env, NewBinary, NifStruct, ResourceArc, Term};
use std::ops::Deref;
use std::result::Result;
use std::str::FromStr;

pub(crate) struct TensorRef(Tensor);

#[derive(NifStruct)]
#[module = "Candlex.Backend"]
pub struct ExTensor {
    device: String,
    resource: ResourceArc<TensorRef>,
}

impl ExTensor {
    pub fn new(tensor: Tensor) -> Self {
        let dev_string = match tensor.device() {
            Device::Cpu => String::from("cpu"),
            Device::Cuda(_) => String::from("cuda"),
        };

        Self {
            device: dev_string,
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
pub fn gather(t: ExTensor, indexes: ExTensor, dim: usize) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(t.gather(indexes.deref(), dim)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn chunk(t: ExTensor, num_chunks: usize) -> Result<Vec<ExTensor>, CandlexError> {
    Ok(t.chunk(num_chunks, 0)?
        .into_iter()
        .map(|t| ExTensor::new(t))
        .collect())
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn squeeze(t: ExTensor, dim: usize) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(t.squeeze(dim)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn transpose(t: ExTensor, dim1: usize, dim2: usize) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(t.transpose(dim1, dim2)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn rsqrt(t: ExTensor) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(t.sqrt()?.recip()?))
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
pub fn reduce_max(ex_tensor: ExTensor, dim: usize, keep_dim: bool) -> Result<ExTensor, CandlexError> {
    let t = if keep_dim {
        ex_tensor.max_keepdim(dim)?
    } else {
        ex_tensor.max(dim)?
    };

    Ok(ExTensor::new(t))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn sum(
    ex_tensor: ExTensor,
    dims: Vec<usize>,
    keep_dims: bool,
) -> Result<ExTensor, CandlexError> {
    let t = if keep_dims {
        ex_tensor.sum_keepdim(dims)?
    } else {
        ex_tensor.sum(dims)?
    };

    Ok(ExTensor::new(t))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn permute(ex_tensor: ExTensor, dims: Vec<usize>) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(ex_tensor.permute(dims)?))
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
pub fn dtype(t: ExTensor) -> Result<&'static str, CandlexError> {
    Ok(t.dtype().as_str())
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn t_shape(env: Env, t: ExTensor) -> Result<Term, CandlexError> {
    Ok(vec_to_tuple(env, t.shape().clone().into_dims()).unwrap())
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn concatenate(ex_tensors: Vec<ExTensor>, dim: usize) -> Result<ExTensor, CandlexError> {
    let tensors = ex_tensors
        .iter()
        .map(|t| t.deref())
        .collect::<Vec<&Tensor>>();
    Ok(ExTensor::new(Tensor::cat(&tensors[..], dim)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn conv1d(tensor: ExTensor, kernel: ExTensor) -> Result<ExTensor, CandlexError> {
    let padding = 0;
    let stride = 1;
    let dilation = 1;
    let groups = 1;

    Ok(ExTensor::new(tensor.conv1d(kernel.deref(), padding, stride, dilation, groups)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn conv2d(tensor: ExTensor, kernel: ExTensor) -> Result<ExTensor, CandlexError> {
    let padding = 0;
    let stride = 1;
    let dilation = 1;
    let groups = 1;

    Ok(ExTensor::new(tensor.conv2d(kernel.deref(), padding, stride, dilation, groups)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn divide(left: ExTensor, right: ExTensor) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(
        // Need to force float in case we receive integers, given
        // candle rounds down integer division.
        left.to_dtype(DType::F32)?
            .broadcast_div(&right.to_dtype(DType::F32)?)?,
    ))
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
            Ok(ExTensor::new(left.$native_fn_name(right.deref())?))
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

macro_rules! custom_binary_nif {
    ($nif_name:ident, $custom_op_name:ident) => {
        #[rustler::nif(schedule = "DirtyCpu")]
        pub fn $nif_name(left: ExTensor, right: ExTensor) -> Result<ExTensor, CandlexError> {
            Ok(ExTensor::new(
                left.apply_op2_no_bwd(right.deref(), &$custom_op_name)?,
            ))
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
custom_unary_nif!(bitwise_not, BitNot);
custom_unary_nif!(cbrt, Cbrt);
custom_unary_nif!(ceil, Ceil);
custom_unary_nif!(erf_inv, ErfInv);
custom_unary_nif!(floor, Floor);
custom_unary_nif!(is_infinity, IsInf);
custom_unary_nif!(log1p, Log1p);
custom_unary_nif!(round, Round);
custom_unary_nif!(sigmoid, Sigmoid);
custom_unary_nif!(tan, Tan);

binary_nif!(add, broadcast_add);
binary_nif!(subtract, broadcast_sub);
binary_nif!(multiply, broadcast_mul);
binary_nif!(max, broadcast_maximum);
binary_nif!(min, broadcast_minimum);
binary_nif!(equal, eq);
binary_nif!(greater, gt);
binary_nif!(greater_equal, ge);
binary_nif!(less, lt);
binary_nif!(less_equal, le);
binary_nif!(matmul, broadcast_matmul);

custom_binary_nif!(bitwise_and, BitAnd);
custom_binary_nif!(bitwise_or, BitOr);
custom_binary_nif!(bitwise_xor, BitXor);
custom_binary_nif!(left_shift, Shl);
custom_binary_nif!(logical_or, LogicalOr);
custom_binary_nif!(logical_xor, LogicalXor);
custom_binary_nif!(pow, Pow);
custom_binary_nif!(right_shift, Shr);

fn tuple_to_vec(term: Term) -> Result<Vec<usize>, rustler::Error> {
    Ok(rustler::types::tuple::get_tuple(term)?
        .iter()
        .map(|elem| elem.decode())
        .collect::<Result<_, _>>()?)
}

fn vec_to_tuple(env: Env, vec: Vec<usize>) -> Result<Term, rustler::Error> {
    Ok(rustler::types::tuple::make_tuple(
        env,
        &vec.into_iter()
            .map(|elem| elem.encode(env))
            .collect::<Vec<_>>(),
    ))
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
