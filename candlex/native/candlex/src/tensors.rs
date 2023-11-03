use crate::atoms;
use crate::error::CandlexError;
use crate::ops::{
    Acos, Acosh, Asin, Asinh, Atan, Atan2, Atanh, BitAnd, BitNot, BitOr, BitXor, Cbrt, Cosh,
    ErfInv, Erfc, Expm1, IsInf, IsNan, Log1p, LogicalAnd, LogicalOr, LogicalXor, Pow, Remainder,
    Shl, Shr, Sigmoid, Sign, Sinh, Tan,
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
    device: Atom,
    resource: ResourceArc<TensorRef>,
}

impl ExTensor {
    pub fn new(tensor: Tensor) -> Self {
        let dev_string = match tensor.device() {
            Device::Cpu => atoms::cpu(),
            Device::Cuda(_) => atoms::cuda(),
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
pub fn to_device(ex_tensor: ExTensor, device: Atom) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(
        ex_tensor.to_device(&device_from_atom(device)?)?,
    ))
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
pub fn index_select(t: ExTensor, indexes: ExTensor, dim: usize) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(t.index_select(indexes.deref(), dim)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn index_add(
    t: ExTensor,
    indexes: ExTensor,
    source: ExTensor,
    dim: usize,
) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(t.index_add(
        indexes.deref(),
        source.deref(),
        dim,
    )?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn chunk(t: ExTensor, num_chunks: usize) -> Result<Vec<ExTensor>, CandlexError> {
    Ok(t.chunk(num_chunks, 0)?
        .into_iter()
        .map(ExTensor::new)
        .collect())
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn squeeze(t: ExTensor, dim: usize) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(t.squeeze(dim)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn clamp(t: ExTensor, min_val: ExTensor, max_val: ExTensor) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(t.clamp(
        &min_val.broadcast_as(t.shape())?,
        &max_val.broadcast_as(t.shape())?,
    )?))
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
    Ok(ExTensor::new(_all(
        &ex_tensor.flatten_all()?,
        vec![0],
        false,
    )?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn all_within_dims(
    ex_tensor: ExTensor,
    dims: Vec<usize>,
    keep_dims: bool,
) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(_all(ex_tensor.deref(), dims, keep_dims)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn any(ex_tensor: ExTensor) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(_any(
        &ex_tensor.flatten_all()?,
        vec![0],
        false,
    )?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn any_within_dims(
    ex_tensor: ExTensor,
    dims: Vec<usize>,
    keep_dims: bool,
) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(_any(ex_tensor.deref(), dims, keep_dims)?))
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
pub fn reduce_max(
    ex_tensor: ExTensor,
    dim: usize,
    keep_dim: bool,
) -> Result<ExTensor, CandlexError> {
    let t = if keep_dim {
        ex_tensor.max_keepdim(dim)?
    } else {
        ex_tensor.max(dim)?
    };

    Ok(ExTensor::new(t))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn reduce_min(
    ex_tensor: ExTensor,
    dim: usize,
    keep_dim: bool,
) -> Result<ExTensor, CandlexError> {
    let t = if keep_dim {
        ex_tensor.min_keepdim(dim)?
    } else {
        ex_tensor.min(dim)?
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
pub fn slice_scatter(
    t: ExTensor,
    src: ExTensor,
    dim: usize,
    start: usize,
) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(t.slice_scatter(src.deref(), dim, start)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn pad_with_zeros(t: ExTensor, left: usize, right: usize) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(t.pad_with_zeros(0, left, right)?))
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

    Ok(ExTensor::new(tensor.conv1d(
        kernel.deref(),
        padding,
        stride,
        dilation,
        groups,
    )?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn conv2d(tensor: ExTensor, kernel: ExTensor) -> Result<ExTensor, CandlexError> {
    let padding = 0;
    let stride = 1;
    let dilation = 1;
    let groups = 1;

    Ok(ExTensor::new(tensor.conv2d(
        kernel.deref(),
        padding,
        stride,
        dilation,
        groups,
    )?))
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

#[rustler::nif(schedule = "DirtyCpu")]
pub fn dot(left: ExTensor, right: ExTensor) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(
        left.mul(&right.broadcast_as(left.shape())?)?
            .sum(left.rank() - 1)?,
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
unary_nif!(ceil);
unary_nif!(cos);
unary_nif!(erf);
unary_nif!(exp);
unary_nif!(floor);
unary_nif!(round);
unary_nif!(sin);
unary_nif!(log);
unary_nif!(sqrt);
unary_nif!(tanh);

custom_unary_nif!(acos, Acos);
custom_unary_nif!(acosh, Acosh);
custom_unary_nif!(asin, Asin);
custom_unary_nif!(asinh, Asinh);
custom_unary_nif!(atan, Atan);
custom_unary_nif!(atanh, Atanh);
custom_unary_nif!(bitwise_not, BitNot);
custom_unary_nif!(cbrt, Cbrt);
custom_unary_nif!(cosh, Cosh);
custom_unary_nif!(erfc, Erfc);
custom_unary_nif!(erf_inv, ErfInv);
custom_unary_nif!(expm1, Expm1);
custom_unary_nif!(is_infinity, IsInf);
custom_unary_nif!(is_nan, IsNan);
custom_unary_nif!(log1p, Log1p);
custom_unary_nif!(sigmoid, Sigmoid);
custom_unary_nif!(sign, Sign);
custom_unary_nif!(sinh, Sinh);
custom_unary_nif!(tan, Tan);

binary_nif!(add, broadcast_add);
binary_nif!(subtract, broadcast_sub);
binary_nif!(multiply, broadcast_mul);
binary_nif!(quotient, broadcast_div);
binary_nif!(max, broadcast_maximum);
binary_nif!(min, broadcast_minimum);
binary_nif!(equal, eq);
binary_nif!(not_equal, ne);
binary_nif!(greater, gt);
binary_nif!(greater_equal, ge);
binary_nif!(less, lt);
binary_nif!(less_equal, le);
binary_nif!(matmul, broadcast_matmul);

custom_binary_nif!(atan2, Atan2);
custom_binary_nif!(bitwise_and, BitAnd);
custom_binary_nif!(bitwise_or, BitOr);
custom_binary_nif!(bitwise_xor, BitXor);
custom_binary_nif!(left_shift, Shl);
custom_binary_nif!(logical_and, LogicalAnd);
custom_binary_nif!(logical_or, LogicalOr);
custom_binary_nif!(logical_xor, LogicalXor);
custom_binary_nif!(pow, Pow);
custom_binary_nif!(right_shift, Shr);
custom_binary_nif!(remainder, Remainder);

fn _any(tensor: &Tensor, dims: Vec<usize>, keep_dims: bool) -> Result<Tensor, CandlexError> {
    let comparison = tensor.ne(&tensor.zeros_like()?)?;

    let result = if keep_dims {
        dims.iter()
            .rev()
            .fold(comparison, |t, dim| t.max_keepdim(*dim).unwrap())
    } else {
        dims.iter()
            .rev()
            .fold(comparison, |t, dim| t.max(*dim).unwrap())
    };

    Ok(result)
}

fn _all(tensor: &Tensor, dims: Vec<usize>, keep_dims: bool) -> Result<Tensor, CandlexError> {
    let comparison = tensor.ne(&tensor.zeros_like()?)?;

    let result = if keep_dims {
        dims.iter()
            .rev()
            .fold(comparison, |t, dim| t.min_keepdim(*dim).unwrap())
    } else {
        dims.iter()
            .rev()
            .fold(comparison, |t, dim| t.min(*dim).unwrap())
    };

    Ok(result)
}

fn tuple_to_vec(term: Term) -> Result<Vec<usize>, rustler::Error> {
    rustler::types::tuple::get_tuple(term)?
        .iter()
        .map(|elem| elem.decode())
        .collect::<Result<_, _>>()
}

fn vec_to_tuple(env: Env, vec: Vec<usize>) -> Result<Term, rustler::Error> {
    Ok(rustler::types::tuple::make_tuple(
        env,
        &vec.into_iter()
            .map(|elem| elem.encode(env))
            .collect::<Vec<_>>(),
    ))
}

static CUDA_DEVICE: std::sync::Mutex<Option<Device>> = std::sync::Mutex::new(None);

fn device_from_atom(atom: Atom) -> Result<Device, CandlexError> {
    if atom == atoms::cpu() {
        Ok(Device::Cpu)
    } else if atom == atoms::cuda() {
        let mut cuda_device = CUDA_DEVICE.lock().unwrap();

        if let Some(device) = cuda_device.as_ref() {
            Ok(device.clone())
        } else {
            let new_cuda_device = Device::new_cuda(0)?;
            *cuda_device = Some(new_cuda_device.clone());

            Ok(new_cuda_device)
        }
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
