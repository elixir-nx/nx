use crate::error::CandlexError;
use candle_core::{DType, Device, Tensor};
use half::{bf16, f16};
use rustler::{Binary, Env, NewBinary, NifStruct, ResourceArc, Term};
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
pub fn from_binary(binary: Binary, dtype_str: &str, shape: Term) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(Tensor::from_raw_buffer(
        binary.as_slice(),
        // TODO: Handle DTypeParseError
        DType::from_str(dtype_str).unwrap(),
        // TODO: Handle rustler::Error
        &tuple_to_vec(shape).unwrap(),
        &Device::Cpu,
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
pub fn add(left: ExTensor, right: ExTensor) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(left.broadcast_add(&right)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn max(left: ExTensor, right: ExTensor) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(left.broadcast_maximum(&right)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn min(left: ExTensor, right: ExTensor) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(left.broadcast_minimum(&right)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn multiply(left: ExTensor, right: ExTensor) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(left.broadcast_mul(&right)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn equal(left: ExTensor, right: ExTensor) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(left.eq(&right)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn greater_equal(left: ExTensor, right: ExTensor) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(left.ge(&right)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn less(left: ExTensor, right: ExTensor) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(left.lt(&right)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn subtract(left: ExTensor, right: ExTensor) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(left.broadcast_sub(&right)?))
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
) -> Result<ExTensor, CandlexError> {
    Ok(ExTensor::new(
        Tensor::arange(start, end, &Device::Cpu)?
            .to_dtype(DType::from_str(dtype_str).unwrap())?
            .reshape(tuple_to_vec(shape).unwrap())?,
    ))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn all(ex_tensor: ExTensor) -> Result<ExTensor, CandlexError> {
    let device = &Device::Cpu;
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

fn tuple_to_vec(term: Term) -> Result<Vec<usize>, rustler::Error> {
    Ok(rustler::types::tuple::get_tuple(term)?
        .iter()
        .map(|elem| elem.decode())
        .collect::<Result<_, _>>()?)
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
