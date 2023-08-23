use crate::error::CandlexError;
use candle_core::{DType, Device, Tensor};
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
    Ok(
        ExTensor::new(
            Tensor::from_raw_buffer(
                binary.as_slice(),
                // TODO: Handle DTypeParseError
                DType::from_str(dtype_str).unwrap(),
                // TODO: Handle rustler::Error
                &tuple_to_vec(shape).unwrap(),
                &Device::Cpu,
            )?
        )
    )
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn to_binary(env: Env, ex_tensor: ExTensor) -> Result<Binary, CandlexError> {
    let bytes = tensor_bytes(ex_tensor.flatten_all()?)?;
    let mut binary = NewBinary::new(env, bytes.len());
    binary.as_mut_slice().copy_from_slice(bytes.as_slice());

    Ok(binary.into())
}

fn tuple_to_vec(term: Term) -> Result<Vec<usize>, rustler::Error> {
    Ok(
        rustler::types::tuple::get_tuple(term)?
        .iter()
        .map(|elem| elem.decode())
        .collect::<Result<_, _>>()?
    )
}

fn tensor_bytes(tensor: Tensor) -> Result<Vec<u8>, CandlexError> {
    Ok(
        match tensor.dtype() {
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
                // TODO: Support f16 and bf16
            _ => tensor
                .to_vec1::<u8>()?
                .iter()
                .flat_map(|val| val.to_ne_bytes())
                .collect(),
        }
    )
}
