use crate::error::CandlexError;
use candle_core::{DType, Device, Tensor};
use rustler::{Binary, Env, NewBinary, NifStruct, ResourceArc, Term};
use std::ops::Deref;
use std::result::Result;
use std::str::FromStr;

pub struct TensorRef(pub Tensor);

#[derive(NifStruct)]
#[module = "Candlex.Backend"]
pub struct ExTensor {
    pub resource: ResourceArc<TensorRef>,
}

impl TensorRef {
    pub fn new(tensor: Tensor) -> Self {
        Self(tensor)
    }
}

impl ExTensor {
    pub fn new(tensor: Tensor) -> Self {
        Self {
            resource: ResourceArc::new(TensorRef::new(tensor)),
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
                &tuple_to_vec(shape),
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

pub fn load(env: Env, _info: Term) -> bool {
    rustler::resource!(TensorRef, env);
    true
}

fn tuple_to_vec(term: Term) -> Vec<usize> {
    rustler::types::tuple::get_tuple(term)
        .unwrap()
        .iter()
        .map(|elem| elem.decode().unwrap())
        .collect()
}

fn tensor_bytes(tensor: Tensor) -> Result<Vec<u8>, CandlexError> {
    Ok(
        match tensor.dtype() {
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
                // TODO: Support all dtypes
            _ => tensor
                .to_vec1::<u8>()?
                .iter()
                .flat_map(|val| val.to_ne_bytes())
                .collect(),
        }
    )
}
