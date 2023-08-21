use candle_core::{Tensor, Device};
use rustler::{Binary, Env, NewBinary, NifStruct, ResourceArc, Term};
use std::ops::Deref;

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

// Implement Deref so we can call `Tensor` functions directly from a `ExTensor` struct.
impl Deref for ExTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.resource.0
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn scalar_tensor(scalar: u32) -> ExTensor {
    ExTensor::new(Tensor::new(scalar, &Device::Cpu).unwrap())
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn to_binary(env: Env, ex_tensor: ExTensor) -> Binary {
    let bytes: Vec<u8> =
        ex_tensor
        .flatten_all()
        .unwrap()
        .to_vec1::<u32>()
        .unwrap()
        .iter()
        .flat_map(|val| val.to_ne_bytes().to_vec())
        .collect();

    let mut binary = NewBinary::new(env, bytes.len());
    binary.as_mut_slice().copy_from_slice(bytes.as_slice());

    binary.into()
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn from_binary(binary: Binary, shape: Term) -> ExTensor {
    let (_prefx, slice, _suffix) = unsafe { binary.as_slice().align_to::<u32>() };

    ExTensor::new(Tensor::from_slice(slice, tuple_to_vec(shape), &Device::Cpu).unwrap())
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
