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
    let tensor = ex_tensor.flatten_all().unwrap();
    let vec = tensor.to_vec1::<u32>().unwrap();

    let bytes: Vec<u8> = vec.iter().flat_map(|val| val.to_ne_bytes().to_vec()).collect();

    let mut binary = NewBinary::new(env, bytes.len());
    binary.as_mut_slice().copy_from_slice(bytes.as_slice());

    binary.into()
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn from_binary(binary: Binary, shape: Term) -> ExTensor {
    let slice = binary.as_slice();

    // let slice: &[u32; 2] = unsafe {
        // std::mem::transmute::<&[u8], &[u32; 2]>(binary.as_slice())
    // };
    let (_prefx, slice, _suffix) = unsafe { slice.align_to::<u32>() };

    // println!("{:?}", &slice);
    // let slice = u32::from_ne_bytes(slice);
    ExTensor::new(Tensor::from_slice(slice, tuple_to_vec(shape), &Device::Cpu).unwrap())

    // let mut vec = vec![0u32; 2];
    // reader.read_u32_into::<LittleEndian>(&mut vec)?;
    // ExTensor::new(Tensor::from_vec(vec, 2, &Device::Cpu).unwrap())
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
