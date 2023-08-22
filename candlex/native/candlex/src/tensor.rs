use candle_core::Tensor;
use rustler::{NifStruct, ResourceArc};
use std::ops::Deref;

pub struct TensorRef(pub Tensor);

impl TensorRef {
    pub fn new(tensor: Tensor) -> Self {
        Self(tensor)
    }
}

#[derive(NifStruct)]
#[module = "Candlex.Backend"]
pub struct ExTensor {
    resource: ResourceArc<TensorRef>,
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
