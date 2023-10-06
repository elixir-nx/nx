#[cfg(feature = "cuda")]
use candle_core::CudaStorage;
use candle_core::{CpuStorage, CustomOp1, CustomOp2, Error, Layout, Shape};
use num_traits::cast::FromPrimitive;
use num_traits::Float;

fn erfc<T: Float + num_traits::FromPrimitive>(v: T) -> T {
    FromPrimitive::from_f64(statrs::function::erf::erfc(v.to_f64().unwrap())).unwrap()
}

fn erf_inv<T: Float + num_traits::FromPrimitive>(v: T) -> T {
    FromPrimitive::from_f64(statrs::function::erf::erf_inv(v.to_f64().unwrap())).unwrap()
}

macro_rules! custom_unary_op {
    ($struct_name:ident, $name:expr, $cpu_closure:expr, ($($dtypes:ident),+)) => {
        pub(crate) struct $struct_name;

        impl CustomOp1 for $struct_name {
            // Box<dyn> does not support const yet, so use a function to get the name.
            fn name(&self) -> &'static str {
                $name
            }

            /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
            /// offsets etc so the associated layout should be used to access it.
            fn cpu_fwd(
                &self,
                storage: &CpuStorage,
                layout: &Layout,
            ) -> Result<(CpuStorage, Shape), candle_core::Error> {
                use candle_core::backend::BackendStorage;

                match storage {
                    $(
                        CpuStorage::$dtypes(vec) => {
                            Ok(
                                (
                                    CpuStorage::$dtypes(candle_core::cpu_backend::unary_map(vec, layout, $cpu_closure)),
                                    layout.shape().clone()
                                )
                            )
                        }
                    )*
                    s => Err(Error::UnsupportedDTypeForOp(s.dtype(), $name).bt())?
                }
            }

            #[cfg(feature = "cuda")]
            fn cuda_fwd(
                &self,
                storage: &CudaStorage,
                layout: &Layout,
            ) -> Result<(CudaStorage, Shape), candle_core::Error> {
                use crate::kernels;
                use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
                use candle_core::cuda_backend::{kernel_name, Map1, WrapErr};
                use candle_core::{CudaDevice, WithDType};

                impl Map1 for $struct_name {
                    fn f<T: DeviceRepr + WithDType>(
                        &self,
                        src: &CudaSlice<T>,
                        device: &CudaDevice,
                        layout: &Layout,
                    ) -> Result<CudaSlice<T>, candle_core::Error> {
                        let src = src.slice(layout.start_offset()..);
                        let func = device.get_or_load_func(&kernel_name::<T>($name), kernels::CUSTOM_UNARY)?;
                        let dims = layout.shape().dims();
                        let elem_count = layout.shape().elem_count();
                        let launch_config = LaunchConfig::for_num_elems(elem_count as u32);
                        let dims_and_strides = device.htod_copy([dims, layout.stride()].concat()).w()?;
                        // SAFETY: Set later by running the kernel.
                        let dst = unsafe { device.alloc::<T>(elem_count) }.w()?;
                        let params = (elem_count, dims.len(), &dims_and_strides, &src, &dst);
                        // SAFETY: ffi.
                        unsafe { func.launch(launch_config, params) }.w()?;

                        Ok(dst)
                    }
                }

                use candle_core::backend::BackendStorage;
                let device = storage.device();
                let slice = $struct_name.map(&storage.slice, device, layout)?;

                Ok(
                    (
                        CudaStorage {
                            slice,
                            device: device.clone(),
                        },
                        layout.shape().clone()
                    )
                )
            }
        }
    };
}

macro_rules! custom_unary_bool_op {
    ($struct_name:ident, $name:expr, $fn_name:ident, ($($dtypes:ident),+)) => {
        pub(crate) struct $struct_name;

        impl CustomOp1 for $struct_name {
            // Box<dyn> does not support const yet, so use a function to get the name.
            fn name(&self) -> &'static str {
                $name
            }

            /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
            /// offsets etc so the associated layout should be used to access it.
            fn cpu_fwd(
                &self,
                storage: &CpuStorage,
                layout: &Layout,
            ) -> Result<(CpuStorage, Shape), candle_core::Error> {
                use candle_core::backend::BackendStorage;

                match storage {
                    $(
                        CpuStorage::$dtypes(vec) => {
                            Ok(
                                (
                                    CpuStorage::U8(
                                        candle_core::cpu_backend::unary_map(vec, layout, |v| u8::from(v.$fn_name()))
                                    ),
                                    layout.shape().clone()
                                )
                            )
                        }
                    )*
                    s => Err(Error::UnsupportedDTypeForOp(s.dtype(), $name).bt())?
                }
            }

            #[cfg(feature = "cuda")]
            fn cuda_fwd(
                &self,
                storage: &CudaStorage,
                layout: &Layout,
            ) -> Result<(CudaStorage, Shape), candle_core::Error> {
                use crate::kernels;
                use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits};
                use candle_core::cuda_backend::{kernel_name, CudaStorageSlice, Map1Any, WrapErr};
                use candle_core::{CudaDevice, WithDType};

                impl Map1Any for $struct_name {
                    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits, W: Fn(CudaSlice<T>) -> CudaStorageSlice>(
                        &self,
                        src: &CudaSlice<T>,
                        device: &CudaDevice,
                        layout: &Layout,
                        _wrap: W,
                    ) -> Result<CudaStorageSlice, candle_core::Error> {
                        let src = src.slice(layout.start_offset()..);
                        let func = device.get_or_load_func(&kernel_name::<T>($name), kernels::CUSTOM_UNARY)?;
                        let dims = layout.shape().dims();
                        let elem_count = layout.shape().elem_count();
                        let launch_config = LaunchConfig::for_num_elems(elem_count as u32);
                        let dims_and_strides = device.htod_copy([dims, layout.stride()].concat()).w()?;
                        // SAFETY: Set later by running the kernel.
                        let dst = unsafe { device.alloc::<u8>(elem_count) }.w()?;
                        let params = (elem_count, dims.len(), &dims_and_strides, &src, &dst);
                        // SAFETY: ffi.
                        unsafe { func.launch(launch_config, params) }.w()?;

                        Ok(CudaStorageSlice::U8(dst))
                    }
                }

                use candle_core::backend::BackendStorage;
                let device = storage.device();
                let slice = $struct_name.map(&storage.slice, device, layout)?;

                Ok(
                    (
                        CudaStorage {
                            slice,
                            device: device.clone(),
                        },
                        layout.shape().clone()
                    )
                )
            }
        }
    };
}

macro_rules! custom_binary_op {
    ($struct_name:ident, $name:literal, $cpu_closure:expr, ($($dtypes:ident),+)) => {
        pub(crate) struct $struct_name;

        impl CustomOp2 for $struct_name {
            fn name(&self) -> &'static str {
                $name
            }

            /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
            /// offsets etc so the associated layout should be used to access it.
            fn cpu_fwd(
                &self,
                s1: &CpuStorage,
                l1: &Layout,
                s2: &CpuStorage,
                l2: &Layout,
            ) -> Result<(CpuStorage, Shape), candle_core::Error> {
                use candle_core::backend::BackendStorage;

                match (s1, s2) {
                    $(
                        (CpuStorage::$dtypes(lhs), CpuStorage::$dtypes(rhs)) => {
                            Ok(
                                (
                                    CpuStorage::$dtypes(
                                        candle_core::cpu_backend::binary_map(l1, l2, lhs, rhs, $cpu_closure)
                                    ),
                                    l1.shape().clone()
                                )
                            )
                        }
                    )*
                    _ => {
                        Err(Error::DTypeMismatchBinaryOp {
                            lhs: s1.dtype(),
                            rhs: s2.dtype(),
                            op: self.name(),
                        }
                        .bt())
                    }
                }
            }

            #[cfg(feature = "cuda")]
            fn cuda_fwd(
                &self,
                s1: &CudaStorage,
                l1: &Layout,
                s2: &CudaStorage,
                l2: &Layout,
            ) -> Result<(CudaStorage, Shape), candle_core::Error> {
                use crate::kernels;
                use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits};
                use candle_core::cuda_backend::{kernel_name, Map2, WrapErr};
                use candle_core::{CudaDevice, WithDType};

                impl Map2 for $struct_name {
                    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
                        &self,
                        src1: &CudaSlice<T>,
                        layout1: &Layout,
                        src2: &CudaSlice<T>,
                        layout2: &Layout,
                        device: &CudaDevice,
                    ) -> Result<CudaSlice<T>, candle_core::Error> {
                        let shape1 = layout1.shape();
                        let dims1 = shape1.dims();
                        let elem_count1 = shape1.elem_count();
                        let launch_config = LaunchConfig::for_num_elems(elem_count1 as u32);
                        let dims_and_strides = device
                            .htod_copy([dims1, layout1.stride(), layout2.stride()].concat())
                            .w()?;
                        let src1 = src1.slice(layout1.start_offset()..);
                        let src2 = src2.slice(layout2.start_offset()..);
                        let func = device.get_or_load_func(&kernel_name::<T>($name), kernels::CUSTOM_BINARY)?;
                        // SAFETY: Set later by running the kernel.
                        let out = unsafe { device.alloc::<T>(elem_count1) }.w()?;
                        let params = (elem_count1, dims1.len(), &dims_and_strides, &src1, &src2, &out);
                        // SAFETY: ffi
                        unsafe { func.launch(launch_config, params) }.w()?;

                        Ok(out)
                    }
                }

                use candle_core::backend::BackendStorage;
                let device = s1.device();
                let slice = $struct_name.map(&s1.slice, l1, &s2.slice, l2, device)?;

                Ok(
                    (
                        CudaStorage {
                            slice,
                            device: device.clone(),
                        },
                        l1.shape().clone()
                    )
                )
            }
        }
    }
}

macro_rules! custom_binary_bool_op {
    ($struct_name:ident, $name:literal, $cpu_closure:expr, ($($dtypes:ident),+)) => {
        pub(crate) struct $struct_name;

        impl CustomOp2 for $struct_name {
            fn name(&self) -> &'static str {
                $name
            }

            /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
            /// offsets etc so the associated layout should be used to access it.
            fn cpu_fwd(
                &self,
                s1: &CpuStorage,
                l1: &Layout,
                s2: &CpuStorage,
                l2: &Layout,
            ) -> Result<(CpuStorage, Shape), candle_core::Error> {
                use candle_core::backend::BackendStorage;

                match (s1, s2) {
                    $(
                        (CpuStorage::$dtypes(lhs), CpuStorage::$dtypes(rhs)) => {
                            Ok(
                                (
                                    CpuStorage::U8(
                                        candle_core::cpu_backend::binary_map(
                                            l1,
                                            l2,
                                            lhs,
                                            rhs,
                                            |v1, v2| u8::from($cpu_closure(v1, v2))
                                        )
                                    ),
                                    l1.shape().clone()
                                )
                            )
                        }
                    )*
                    _ => {
                        Err(Error::DTypeMismatchBinaryOp {
                            lhs: s1.dtype(),
                            rhs: s2.dtype(),
                            op: self.name(),
                        }
                        .bt())
                    }
                }
            }

            #[cfg(feature = "cuda")]
            fn cuda_fwd(
                &self,
                s1: &CudaStorage,
                l1: &Layout,
                s2: &CudaStorage,
                l2: &Layout,
            ) -> Result<(CudaStorage, Shape), candle_core::Error> {
                use crate::kernels;
                use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits};
                use candle_core::cuda_backend::{kernel_name, CudaStorageSlice, Map2Any, WrapErr};
                use candle_core::{CudaDevice, WithDType};

                impl Map2Any for $struct_name {
                    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
                        &self,
                        src1: &CudaSlice<T>,
                        layout1: &Layout,
                        src2: &CudaSlice<T>,
                        layout2: &Layout,
                        device: &CudaDevice,
                    ) -> Result<CudaStorageSlice, candle_core::Error> {
                        let shape1 = layout1.shape();
                        let dims1 = shape1.dims();
                        let elem_count1 = shape1.elem_count();
                        let launch_config = LaunchConfig::for_num_elems(elem_count1 as u32);
                        let dims_and_strides = device
                            .htod_copy([dims1, layout1.stride(), layout2.stride()].concat())
                            .w()?;
                        let src1 = src1.slice(layout1.start_offset()..);
                        let src2 = src2.slice(layout2.start_offset()..);
                        let func = device.get_or_load_func(&kernel_name::<T>($name), kernels::CUSTOM_BINARY)?;
                        // SAFETY: Set later by running the kernel.
                        let out = unsafe { device.alloc::<u8>(elem_count1) }.w()?;
                        let params = (elem_count1, dims1.len(), &dims_and_strides, &src1, &src2, &out);
                        // SAFETY: ffi
                        unsafe { func.launch(launch_config, params) }.w()?;

                        Ok(CudaStorageSlice::U8(out))
                    }
                }

                use candle_core::backend::BackendStorage;
                let device = s1.device();
                let slice = $struct_name.map(&s1.slice, l1, &s2.slice, l2, device)?;

                Ok(
                    (
                        CudaStorage {
                            slice,
                            device: device.clone(),
                        },
                        l1.shape().clone()
                    )
                )
            }
        }
    }
}

custom_unary_op!(Acos, "acos", |v| v.acos(), (BF16, F16, F32, F64));
custom_unary_op!(Acosh, "acosh", |v| v.acosh(), (BF16, F16, F32, F64));
custom_unary_op!(Asin, "asin", |v| v.asin(), (BF16, F16, F32, F64));
custom_unary_op!(Asinh, "asinh", |v| v.asinh(), (BF16, F16, F32, F64));
custom_unary_op!(Atan, "atan", |v| v.atan(), (BF16, F16, F32, F64));
custom_unary_op!(Atanh, "atanh", |v| v.atanh(), (BF16, F16, F32, F64));
custom_unary_op!(BitNot, "bit_not", |v| !v, (U8, U32, I64));
custom_unary_op!(Cbrt, "cbrt", |v| v.cbrt(), (BF16, F16, F32, F64));
custom_unary_op!(Cosh, "cosh", |v| v.cosh(), (BF16, F16, F32, F64));
custom_unary_op!(Erfc, "erfc", |v| erfc(v), (BF16, F16, F32, F64));
custom_unary_op!(ErfInv, "erf_inv", |v| erf_inv(v), (BF16, F16, F32, F64));
custom_unary_op!(Expm1, "expm1", |v| v.exp_m1(), (BF16, F16, F32, F64));
custom_unary_op!(Log1p, "ln_1p", |v| v.ln_1p(), (BF16, F16, F32, F64));
custom_unary_op!(Sigmoid, "sigmoid", |v| 1. / (1. + (-v).exp()), (F32, F64));
custom_unary_op!(Sign, "sign", |v| v.signum(), (I64, BF16, F16, F32, F64));
custom_unary_op!(Sinh, "sinh", |v| v.sinh(), (BF16, F16, F32, F64));
custom_unary_op!(Tan, "tan", |v| v.tan(), (BF16, F16, F32, F64));
custom_unary_bool_op!(IsInf, "is_inf", is_infinite, (F32, F64));
custom_unary_bool_op!(IsNan, "is_nan", is_nan, (F32, F64));

custom_binary_op!(BitAnd, "bit_and", |v1, v2| v1 & v2, (U32, I64));
custom_binary_op!(BitOr, "bit_or", |v1, v2| v1 | v2, (U32, I64));
custom_binary_op!(BitXor, "bit_xor", |v1, v2| v1 ^ v2, (U32, I64));
custom_binary_op!(Atan2, "atan2", |v1, v2| v1.atan2(v2), (F32, F64));
custom_binary_op!(Pow, "pow", |v1, v2| v1.powf(v2), (F32, F64));
custom_binary_op!(
    Remainder,
    "remainder",
    |v1, v2| v1 % v2,
    (U8, I64, F32, F64)
);
custom_binary_op!(Shl, "shl", |v1, v2| v1 << v2, (U32, I64));
custom_binary_op!(Shr, "shr", |v1, v2| v1 >> v2, (U32, I64));
custom_binary_bool_op!(
    LogicalAnd,
    "logical_and",
    |v1, v2| if v1 as i8 != 0 && v2 as i8 != 0 { 1 } else { 0 },
    (U8, U32, I64, F32, F64)
);
custom_binary_bool_op!(
    LogicalOr,
    "logical_or",
    |v1, v2| if v1 as i8 == 0 && v2 as i8 == 0 { 0 } else { 1 },
    (U8, U32, I64, F32, F64)
);
custom_binary_bool_op!(
    LogicalXor,
    "logical_xor",
    |v1, v2| if (v1 as i8 != 0) == (v2 as i8 != 0) {
        0
    } else {
        1
    },
    (U8, U32, I64, F32, F64)
);
