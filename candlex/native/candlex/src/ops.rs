use candle_core::{CpuStorage, CustomOp1, CustomOp2, Error, Layout, Shape};
use num_traits::Float;

macro_rules! custom_unary_op {
    ($struct_name:ident, $name:expr, $fn_name:ident) => {
        pub(crate) struct $struct_name;

        fn $fn_name<T: Float>(value: T) -> T {
            value.$fn_name()
        }

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

                let storage = candle_core::map_dtype!(
                    $name,
                    storage,
                    |vec| candle_core::cpu_backend::unary_map(vec, layout, |v| $fn_name(v)),
                    (BF16, F16, F32, F64)
                );

                Ok((storage, layout.shape().clone()))
            }
        }
    };
}

macro_rules! custom_unary_op_closure {
    ($struct_name:ident, $name:expr, $closure:expr) => {
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

                // TODO: Find a way to make map_dtype! play well with inferred closure
                //       params types?
                //
                // let storage = candle_core::map_dtype!(
                //     $name,
                //     storage,
                //     |vec| candle_core::cpu_backend::unary_map(vec, layout, $closure),
                //     (U8, U32, I64)
                // );

                // Ok((storage, layout.shape().clone()))

                match storage {
                    CpuStorage::U8(vec) => {
                        let data = candle_core::cpu_backend::unary_map(vec, layout, $closure);
                        Ok((CpuStorage::U8(data), layout.shape().clone()))
                    }
                    CpuStorage::U32(vec) => {
                        let data = candle_core::cpu_backend::unary_map(vec, layout, $closure);
                        Ok((CpuStorage::U32(data), layout.shape().clone()))
                    }
                    CpuStorage::I64(vec) => {
                        let data = candle_core::cpu_backend::unary_map(vec, layout, $closure);
                        Ok((CpuStorage::I64(data), layout.shape().clone()))
                    }
                    s => Err(Error::UnsupportedDTypeForOp(s.dtype(), $name).bt())?
                }
            }
        }
    };
}

macro_rules! custom_binary_op {
    ($struct_name:ident, $name:literal, $closure:expr) => {
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

                // let storage = candle_core::map_dtype!(
                //     $name,
                //     s1,
                //     |vec1| candle_core::cpu_backend::binary_map(l1, l2, vec1, s2, |v1, v2| op_wrapper(v1, v2)),
                //     (U8, U32, I64)
                // );

                match (s1, s2) {
                    (CpuStorage::U32(lhs), CpuStorage::U32(rhs)) => {
                        let data = candle_core::cpu_backend::binary_map(l1, l2, lhs, rhs, $closure);

                        Ok((CpuStorage::U32(data), l1.shape().clone()))
                    }
                    (CpuStorage::I64(lhs), CpuStorage::I64(rhs)) => {
                        let data = candle_core::cpu_backend::binary_map(l1, l2, lhs, rhs, $closure);

                        Ok((CpuStorage::I64(data), l1.shape().clone()))
                    }
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
        }
    }
}

custom_unary_op!(Acos, "acos", acos);
custom_unary_op!(Asin, "asin", asin);
custom_unary_op!(Atan, "atan", atan);
custom_unary_op!(Cbrt, "cbrt", cbrt);
custom_unary_op!(Ceil, "ceil", ceil);
custom_unary_op!(Floor, "floor", floor);
custom_unary_op!(Log1p, "ln_1p", ln_1p);
custom_unary_op!(Round, "round", round);
custom_unary_op!(Tan, "tan", tan);
custom_unary_op_closure!(BitNot, "bit_not", |v| !v);

custom_binary_op!(BitAnd, "bit_and", |v1, v2| v1 & v2);
custom_binary_op!(BitOr, "bit_or", |v1, v2| v1 | v2);
custom_binary_op!(BitXor, "bit_xor", |v1, v2| v1 ^ v2);
custom_binary_op!(Shl, "shl", |v1, v2| v1 << v2);
custom_binary_op!(Shr, "shr", |v1, v2| v1 >> v2);
