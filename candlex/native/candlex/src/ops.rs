use candle_core::{CpuStorage, CustomOp1, CustomOp2, Error, Layout, Shape};
use num_traits::Float;
use statrs::function::erf::erf_inv;

macro_rules! custom_unary_op {
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
                            let data = candle_core::cpu_backend::unary_map(vec, layout, |v| v.$fn_name());
                            Ok((CpuStorage::$dtypes(data), layout.shape().clone()))
                        }
                    )*
                    s => Err(Error::UnsupportedDTypeForOp(s.dtype(), $name).bt())?
                }
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
                            let data = candle_core::cpu_backend::unary_map(vec, layout, |v| u8::from(v.$fn_name()));
                            Ok((CpuStorage::U8(data), layout.shape().clone()))
                        }
                    )*
                    s => Err(Error::UnsupportedDTypeForOp(s.dtype(), $name).bt())?
                }
            }
        }
    };
}

macro_rules! custom_unary_op_closure {
    ($struct_name:ident, $name:expr, $closure:expr, ($($dtypes:ident),+)) => {
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
                            let data = candle_core::cpu_backend::unary_map(vec, layout, $closure);
                            Ok((CpuStorage::$dtypes(data), layout.shape().clone()))
                        }
                    )*
                    s => Err(Error::UnsupportedDTypeForOp(s.dtype(), $name).bt())?
                }
            }
        }
    };
}

macro_rules! custom_binary_op {
    ($struct_name:ident, $name:literal, $closure:expr, ($($dtypes:ident),+)) => {
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
                            let data = candle_core::cpu_backend::binary_map(l1, l2, lhs, rhs, $closure);

                            Ok((CpuStorage::$dtypes(data), l1.shape().clone()))
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
        }
    }
}

macro_rules! custom_binary_bool_op {
    ($struct_name:ident, $name:literal, $closure:expr, ($($dtypes:ident),+)) => {
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
                            let data = candle_core::cpu_backend::binary_map(l1, l2, lhs, rhs, |v1, v2| u8::from($closure(v1, v2)));

                            Ok((CpuStorage::U8(data), l1.shape().clone()))
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
        }
    }
}

custom_unary_op!(Acos, "acos", acos, (BF16, F16, F32, F64));
custom_unary_op!(Asin, "asin", asin, (BF16, F16, F32, F64));
custom_unary_op!(Atan, "atan", atan, (BF16, F16, F32, F64));
custom_unary_op!(Cbrt, "cbrt", cbrt, (BF16, F16, F32, F64));
custom_unary_op!(Ceil, "ceil", ceil, (BF16, F16, F32, F64));
custom_unary_op!(Floor, "floor", floor, (BF16, F16, F32, F64));
custom_unary_op!(Log1p, "ln_1p", ln_1p, (BF16, F16, F32, F64));
custom_unary_op!(Round, "round", round, (BF16, F16, F32, F64));
custom_unary_op!(Tan, "tan", tan, (BF16, F16, F32, F64));
custom_unary_bool_op!(IsInf, "is_inf", is_infinite, (F32, F64));
custom_unary_op_closure!(BitNot, "bit_not", |v| !v, (U8, U32, I64));
custom_unary_op_closure!(ErfInv, "erf_inv", |v| erf_inv(v), (F64));

custom_binary_op!(BitAnd, "bit_and", |v1, v2| v1 & v2, (U32, I64));
custom_binary_op!(BitOr, "bit_or", |v1, v2| v1 | v2, (U32, I64));
custom_binary_op!(BitXor, "bit_xor", |v1, v2| v1 ^ v2, (U32, I64));
custom_binary_op!(Shl, "shl", |v1, v2| v1 << v2, (U32, I64));
custom_binary_op!(Shr, "shr", |v1, v2| v1 >> v2, (U32, I64));
custom_binary_bool_op!(
    LogicalOr,
    "logical_or",
    |v1, v2| if v1 as i8 == 0 && v2 as i8 == 0 { 0 } else { 1 },
    (U8, U32, I64, F32, F64)
);
