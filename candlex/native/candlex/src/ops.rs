use candle_core::{CpuStorage, CustomOp1, Error, Layout, Shape};
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

custom_unary_op!(Acos, "acos", acos);
custom_unary_op!(Asin, "asin", asin);
custom_unary_op!(Atan, "atan", atan);
custom_unary_op!(Cbrt, "cbrt", cbrt);
custom_unary_op!(Ceil, "ceil", ceil);
custom_unary_op!(Floor, "floor", floor);
custom_unary_op!(Tan, "tan", tan);
