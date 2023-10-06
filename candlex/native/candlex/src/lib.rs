mod atoms {
    rustler::atoms! {
        cpu,
        cuda
    }
}

mod devices;
mod error;
#[cfg(feature = "cuda")]
mod kernels;
mod ops;
mod tensors;

use rustler::{Env, Term};
use tensors::TensorRef;

fn load(env: Env, _info: Term) -> bool {
    rustler::resource!(TensorRef, env);
    true
}

rustler::init! {
    "Elixir.Candlex.Native",
    [
        tensors::from_binary,
        tensors::to_binary,
        tensors::add,
        tensors::atan2,
        tensors::subtract,
        tensors::multiply,
        tensors::divide,
        tensors::quotient,
        tensors::remainder,
        tensors::pow,
        tensors::max,
        tensors::min,
        tensors::equal,
        tensors::not_equal,
        tensors::greater,
        tensors::greater_equal,
        tensors::less,
        tensors::less_equal,
        tensors::all,
        tensors::sum,
        tensors::dtype,
        tensors::t_shape,
        tensors::argmax,
        tensors::argmin,
        tensors::reduce_max,
        tensors::reduce_min,
        tensors::negate,
        tensors::where_cond,
        tensors::narrow,
        tensors::gather,
        tensors::index_select,
        tensors::index_add,
        tensors::chunk,
        tensors::squeeze,
        tensors::clamp,
        tensors::arange,
        tensors::to_type,
        tensors::broadcast_to,
        tensors::reshape,
        tensors::concatenate,
        tensors::conv1d,
        tensors::conv2d,
        tensors::permute,
        tensors::slice_scatter,
        tensors::pad_with_zeros,
        tensors::matmul,
        tensors::abs,
        tensors::acos,
        tensors::acosh,
        tensors::asin,
        tensors::asinh,
        tensors::atan,
        tensors::atanh,
        tensors::cbrt,
        tensors::ceil,
        tensors::cos,
        tensors::cosh,
        tensors::sigmoid,
        tensors::sign,
        tensors::sin,
        tensors::sinh,
        tensors::erf,
        tensors::erfc,
        tensors::erf_inv,
        tensors::exp,
        tensors::expm1,
        tensors::floor,
        tensors::is_infinity,
        tensors::is_nan,
        tensors::round,
        tensors::log,
        tensors::log1p,
        tensors::rsqrt,
        tensors::sqrt,
        tensors::tan,
        tensors::tanh,
        tensors::bitwise_not,
        tensors::bitwise_and,
        tensors::bitwise_or,
        tensors::bitwise_xor,
        tensors::logical_and,
        tensors::logical_or,
        tensors::logical_xor,
        tensors::left_shift,
        tensors::right_shift,
        tensors::to_device,
        devices::is_cuda_available
    ],
    load = load
}
