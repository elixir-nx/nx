mod error;
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
        tensors::multiply,
        tensors::equal,
        tensors::greater_equal,
        tensors::subtract,
        tensors::all,
        tensors::where_cond,
        tensors::narrow,
        tensors::squeeze,
        tensors::arange,
        tensors::to_type,
        tensors::broadcast_to
    ],
    load = load
}
