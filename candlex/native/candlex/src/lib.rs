mod error;
mod tensor;
mod candlex;

rustler::init! {
    "Elixir.Candlex.Native",
    [
        candlex::from_binary,
        candlex::to_binary
    ],
    load = candlex::load
}
