mod candlex;

rustler::init! {
    "Elixir.Candlex.Native",
    [
        candlex::scalar_tensor,
        candlex::from_binary,
        candlex::to_binary
    ],
    load = candlex::load
}
