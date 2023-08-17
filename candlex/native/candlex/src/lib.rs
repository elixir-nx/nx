mod candlex;

rustler::init! {
    "Elixir.Candlex.Native",
    [
        candlex::scalar_tensor,
        candlex::to_binary,
        candlex::from_binary
    ],
    load = candlex::load
}
