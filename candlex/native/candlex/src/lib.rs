mod candlex;
mod error;

rustler::init! {
    "Elixir.Candlex.Native",
    [
        candlex::from_binary,
        candlex::to_binary
    ],
    load = candlex::load
}
