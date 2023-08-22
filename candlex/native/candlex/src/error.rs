use rustler::{Encoder, Env, Term};
use thiserror::Error;

// Defines the atoms for each value of ExplorerError.
rustler::atoms! {
    candle,
}

#[derive(Error, Debug)]
pub enum CandlexError {
    #[error("Candle Error: {0}")]
    Candle(#[from] candle_core::Error)
}

impl Encoder for CandlexError {
    fn encode<'b>(&self, env: Env<'b>) -> Term<'b> {
        format!("{self}").encode(env)
    }
}
