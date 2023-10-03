#[rustfmt::skip]
pub const CUSTOM_UNARY: &str = include_str!(concat!(env!("OUT_DIR"), "/src/kernels//custom_unary.ptx"));
#[rustfmt::skip]
pub const CUSTOM_BINARY: &str = include_str!(concat!(env!("OUT_DIR"), "/src/kernels//custom_binary.ptx"));
