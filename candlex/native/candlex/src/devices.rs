#[rustler::nif(schedule = "DirtyCpu")]
pub fn is_cuda_available() -> bool {
    candle_core::utils::cuda_is_available()
}
