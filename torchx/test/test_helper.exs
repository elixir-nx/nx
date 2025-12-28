default_device = System.get_env("TORCHX_DEFAULT_DEVICE", "cpu") |> String.to_existing_atom()

Application.put_env(:nx, :default_backend, {Torchx.Backend, device: default_device})

skip_apple_arm64 = Application.get_env(:torchx, :is_apple_arm64)
device_is_mps = default_device == :mps

ExUnit.start(
  exclude: [
    skip_apple_arm64: skip_apple_arm64,
    mps_f64_not_supported: device_is_mps,
    mps_round_difference: device_is_mps
  ]
)
