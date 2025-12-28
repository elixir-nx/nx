default_device = System.get_env("TORCHX_DEFAULT_DEVICE", "cpu") |> String.to_existing_atom()

Application.put_env(:nx, :default_backend, {Torchx.Backend, device: default_device})

skip_apple_arm64 = Application.get_env(:torchx, :is_apple_arm64)

# MPS doesn't handle concurrent command encoders well
# Tests must run synchronously to avoid GPU framework crashes
mps_opts =
  if default_device == :mps do
    [max_cases: 1]
  else
    []
  end

ExUnit.start(
  [exclude: [skip_apple_arm64: skip_apple_arm64, skip_on_mps: default_device == :mps]] ++
    mps_opts
)
