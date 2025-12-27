Application.put_env(:nx, :default_backend, {Torchx.Backend, device: :mps})

skip_apple_arm64 = Application.get_env(:torchx, :is_apple_arm64)

# Check if we're running on MPS device
skip_on_mps =
  case Application.get_env(:nx, :default_backend) do
    {Torchx.Backend, opts} -> Keyword.get(opts, :device) == :mps
    _ -> false
  end

# MPS doesn't handle concurrent command encoders well
# Tests must run synchronously to avoid GPU framework crashes
ExUnit.start(
  exclude: [skip_apple_arm64: skip_apple_arm64, skip_on_mps: skip_on_mps],
  max_cases: 1
)
