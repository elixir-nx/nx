Application.put_env(:nx, :default_backend, Torchx.Backend)

skip_apple_arm64 = Application.get_env(:torchx, :is_apple_arm64)

ExUnit.start(exclude: [skip_apple_arm64: skip_apple_arm64])
