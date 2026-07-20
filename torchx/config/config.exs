import Config

is_apple_arm64 =
  :os.type() == {:unix, :darwin} and
    not List.starts_with?(:erlang.system_info(:system_architecture), ~c"x86_64")

config :torchx,
  add_backend_on_inspect: config_env() != :test,
  check_shape_and_type: config_env() == :test,
  is_apple_arm64: is_apple_arm64,
  # Optional override for the Windows libtorch DLL directory passed to the
  # DLL loader NIF. Defaults to Application.app_dir(:torchx, "priv/libtorch").
  libtorch_dll_dir: System.get_env("LIBTORCH_DLL_DIR")
