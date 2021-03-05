import Config

config :torchx,
  add_backend_on_inspect: config_env() != :test,
  check_shape_and_type: config_env() == :test
