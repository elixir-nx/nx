import Config

config :exla, :add_backend_on_inspect, config_env() != :test
