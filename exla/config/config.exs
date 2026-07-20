import Config

config :exla, :add_backend_on_inspect, config_env() != :test

config :logger, :default_formatter, metadata: [:test, :test_module, :file, :line]
