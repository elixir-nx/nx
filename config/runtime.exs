import Config

target = System.get_env("EXLA_TARGET", "host")

config :exla, :clients, default: [platform: String.to_atom(target)]
