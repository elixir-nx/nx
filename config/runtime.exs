import Config

target = System.get_env("EXLA_TARGET", "host")

config :exla, :clients, cuda: [platform: :cuda],
                        host: [platform: :host]
