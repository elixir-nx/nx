import Config

config :exla, :clients,
  default: [platform: :host],
  cuda: [platform: :cuda]
