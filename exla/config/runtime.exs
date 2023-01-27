import Config

config :exla, :clients,
  cuda: [platform: :cuda, memory_fraction: 0.8],
  rocm: [platform: :rocm, memory_fraction: 0.8],
  other_host: [platform: :host]

config :exla, default_client: String.to_atom(System.get_env("EXLA_TARGET", "host"))

config :logger, :console,
  format: "\n$time [$level] $metadata $message\n",
  metadata: [:domain, :file, :line]
