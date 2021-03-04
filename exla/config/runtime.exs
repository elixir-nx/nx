import Config

target = System.get_env("EXLA_TARGET", "host")

config :exla, :clients, default: [platform: String.to_atom(target), memory_fraction: 0.8]

config :logger, :console,
  format: "\n$time [$level] $metadata $levelpad$message\n",
  metadata: [:domain, :file, :line]
