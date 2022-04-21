import Config

target = System.get_env("EXLA_TARGET", "host")

config :exla, :clients,
  cuda: [platform: :cuda, memory_fraction: 0.8],
  rocm: [platform: :rocm, memory_fraction: 0.8]

config :nx, :default_defn_options, compiler: EXLA, client: String.to_atom(target)

config :logger, :console,
  format: "\n$time [$level] $metadata $message\n",
  metadata: [:domain, :file, :line]
