import Config

# These are compile time configs. We want them off by default
# but since config files are not imported, they are only set to
# true inside Nx.
config :nx, :verify_grad, true
config :nx, :verify_binary_size, true

# config :logger,
#   # handle_otp_reports: true,
#   # handle_sasl_reports: true,
#   backends: [:console]
#   # sync_threshold: 50,
#   # translate_otp_reports: true

# config :logger, :console,
#   format: "$time $metadata[$level] $message\n",
#   metadata: [:node]
