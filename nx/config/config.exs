import Config

# These are compile time configs. We want them off by default
# but since config files are not imported, they are only set to
# true inside Nx.
config :nx, :verify_grad, true
config :nx, :verify_binary_size, true

# If set to true, shards and sharding stages will be
# inspected with their debug ids alongside their unique ref ids
config :nx, :debug_shards, true
