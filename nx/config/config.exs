import Config

# These are compile time configs. We want them off by default
# but since config files are not imported, they are only set to
# true inside Nx.
config :nx, :verify_grad, true
config :nx, :verify_binary_size, true
