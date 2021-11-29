import Config

# This is a compile time config. We want this off by default
# but since config files are not imported, it is only set to
# true inside Nx.
config :nx, :verify_grad, true
