import Config

target = System.get_env("EXLA_TARGET", "host")

config :exla, :clients, default: [platform: String.to_atom(target)]

# Certain XLA Ops support configuring precision accumulation
# specifically when dealing with TPUs and `bfloat16`
#
# This option can be adjusted without changes to higher level
# Nx calls
#
# The following are valid precision configurations:
#   - :default - uses mixed bfloat16 precision with F32 accumulation
#   - :high - uses multiple MXU passes for higher precision
#   - :highest - uses more MXU passes for full F32 precision
config :exla, :precision, :default
