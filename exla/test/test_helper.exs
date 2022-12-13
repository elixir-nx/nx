target = System.get_env("EXLA_TARGET", "host")
client = EXLAHelpers.client()
Nx.Defn.global_default_options(compiler: EXLA)

multi_device =
  if client.device_count > 1 and client.platform == :host, do: [], else: [:multi_device]

skip_tpu_tests =
  if client.platform == :tpu,
    do: [:unsupported_dilated_window_reduce, :unsupported_64_bit_op],
    else: []

if client.platform == :host and client.device_count == 1 and System.schedulers_online() > 1 do
  IO.puts(
    "To run multi-device tests: XLA_FLAGS=--xla_force_host_platform_device_count=#{System.schedulers_online()} mix test"
  )
end

ExUnit.start(
  exclude: [:platform, :integration] ++ multi_device ++ skip_tpu_tests,
  include: [platform: String.to_atom(target)],
  assert_receive_timeout: 1000
)
