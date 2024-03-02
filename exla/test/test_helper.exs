target = System.get_env("EXLA_TARGET", "host")
client = EXLAHelpers.client()

if System.get_env("DEBUG") in ["1", "true"] do
  IO.gets("Press enter to continue... -- PID: #{System.pid()}")
end

Nx.Defn.global_default_options(compiler: EXLA)

exclude_multi_device =
  if client.device_count > 1 and client.platform == :host, do: [], else: [:multi_device]

exclude =
  case client.platform do
    :tpu -> [:unsupported_dilated_window_reduce, :unsupported_64_bit_op]
    :host -> []
    _ -> [:conditional_inside_map_reduce]
  end

if client.platform == :host and client.device_count == 1 and System.schedulers_online() > 1 do
  IO.puts(
    "To run multi-device tests: XLA_FLAGS=--xla_force_host_platform_device_count=#{System.schedulers_online()} mix test"
  )
end

ExUnit.start(
  exclude: [:platform, :integration] ++ exclude_multi_device ++ exclude,
  include: [platform: String.to_atom(target)],
  assert_receive_timeout: 1000
)
