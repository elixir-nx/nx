target = System.get_env("EXLA_TARGET", "host")
client = EXLAHelpers.client()

multi_device =
  if client.device_count > 1 and client.platform == :host, do: [], else: [:multi_device]

skip_tpu_tests =
  if client.platform == :tpu,
    do: [:unsupported_dilated_reduce_window, :unsupported_64_bit_op],
    else: []

if client.platform == :host and client.device_count == 1 do
  cores = System.schedulers_online()

  IO.puts(
    "To run multi-device tests, set XLA_FLAGS=--xla_force_host_platform_device_count=#{cores}"
  )
end

ExUnit.start(
  exclude: [:platform] ++ multi_device ++ skip_tpu_tests,
  include: [platform: String.to_atom(target)],
  assert_receive_timeout: 1000
)
