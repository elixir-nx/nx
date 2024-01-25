target = System.get_env("EXLA_TARGET", "host")
client = EXLAHelpers.client()

compiler_mode =
  case System.get_env("EXLA_COMPILER_MODE", "xla") do
    "xla" -> :xla
    "mlir" -> :mlir
    _ -> raise "Invalid EXLA_COMPILER_MODE"
  end

Nx.Defn.global_default_options(compiler: EXLA, compiler_mode: compiler_mode)

exclude_multi_device =
  if client.device_count > 1 and client.platform == :host, do: [], else: [:multi_device]

exclude =
  case client.platform do
    :tpu -> [:unsupported_dilated_window_reduce, :unsupported_64_bit_op]
    :host -> []
    _ -> [:conditional_inside_map_reduce]
  end

exclude =
  if compiler_mode == :mlir do
    exclude ++
      [
        :mlir_linalg_nor_supported_yet,
        :mlir_token_error,
        :mlir_cond_inside_while,
        :mlir_vectorization
      ]
  else
    exclude
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
