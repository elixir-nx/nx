target = System.get_env("EXLA_TARGET", "host")
client = EXLAHelpers.client()

if System.get_env("DEBUG") in ["1", "true"] do
  dbg(System.schedulers_online())
  IO.gets("Press enter to continue... -- PID: #{System.pid()}")
end

runtime = if System.get_env("EXLA_RUNTIME") == "iree", do: :iree, else: :xla

Nx.Defn.global_default_options(compiler: EXLA, runtime: runtime)

exclude_multi_device =
  if client.device_count > 1 and client.platform == :host, do: [], else: [:multi_device]

exclude =
  case client.platform do
    :tpu -> [:unsupported_dilated_window_reduce, :unsupported_64_bit_op]
    :host -> []
    _ -> [:conditional_inside_map_reduce]
  end

case {:os.type(), List.to_string(:erlang.system_info(:system_architecture))} do
  {{:unix, :darwin}, "aarch64" <> _} ->
    Application.put_env(:exla, :is_mac_arm, true)

  _ ->
    Application.put_env(:exla, :is_mac_arm, false)
end

if client.platform == :host and client.device_count == 1 and System.schedulers_online() > 1 do
  IO.puts(
    "To run multi-device tests: XLA_FLAGS=--xla_force_host_platform_device_count=#{System.schedulers_online()} mix test"
  )
end

cuda_required =
  if Map.has_key?(EXLA.Client.get_supported_platforms(), :cuda) do
    []
  else
    [:cuda_required]
  end

compiler_mode = :iree

iree_excludes =
  if compiler_mode == :iree do
    [
      :token,
      :iree_hangup_error,
      :iree_type_mismatch_error,
      :iree_resource_exhausted_error,
      :iree_key_not_found_error,
      :iree_wrong_result_error,
      :iree_unsupported_fft_error,
      :iree_segfault_error,
      :iree_illegal_op_error,
      :iree_offset_error,
      :multi_device
    ]
  else
    []
  end

ExUnit.start(
  exclude:
    [:platform, :integration] ++ exclude_multi_device ++ exclude ++ cuda_required ++ iree_excludes,
  include: [platform: String.to_atom(target)],
  assert_receive_timeout: 1000
)
