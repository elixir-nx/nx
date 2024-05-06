defmodule EXLA.NIF do
  @moduledoc false
  @on_load :__on_load__

  def __on_load__ do
    path = :filename.join(:code.priv_dir(:exla), ~c"libexla")
    :erlang.load_nif(path, 0)
  end

  def mlir_new_context, do: :erlang.nif_error(:undef)

  def mlir_new_module(_context), do: :erlang.nif_error(:undef)

  def mlir_create_function(_module, _name, _arg_types, _ret_type, _is_public),
    do: :erlang.nif_error(:undef)

  def mlir_get_function_arguments(_function), do: :erlang.nif_error(:undef)

  def mlir_op(_function, _op_name, _operands, _result_type, _attributes, _blocks),
    do: :erlang.nif_error(:undef)

  def mlir_push_region(_function, _arg_types),
    do: :erlang.nif_error(:undef)

  def mlir_pop_region(_function),
    do: :erlang.nif_error(:undef)

  def mlir_build(_function, _root), do: :erlang.nif_error(:undef)

  def mlir_compile(
        _client,
        _computation,
        _argument_layouts,
        _num_replicas,
        _num_partitions,
        _use_spmd,
        _device_id
      ),
      do: :erlang.nif_error(:undef)

  def mlir_get_typespec(_tensor), do: :erlang.nif_error(:undef)

  def mlir_module_to_string(_builder), do: :erlang.nif_error(:undef)

  def get_host_client(),
    do: :erlang.nif_error(:undef)

  def get_gpu_client(
        _memory_fraction,
        _preallocate
      ),
      do: :erlang.nif_error(:undef)

  def get_tpu_client(), do: :erlang.nif_error(:undef)

  def get_supported_platforms, do: :erlang.nif_error(:undef)

  def get_device_count(_client),
    do: :erlang.nif_error(:undef)

  def serialize_executable(_executable), do: :erlang.nif_error(:undef)
  def deserialize_executable(_client, _string), do: :erlang.nif_error(:undef)

  def run_cpu(
        _client,
        _executable,
        _arguments,
        _device_id
      ),
      do: :erlang.nif_error(:undef)

  def run_io(
        _client,
        _executable,
        _arguments,
        _device_id
      ),
      do: :erlang.nif_error(:undef)

  def get_buffer_device_pointer(_client, _buffer, _pointer_kind), do: :erlang.nif_error(:undef)

  def create_buffer_from_device_pointer(
        _client,
        _opaque_pointer,
        _pointer_kind,
        _typespec,
        _device_id
      ),
      do: :erlang.nif_error(:undef)

  def binary_to_device_mem(_client, _binary, _typespec, _device_ordinal),
    do: :erlang.nif_error(:undef)

  def read_device_mem(_buffer, _size),
    do: :erlang.nif_error(:undef)

  def deallocate_device_mem(_buffer),
    do: :erlang.nif_error(:undef)

  def transfer_to_infeed(_client, _device, _data_typespecs),
    do: :erlang.nif_error(:undef)

  def transfer_from_outfeed(_client, _device, _typespecs, _pid, _ref),
    do: :erlang.nif_error(:undef)

  def copy_buffer_to_device(_client, _buffer, _device),
    do: :erlang.nif_error(:undef)

  def start_log_sink(_sink_pid),
    do: :erlang.nif_error(:undef)

  def get_c_api_client(_device_type), do: :erlang.nif_error(:undef)

  def load_pjrt_plugin(_device_type, _library_path), do: :erlang.nif_error(:undef)
end
