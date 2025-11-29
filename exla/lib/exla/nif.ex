defmodule EXLA.NIF do
  @moduledoc false
  @on_load :__on_load__

  def __on_load__ do
    path = :filename.join(:code.priv_dir(:exla), ~c"libexla")

    case :erlang.load_nif(path, 0) do
      :ok ->
        :ok

      {:error, {reason, text}} ->
        raise """
        Failed to load NIF library.
        Follow the steps in the :exla README Troubleshooting section for more information.

        #{inspect(reason)}
        #{text}
        """
    end
  end

  def mlir_new_thread_pool(_concurrency), do: err!()
  def mlir_new_context(_thread_pool_ref), do: err!()
  def mlir_new_module(_context), do: err!()
  def mlir_create_function(_module, _name, _arg_types, _ret_type, _is_public), do: err!()
  def mlir_get_function_arguments(_function), do: err!()
  def mlir_op(_function, _op_name, _operands, _result_type, _attributes, _blocks), do: err!()
  def mlir_push_region(_function, _arg_types), do: err!()
  def mlir_pop_region(_function), do: err!()
  def mlir_build(_function, _root), do: err!()

  def mlir_compile(
        _client,
        _computation,
        _argument_layouts,
        _num_replicas,
        _num_partitions,
        _use_spmd,
        _device_id,
        _callback_server_pid
      ),
      do: err!()

  def mlir_get_typespec(_tensor), do: err!()
  def mlir_module_to_string(_builder), do: err!()

  def get_buffer_device_pointer(_client, _buffer, _pointer_kind), do: err!()

  def create_buffer_from_device_pointer(
        _client,
        _pointer_kind,
        _pointer_data,
        _typespec,
        _device_id
      ),
      do: err!()

  def binary_to_device_mem(_client, _binary, _typespec, _device_ordinal), do: err!()
  def read_device_mem(_buffer, _size), do: err!()
  def deallocate_device_mem(_buffer), do: err!()
  def transfer_to_infeed(_client, _device, _buffers, _typespecs), do: err!()
  def transfer_from_outfeed(_client, _device, _typespecs, _pid, _ref), do: err!()
  def copy_buffer_to_device(_client, _buffer, _device), do: err!()
  def get_host_client(), do: err!()
  def get_gpu_client(_memory_fraction, _preallocate), do: err!()
  def get_tpu_client(), do: err!()
  def get_c_api_client(_device_type), do: err!()
  def load_pjrt_plugin(_device_type, _library_path), do: err!()
  def get_device_count(_client), do: err!()
  def get_supported_platforms, do: err!()
  def run_cpu(_executable, _arguments, _device_id), do: err!()
  def run_io(_executable, _arguments, _device_id), do: err!()
  def serialize_executable(_executable), do: err!()
  def deserialize_executable(_client, _string), do: err!()
  def start_log_sink(_sink_pid), do: err!()
  def get_allocated_memory(_client), do: err!()
  def get_peak_memory(_client), do: err!()
  def reset_peak_memory(_client), do: err!()
  def get_per_device_memory(_client), do: err!()

  # Elixir callback bridge
  def start_runtime_callback_bridge(_dispatcher_pid), do: err!()
  def clear_runtime_callback_bridge(_dispatcher_pid), do: err!()
  def runtime_callback_reply(_reply_tag, _status, _result), do: err!()

  defp err!(), do: :erlang.nif_error(:undef)
end
