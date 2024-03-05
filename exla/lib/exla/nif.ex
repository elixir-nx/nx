defmodule EXLA.NIF do
  @moduledoc false
  @on_load :__on_load__

  def __on_load__ do
    path = :filename.join(:code.priv_dir(:exla), ~c"libexla")
    :erlang.load_nif(path, 0)
  end

  def new_mlir_context, do: :erlang.nif_error(:undef)

  def new_mlir_module(_context), do: :erlang.nif_error(:undef)

  def create_mlir_function(_module, _name, _arg_types, _ret_type, _is_public),
    do: :erlang.nif_error(:undef)

  def get_mlir_function_arguments(_function), do: :erlang.nif_error(:undef)

  @bin_ops [:add, :subtract, :multiply, :divide, :pow, :min] ++
             [:max, :remainder, :atan2, :equal, :not_equal] ++
             [:less, :less_equal, :greater, :greater_equal] ++
             [:bitwise_and, :bitwise_or, :bitwise_xor] ++
             [:left_shift, :right_shift_arithmetic, :right_shift_logical]

  for op <- @bin_ops do
    mlir_op = :"mlir_#{op}"
    def unquote(mlir_op)(_function, _lhs, _rhs), do: :erlang.nif_error(:undef)
  end

  @unary_ops [:abs, :exp, :expm1, :floor, :ceil, :round] ++
               [:log, :log1p, :sigmoid, :sign, :cos] ++
               [:sin, :tan, :acos, :asin, :atan, :cosh, :sinh] ++
               [:tanh, :acosh, :asinh, :atanh, :sqrt, :cbrt] ++
               [:bitwise_not, :erf, :erfc, :erf_inv] ++
               [:is_infinity, :is_nan, :rsqrt, :negate, :count_leading_zeros] ++
               [:population_count, :real, :imag, :conjugate]

  for op <- @unary_ops do
    mlir_op = :"mlir_#{op}"
    def unquote(mlir_op)(_function, _operand), do: :erlang.nif_error(:undef)
  end

  def mlir_reshape(_function, _operand, _shape), do: :erlang.nif_error(:undef)
  def mlir_reverse(_function, _operand, _shape), do: :erlang.nif_error(:undef)
  def mlir_transpose(_function, _operand, _shape), do: :erlang.nif_error(:undef)
  def mlir_slice(_function, _operand, _starts, _limits, _strides), do: :erlang.nif_error(:undef)
  def mlir_dynamic_slice(_function, _operand, _starts, _lengths), do: :erlang.nif_error(:undef)
  def mlir_pad(_function, _tensor, _pad, _low, _high, _mid), do: :erlang.nif_error(:undef)

  def mlir_reduce(_function, _reducer, _init_values, _inputs, _dimensions),
    do: :erlang.nif_error(:undef)

  def mlir_window_reduce(
        _function,
        _reducer,
        _init_values,
        _inputs,
        _window_dimensions,
        _window_strides,
        _input_dilations,
        _window_dilations,
        _padding
      ),
      do: :erlang.nif_error(:undef)

  def mlir_map(_function, _mapper, _inputs, _dimensions),
    do: :erlang.nif_error(:undef)

  def mlir_if(_function, _pred, _output_shape),
    do: :erlang.nif_error(:undef)

  def mlir_push_region(_function, _region), do: :erlang.nif_error(:undef)

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

  def mlir_convert(_function, _tensor, _type), do: :erlang.nif_error(:undef)
  def mlir_bitcast_convert(_function, _tensor, _type, _dims), do: :erlang.nif_error(:undef)
  def mlir_top_k(_function, _tensor, _k), do: :erlang.nif_error(:undef)
  def mlir_sort(_function, _tensors, _dim, _comparator, _stable), do: :erlang.nif_error(:undef)

  def mlir_get_shape(_tensor), do: :erlang.nif_error(:undef)

  def dump_mlir_module(_builder), do: :erlang.nif_error(:undef)

  def mlir_iota(_function, _shape, _dim), do: :erlang.nif_error(:undef)
  def mlir_constant_r0(_function, _value, _type), do: :erlang.nif_error(:undef)
  def mlir_constant_from_binary(_function, _data, _type, _dims), do: :erlang.nif_error(:undef)

  def mlir_dot_general(_function, _shape, _lhs, _rhs, _dims, _precision),
    do: :erlang.nif_error(:undef)

  def mlir_broadcast_in_dim(_function, _shape, _operand, _axes), do: :erlang.nif_error(:undef)
  def mlir_concatenate(_function, _operands, _dimension), do: :erlang.nif_error(:undef)
  def mlir_optimization_barrier(_function, _operand), do: :erlang.nif_error(:undef)
  def mlir_clamp(_function, _operand, _min, _max), do: :erlang.nif_error(:undef)

  def mlir_select(_function, _pred, _on_true, _on_false),
    do: :erlang.nif_error(:undef)

  def mlir_scatter(
        _function,
        _target,
        _indices,
        _updates,
        _add_or_put,
        _indices_rank,
        _update_window_dims,
        _inserted_window_dims,
        _index_axes_to_target_axes
      ),
      do: :erlang.nif_error(:undef)

  def mlir_select_and_scatter(
        _function,
        _target,
        _source,
        _init_value,
        _gt_or_lt,
        _window_dimensions,
        _window_strides,
        _padding
      ),
      do: :erlang.nif_error(:undef)

  def mlir_gather(
        _function,
        _sorce,
        _indices,
        _slice_sizes,
        _offset_dims,
        _collapsed_slice_dims,
        _start_index_map,
        _index_vector_dim
      ),
      do: :erlang.nif_error(:undef)

  def mlir_fft(_function, _tensor, _forward_fft, _fft_lenght), do: :erlang.nif_error(:undef)

  def mlir_convolution(
        _function,
        _tensor,
        _kernel,
        _strides,
        _padding_config,
        _tensor_dilation,
        _kernel_dilation,
        _dimension_numbers,
        _feature_group_count,
        _batch_group_count,
        _precision_config,
        _output_dims
      ),
      do: :erlang.nif_error(:undef)

  def mlir_create_token(_function), do: :erlang.nif_error(:undef)

  def mlir_triangular_solve(_function, _a, _b, _left_side, _lower, _transpose_a),
    do: :erlang.nif_error(:undef)

  def mlir_dynamic_update_slice(_function, _operand, _updates, _starts),
    do: :erlang.nif_error(:undef)

  def mlir_infeed(_function, _token, _shape), do: :erlang.nif_error(:undef)
  def mlir_outfeed(_function, _token, _inputs), do: :erlang.nif_error(:undef)

  def mlir_call(_function, _args, _computation), do: :erlang.nif_error(:undef)
  def mlir_while(_function, _initial), do: :erlang.nif_error(:undef)
  def mlir_return(_function, _operands), do: :erlang.nif_error(:undef)

  def mlir_qr(_function, _operand, _q_shape, _r_shape), do: :erlang.nif_error(:undef)

  def get_shape_info(_ref), do: :erlang.nif_error(:undef)

  def make_shape(_type, _dims),
    do: :erlang.nif_error(:undef)

  def make_token_shape(),
    do: :erlang.nif_error(:undef)

  def make_tuple_shape(_shapes),
    do: :erlang.nif_error(:undef)

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

  def binary_to_device_mem(_client, _binary, _shape, _device_ordinal),
    do: :erlang.nif_error(:undef)

  def read_device_mem(_buffer, _size),
    do: :erlang.nif_error(:undef)

  def deallocate_device_mem(_buffer),
    do: :erlang.nif_error(:undef)

  def transfer_to_infeed(_client, _device, _data_shapes),
    do: :erlang.nif_error(:undef)

  def transfer_from_outfeed(_client, _device, _shapes, _pid, _ref),
    do: :erlang.nif_error(:undef)

  def copy_buffer_to_device(_client, _buffer, _device),
    do: :erlang.nif_error(:undef)

  def start_log_sink(_sink_pid),
    do: :erlang.nif_error(:undef)

  def get_c_api_client(_device_type), do: :erlang.nif_error(:undef)

  def load_pjrt_plugin(_device_type, _library_path), do: :erlang.nif_error(:undef)
end
