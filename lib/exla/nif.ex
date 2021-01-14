defmodule EXLA.NIF do
  @moduledoc false
  @on_load :__on_load__

  def __on_load__ do
    path = :filename.join(:code.priv_dir(:exla), 'libexla')
    :erlang.load_nif(path, 0)
  end

  def new_builder(_name),
    do: nif_error(__ENV__.function)

  def create_sub_builder(_builder, _name),
    do: nif_error(__ENV__.function)

  def get_shape_info(_ref), do: nif_error(__ENV__.function)

  def make_shape(_type, _dims),
    do: nif_error(__ENV__.function)

  def parameter(_builder, _number, _shape, _name),
    do: nif_error(__ENV__.function)

  def add(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def subtract(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def multiply(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def divide(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def remainder(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def min(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def max(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def bitwise_and(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def bitwise_or(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def bitwise_xor(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def left_shift(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def right_shift_arithmetic(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def right_shift_logical(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def equal(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def not_equal(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def greater_equal(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def greater(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def less_equal(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def less(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def power(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def complex(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def arctan2(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def abs(_a), do: nif_error(__ENV__.function)
  def exp(_a), do: nif_error(__ENV__.function)
  def expm1(_a), do: nif_error(__ENV__.function)
  def floor(_a), do: nif_error(__ENV__.function)
  def ceil(_a), do: nif_error(__ENV__.function)
  def round(_a), do: nif_error(__ENV__.function)
  def log(_a), do: nif_error(__ENV__.function)
  def log1p(_a), do: nif_error(__ENV__.function)
  def logistic(_a), do: nif_error(__ENV__.function)
  def sign(_a), do: nif_error(__ENV__.function)
  def cos(_a), do: nif_error(__ENV__.function)
  def sin(_a), do: nif_error(__ENV__.function)
  def tanh(_a), do: nif_error(__ENV__.function)
  def real(_a), do: nif_error(__ENV__.function)
  def imag(_a), do: nif_error(__ENV__.function)
  def sqrt(_a), do: nif_error(__ENV__.function)
  def rsqrt(_a), do: nif_error(__ENV__.function)
  def cbrt(_a), do: nif_error(__ENV__.function)
  def is_finite(_a), do: nif_error(__ENV__.function)
  def negate(_a), do: nif_error(__ENV__.function)
  def conj(_a), do: nif_error(__ENV__.function)
  def bitwise_not(_a), do: nif_error(__ENV__.function)
  def count_leading_zeros(_a), do: nif_error(__ENV__.function)
  def population_count(_a), do: nif_error(__ENV__.function)

  def dot(_a, _b, _precision),
    do: nif_error(__ENV__.function)

  def dot_general(_a, _b, _dims, _precision),
    do: nif_error(__ENV__.function)

  def conv_general_dilated(
        _operand,
        _kernel,
        _strides,
        _padding_config,
        _lhs_dilation,
        _rhs_dilation,
        _dimension_numbers,
        _precision_config
      ),
      do: nif_error(__ENV__.function)

  def transpose(_a, _permutation),
    do: nif_error(__ENV__.function)

  def conditional(_pred, _true_op, _true_comp, _false_op, _false_comp),
    do: nif_error(__ENV__.function)

  def conditional(_index, _branches, _operands),
    do: nif_error(__ENV__.function)

  def select(_pred, _on_true, _on_false),
    do: nif_error(__ENV__.function)

  def slice(_op, _start_indices, _limit_indices, _strides),
    do: nif_error(__ENV__.function)

  def slice_in_dim(_op, _start_index, _limit_index, _stride, _dimno),
    do: nif_error(__ENV__.function)

  def dynamic_slice(_op, _start_indices, _sizes),
    do: nif_error(__ENV__.function)

  def dynamic_update_slice(_op, _update, _start_indices),
    do: nif_error(__ENV__.function)

  def rng_normal(_mu, _sigma, _shape),
    do: nif_error(__ENV__.function)

  def rng_uniform(_a, _b, _shape),
    do: nif_error(__ENV__.function)

  def iota(_builder, _shape, _dim),
    do: nif_error(__ENV__.function)

  def reduce(_operand, _init_value, _computation, _dimensions),
    do: nif_error(__ENV__.function)

  def variadic_reduce(_builder, _operands, _init_value, _computation, _dimensions),
    do: nif_error(__ENV__.function)

  def reduce_window(
        _operand,
        _init_value,
        _computation,
        _window_dimensions,
        _window_strides,
        _padding_config
      ),
      do: nif_error(__ENV__.function)

  def reshape(_operand, _dimensions),
    do: nif_error(__ENV__.function)

  def broadcast_in_dim(_operand, _dimensions, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def pad(_operand, _value, _padding_config),
    do: nif_error(__ENV__.function)

  def get_shape(_builder, _operand),
    do: nif_error(__ENV__.function)

  def convert_element_type(_operand, _type),
    do: nif_error(__ENV__.function)

  def clamp(_operand, _min, _max),
    do: nif_error(__ENV__.function)

  def constant_r0(_builder, _value, _type),
    do: nif_error(__ENV__.function)

  def constant_from_binary(_builder, _data, _shape),
    do: nif_error(__ENV__.function)

  def reverse(_operand, _dimensions),
    do: nif_error(__ENV__.function)

  def tuple(_builder, _elements), do: nif_error(__ENV__.function)

  def get_tuple_element(_operand, _indeX), do: nif_error(__ENV__.function)

  def get_host_client(_num_replicas, _intra_op_parallelism_threads),
    do: nif_error(__ENV__.function)

  def get_cuda_client(_num_replicas, _intra_op_parallelism_threads),
    do: nif_error(__ENV__.function)

  def get_rocm_client(_num_replicas, _intra_op_parallelism_threads),
    do: nif_error(__ENV__.function)

  def get_supported_platforms, do: nif_error(__ENV__.function)

  def get_default_device_ordinal(_client),
    do: nif_error(__ENV__.function)

  def get_device_count(_client),
    do: nif_error(__ENV__.function)

  def build(_builder, _root),
    do: nif_error(__ENV__.function)

  def compile(
        _client,
        _computation,
        _argument_layouts,
        _device_ordinal,
        _num_replicas,
        _num_partitions,
        _use_spmd
      ),
      do: nif_error(__ENV__.function)

  def run_cpu(
        _client,
        _executable,
        _arguments,
        _device_ordinal,
        _run_id,
        _rng_seed,
        _launch_id,
        _replica,
        _partition,
        _keep_on_device
      ),
      do: nif_error(__ENV__.function)

  def run_io(
        _client,
        _executable,
        _arguments,
        _device_ordinal,
        _run_id,
        _rng_seed,
        _launch_id,
        _replica,
        _partition,
        _keep_on_device
      ),
      do: nif_error(__ENV__.function)

  def device_assignment_to_device_id(_exec, _replica, _partition), do: nif_error(__ENV__.function)

  def binary_to_device_mem(_client, _binary, _shape, _device_ordinal),
    do: nif_error(__ENV__.function)

  def read_device_mem(_client, _buffer),
    do: nif_error(__ENV__.function)

  def deallocate_device_mem(_buffer),
    do: nif_error(__ENV__.function)

  def start_log_sink(_sink_pid),
    do: nif_error(__ENV__.function)

  defp nif_error({name, arity}) do
    raise "failed to load implementation of #{inspect(__MODULE__)}.#{name}/#{arity}"
  end
end
