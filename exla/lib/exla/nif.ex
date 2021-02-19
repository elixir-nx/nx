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

  def make_tuple_shape(_shapes),
    do: nif_error(__ENV__.function)

  def parameter(_builder, _number, _shape, _name),
    do: nif_error(__ENV__.function)

  binary_broadcast_ops =
    [:add, :subtract, :multiply, :divide, :remainder, :min, :max] ++
      [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift_arithmetic] ++
      [:right_shift_logical, :equal, :not_equal, :greater_equal, :greater, :less_equal] ++
      [:less, :power, :complex, :arctan2]

  for op <- binary_broadcast_ops do
    def unquote(op)(_a, _b, _broadcast_dims) do
      nif_error(__ENV__.function)
    end
  end

  unary_ops =
    [:exp, :expm1, :log, :log1p, :logistic, :cos, :sin, :tanh, :real, :imag, :erf_inv] ++
      [:is_finite, :conj, :arccos, :arcsin, :arctan, :cosh, :sinh, :erf, :erfc] ++
      [:arccosh, :arcsinh, :arctanh, :sqrt, :rsqrt, :cbrt, :negate, :sign, :abs] ++
      [:bitwise_not, :population_count, :count_leading_zeros, :floor, :ceil, :round]

  for op <- unary_ops do
    def unquote(op)(_x) do
      nif_error(__ENV__.function)
    end
  end

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
        _window_dilations,
        _padding_config
      ),
      do: nif_error(__ENV__.function)

  def map(_builder, _operand, _computation, _dimensions),
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

  def concatenate(_builder, _operands, _dimension),
    do: nif_error(__ENV__.function)

  def sort(_operand, _comparator, _dimension),
    do: nif_error(__ENV__.function)

  def tuple(_builder, _elements), do: nif_error(__ENV__.function)

  def get_tuple_element(_operand, _index), do: nif_error(__ENV__.function)

  def cholesky(_operand), do: nif_error(__ENV__.function)

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
        _num_replicas,
        _num_partitions,
        _use_spmd
      ),
      do: nif_error(__ENV__.function)

  def run_cpu(
        _client,
        _executable,
        _arguments,
        _output_shape,
        _run_id,
        _rng_seed,
        _launch_id,
        _replica,
        _partition,
        _async_run,
        _keep_on_device
      ),
      do: nif_error(__ENV__.function)

  def run_io(
        _client,
        _executable,
        _arguments,
        _output_shape,
        _run_id,
        _rng_seed,
        _launch_id,
        _replica,
        _partition,
        _async_run,
        _keep_on_device
      ),
      do: nif_error(__ENV__.function)

  def await_streams_cpu(_client, _buffer, _keep_on_device),
    do: nif_error(__ENV__.function)

  def await_streams_io(_client, _buffer, _keep_on_device),
    do: nif_error(__ENV__.function)

  def compile_aot(_computation, _pbtext_path, _aot_path, _function_name, _class_name, _target_triple),
    do: nif_error(__ENV__.function)

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
