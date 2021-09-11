defmodule EXLA.NIF do
  @moduledoc false
  @on_load :__on_load__

  def __on_load__ do
    path = :filename.join(:code.priv_dir(:exla), 'libexla')
    :erlang.load_nif(path, 0)
  end

  def new_builder(_name),
    do: :erlang.nif_error(:undef)

  def create_sub_builder(_builder, _name),
    do: :erlang.nif_error(:undef)

  def get_shape_info(_ref), do: :erlang.nif_error(:undef)

  def make_shape(_type, _dims),
    do: :erlang.nif_error(:undef)

  def make_token_shape(),
    do: :erlang.nif_error(:undef)

  def make_tuple_shape(_shapes),
    do: :erlang.nif_error(:undef)

  def parameter(_builder, _number, _shape, _name),
    do: :erlang.nif_error(:undef)

  binary_broadcast_ops =
    [:add, :subtract, :multiply, :divide, :remainder, :min, :max] ++
      [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift_arithmetic] ++
      [:right_shift_logical, :equal, :not_equal, :greater_equal, :greater, :less_equal] ++
      [:less, :power, :complex, :atan2]

  for op <- binary_broadcast_ops do
    def unquote(op)(_a, _b, _broadcast_dims) do
      :erlang.nif_error(:undef)
    end
  end

  unary_ops =
    [:exp, :expm1, :log, :log1p, :logistic, :cos, :sin, :tanh, :real, :imag, :erf_inv] ++
      [:is_finite, :conj, :acos, :asin, :atan, :cosh, :sinh, :erf, :erfc] ++
      [:acosh, :asinh, :atanh, :sqrt, :rsqrt, :cbrt, :negate, :sign, :abs] ++
      [:bitwise_not, :population_count, :count_leading_zeros, :floor, :ceil, :round]

  for op <- unary_ops do
    def unquote(op)(_x) do
      :erlang.nif_error(:undef)
    end
  end

  def dot(_a, _b, _precision),
    do: :erlang.nif_error(:undef)

  def dot_general(_a, _b, _dims, _precision),
    do: :erlang.nif_error(:undef)

  def conv_general_dilated(
        _operand,
        _kernel,
        _strides,
        _padding_config,
        _lhs_dilation,
        _rhs_dilation,
        _dimension_numbers,
        _feature_group_count,
        _batch_group_count,
        _precision_config
      ),
      do: :erlang.nif_error(:undef)

  def transpose(_a, _permutation),
    do: :erlang.nif_error(:undef)

  def conditional(_pred, _true_op, _true_comp, _false_op, _false_comp),
    do: :erlang.nif_error(:undef)

  def conditional(_index, _branches, _operands),
    do: :erlang.nif_error(:undef)

  def select(_pred, _on_true, _on_false),
    do: :erlang.nif_error(:undef)

  def slice(_op, _start_indices, _limit_indices, _strides),
    do: :erlang.nif_error(:undef)

  def dynamic_slice(_op, _start_indices, _sizes),
    do: :erlang.nif_error(:undef)

  def dynamic_update_slice(_op, _update, _start_indices),
    do: :erlang.nif_error(:undef)

  def gather(
        _op,
        _start_indices,
        _index_vector_dim,
        _slice_sizes,
        _offset_dims,
        _collapsed_slice_dims,
        _start_index_map
      ),
      do: :erlang.nif_error(:undef)

  def rng_normal(_mu, _sigma, _shape),
    do: :erlang.nif_error(:undef)

  def rng_uniform(_a, _b, _shape),
    do: :erlang.nif_error(:undef)

  def iota(_builder, _shape, _dim),
    do: :erlang.nif_error(:undef)

  def reduce(_operand, _init_value, _computation, _dimensions),
    do: :erlang.nif_error(:undef)

  def variadic_reduce(_builder, _operands, _init_value, _computation, _dimensions),
    do: :erlang.nif_error(:undef)

  def reduce_window(
        _operand,
        _init_value,
        _computation,
        _window_dimensions,
        _window_strides,
        _window_dilations,
        _padding_config
      ),
      do: :erlang.nif_error(:undef)

  def select_and_scatter(
        _operand,
        _select_fn,
        _window_dimensions,
        _window_strides,
        _padding_config,
        _source,
        _init_value,
        _scatter_fn
      ),
      do: :erlang.nif_error(:undef)

  def map(_builder, _operand, _computation, _dimensions),
    do: :erlang.nif_error(:undef)

  def while(_cond_fn, _body_fn, _init_value),
    do: :erlang.nif_error(:undef)

  def reshape(_operand, _dimensions),
    do: :erlang.nif_error(:undef)

  def broadcast_in_dim(_operand, _dimensions, _broadcast_dims),
    do: :erlang.nif_error(:undef)

  def pad(_operand, _value, _padding_config),
    do: :erlang.nif_error(:undef)

  def get_shape(_builder, _operand),
    do: :erlang.nif_error(:undef)

  def convert_element_type(_operand, _type),
    do: :erlang.nif_error(:undef)

  def bitcast_convert_type(_operand, _type),
    do: :erlang.nif_error(:undef)

  def clamp(_operand, _min, _max),
    do: :erlang.nif_error(:undef)

  def constant_r0(_builder, _value, _type),
    do: :erlang.nif_error(:undef)

  def constant_from_binary(_builder, _data, _shape),
    do: :erlang.nif_error(:undef)

  def reverse(_operand, _dimensions),
    do: :erlang.nif_error(:undef)

  def concatenate(_builder, _operands, _dimension),
    do: :erlang.nif_error(:undef)

  def sort(_operand, _comparator, _dimension),
    do: :erlang.nif_error(:undef)

  def variadic_sort(_operands, _comparator, _dimension),
    do: :erlang.nif_error(:undef)

  def tuple(_builder, _elements), do: :erlang.nif_error(:undef)

  def get_tuple_element(_operand, _index), do: :erlang.nif_error(:undef)

  def cholesky(_operand), do: :erlang.nif_error(:undef)

  def eigh(_operand, _lower), do: :erlang.nif_error(:undef)

  def lu(_operand), do: :erlang.nif_error(:undef)

  def qr(_operand, _full_matrices), do: :erlang.nif_error(:undef)

  def svd(_a, _precision_config), do: :erlang.nif_error(:undef)

  def triangular_solve(_a, _b, _left_side, _lower, _unit_diagonal, _transpose_a),
    do: :erlang.nif_error(:undef)

  def infeed(_token, _shape), do: :erlang.nif_error(:undef)

  def outfeed(_operand, _token, _shape), do: :erlang.nif_error(:undef)

  def create_token(_builder), do: :erlang.nif_error(:undef)

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

  def get_devices(_client),
    do: :erlang.nif_error(:undef)

  def build(_builder, _root),
    do: :erlang.nif_error(:undef)

  def compile(
        _client,
        _computation,
        _argument_layouts,
        _num_replicas,
        _num_partitions,
        _use_spmd,
        _device_id
      ),
      do: :erlang.nif_error(:undef)

  def run_cpu(
        _client,
        _executable,
        _arguments,
        _keep_on_device,
        _device_id
      ),
      do: :erlang.nif_error(:undef)

  def run_io(
        _client,
        _executable,
        _arguments,
        _keep_on_device,
        _device_id
      ),
      do: :erlang.nif_error(:undef)

  def binary_to_device_mem(_client, _binary, _shape, _device_ordinal),
    do: :erlang.nif_error(:undef)

  def read_device_mem(_client, _buffer),
    do: :erlang.nif_error(:undef)

  def deallocate_device_mem(_buffer),
    do: :erlang.nif_error(:undef)

  def transfer_to_infeed(_client, _device, _data, _shape),
    do: :erlang.nif_error(:undef)

  def transfer_from_outfeed(_client, _device, _shape),
    do: :erlang.nif_error(:undef)

  def start_log_sink(_sink_pid),
    do: :erlang.nif_error(:undef)
end
