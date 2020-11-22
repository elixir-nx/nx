defmodule Exla.NIF do
  @moduledoc false
  @on_load :__on_load__

  def __on_load__ do
    path = :filename.join(:code.priv_dir(:exla), 'libexla')
    :erlang.load_nif(path, 0)
  end

  def new_builder(_name),
    do: nif_error(__ENV__.function)

  def create_sub_builder(_builder, _name), do: nif_error(__ENV__.function)

  def binary_to_shaped_buffer(_client, _binary, _shape, _device_ordinal),
    do: nif_error(__ENV__.function)

  def on_host_shape(_buffer),
    do: nif_error(__ENV__.function)

  def make_shape(_type, _dims),
    do: nif_error(__ENV__.function)

  def make_scalar_shape(_type),
    do: nif_error(__ENV__.function)

  def human_string(_shape),
    do: nif_error(__ENV__.function)

  def create_r0(_value),
    do: nif_error(__ENV__.function)

  def literal_to_string(_literal),
    do: nif_error(__ENV__.function)

  def parameter(_builder, _number, _shape, _name),
    do: nif_error(__ENV__.function)

  def add(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def sub(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def mul(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def div(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def rem(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def min(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def max(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def logical_and(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def logical_or(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def logical_xor(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def shift_left(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def shift_right_arithmetic(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def shift_right_logical(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def eq(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def eq_total_order(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def ne(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def ne_total_order(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def ge(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def ge_total_order(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def gt(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def gt_total_order(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def le(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def le_total_order(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def lt(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def lt_total_order(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def pow(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def complex(_a, _b, _broadcast_dims),
    do: nif_error(__ENV__.function)

  def atan2(_a, _b, _broadcast_dims),
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
  def clz(_a), do: nif_error(__ENV__.function)
  def cos(_a), do: nif_error(__ENV__.function)
  def sin(_a), do: nif_error(__ENV__.function)
  def tanh(_a), do: nif_error(__ENV__.function)
  def real(_a), do: nif_error(__ENV__.function)
  def imag(_a), do: nif_error(__ENV__.function)
  def sqrt(_a), do: nif_error(__ENV__.function)
  def rsqrt(_a), do: nif_error(__ENV__.function)
  def cbrt(_a), do: nif_error(__ENV__.function)
  def is_finite(_a), do: nif_error(__ENV__.function)
  def logical_not(_a), do: nif_error(__ENV__.function)
  def neg(_a), do: nif_error(__ENV__.function)
  def conj(_a), do: nif_error(__ENV__.function)
  def copy(_a), do: nif_error(__ENV__.function)

  def population_count(_a),
    do: nif_error(__ENV__.function)

  def dot(_a, _b),
    do: nif_error(__ENV__.function)

  def conditional(_pred, _true_op, _true_comp, _false_op, _false_comp), do: nif_error(__ENV__.function)

  def slice(_op, _start_indices, _limit_indices, _strides),
    do: nif_error(__ENV__.function)

  def slice_in_dim(_op, _start_index, _limit_index, _stride, _dimno), do: nif_error(__ENV__.function)

  def dynamic_slice(_op, _start_indices, _sizes), do: nif_error(__ENV__.function)

  def dynamic_update_slice(_op, _update, _start_indices), do: nif_error(__ENV__.function)

  def reduce(_operand, _init_value, _computation, _dimensions), do: nif_error(__ENV__.function)

  def zero(_builder, _dtype), do: nif_error(__ENV__.function)

  def constant_r0(_builder, _value),
    do: nif_error(__ENV__.function)

  def constant_r1(_builder, _length, _value),
    do: nif_error(__ENV__.function)

  def get_cpu_client(_num_replicas, _intra_op_parallelism_threads), do: nif_error(__ENV__.function)

  def get_gpu_client(_num_replicas, _intra_op_parallelism_threads), do: nif_error(__ENV__.function)

  def get_default_device_ordinal(_client),
    do: nif_error(__ENV__.function)

  def get_device_count(_client),
    do: nif_error(__ENV__.function)

  def get_computation_hlo_proto(_computation),
    do: nif_error(__ENV__.function)

  def get_computation_hlo_text(_computation),
    do: nif_error(__ENV__.function)

  def build(_builder, _root),
    do: nif_error(__ENV__.function)

  def compile(_client, _computation, _argument_layouts, _device_ordinal, _num_replicas, _num_partitions),
    do: nif_error(__ENV__.function)

  def run(_client, _executable, _arguments, _device_ordinal, _run_id, _rng_seed, _launch_id, _keep_on_device),
    do: nif_error(__ENV__.function)

  def literal_to_shaped_buffer(_client, _literal, _device_ordinal),
    do: nif_error(__ENV__.function)

  def shaped_buffer_to_literal(_client, _shaped_buffer),
    do: nif_error(__ENV__.function)

  def shaped_buffer_to_binary(_client, _shaped_buffer), do: nif_error(__ENV__.function)

  defp nif_error({name, arity}) do
    raise "failed to load implementation of #{inspect(__MODULE__)}.#{name}/#{arity}"
  end
end
