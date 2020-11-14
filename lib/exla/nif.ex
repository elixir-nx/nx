defmodule Exla.NIF do
  @on_load :load_nifs

  app = Mix.Project.config()[:app]
  @doc false
  def load_nifs do
    path = :filename.join(:code.priv_dir(unquote(app)), 'libexla')
    :erlang.load_nif(path, 0)
  end

  defmacrop nif_error() do
    quote do
      {name, arity} = __ENV__.function
      raise "failed to load implementation of #{inspect(__MODULE__)}.#{name}/#{arity}"
    end
  end

  def new_builder(_name),
    do: nif_error()

  def binary_to_shaped_buffer(_client, _binary, _shape, _device_ordinal),
    do: nif_error()

  def on_host_shape(_buffer),
    do: nif_error()

  def make_shape(_type, _dims),
    do: nif_error()

  def make_scalar_shape(_type),
    do: nif_error()

  def human_string(_shape),
    do: nif_error()

  def create_r0(_value),
    do: nif_error()

  def literal_to_string(_literal),
    do: nif_error()

  def parameter(_builder, _number, _shape, _name),
    do: nif_error()

  def add(_a, _b, _broadcast_dims),
    do: nif_error()

  def sub(_a, _b, _broadcast_dims),
    do: nif_error()

  def mul(_a, _b, _broadcast_dims),
    do: nif_error()

  def div(_a, _b, _broadcast_dims),
    do: nif_error()

  def rem(_a, _b, _broadcast_dims),
    do: nif_error()

  def min(_a, _b, _broadcast_dims),
    do: nif_error()

  def max(_a, _b, _broadcast_dims),
    do: nif_error()

  def logical_and(_a, _b, _broadcast_dims),
    do: nif_error()

  def logical_or(_a, _b, _broadcast_dims),
    do: nif_error()

  def logical_xor(_a, _b, _broadcast_dims),
    do: nif_error()

  def shift_left(_a, _b, _broadcast_dims),
    do: nif_error()

  def shift_right_arithmetic(_a, _b, _broadcast_dims),
    do: nif_error()

  def shift_right_logical(_a, _b, _broadcast_dims),
    do: nif_error()

  def eq(_a, _b, _broadcast_dims),
    do: nif_error()

  def eq_total_order(_a, _b, _broadcast_dims),
    do: nif_error()

  def ne(_a, _b, _broadcast_dims),
    do: nif_error()

  def ne_total_order(_a, _b, _broadcast_dims),
    do: nif_error()

  def ge(_a, _b, _broadcast_dims),
    do: nif_error()

  def ge_total_order(_a, _b, _broadcast_dims),
    do: nif_error()

  def gt(_a, _b, _broadcast_dims),
    do: nif_error()

  def gt_total_order(_a, _b, _broadcast_dims),
    do: nif_error()

  def le(_a, _b, _broadcast_dims),
    do: nif_error()

  def le_total_order(_a, _b, _broadcast_dims),
    do: nif_error()

  def lt(_a, _b, _broadcast_dims),
    do: nif_error()

  def lt_total_order(_a, _b, _broadcast_dims),
    do: nif_error()

  def pow(_a, _b, _broadcast_dims),
    do: nif_error()

  def complex(_a, _b, _broadcast_dims),
    do: nif_error()

  def atan2(_a, _b, _broadcast_dims),
    do: nif_error()

  def abs(_a), do: nif_error()
  def exp(_a), do: nif_error()
  def expm1(_a), do: nif_error()
  def floor(_a), do: nif_error()
  def ceil(_a), do: nif_error()
  def round(_a), do: nif_error()
  def log(_a), do: nif_error()
  def log1p(_a), do: nif_error()
  def logistic(_a), do: nif_error()
  def sign(_a), do: nif_error()
  def clz(_a), do: nif_error()
  def cos(_a), do: nif_error()
  def sin(_a), do: nif_error()
  def tanh(_a), do: nif_error()
  def real(_a), do: nif_error()
  def imag(_a), do: nif_error()
  def sqrt(_a), do: nif_error()
  def rsqrt(_a), do: nif_error()
  def cbrt(_a), do: nif_error()
  def is_finite(_a), do: nif_error()
  def logical_not(_a), do: nif_error()
  def neg(_a), do: nif_error()
  def conj(_a), do: nif_error()
  def copy(_a), do: nif_error()

  def population_count(_a),
    do: nif_error()

  def dot(_a, _b),
    do: nif_error()

  def constant_r0(_builder, _value),
    do: nif_error()

  def constant_r1(_length, _value),
    do: nif_error()

  def get_or_create_local_client(_platform, _number_of_replicas, _intra_op_parallelism_threads),
    do: nif_error()

  def get_default_device_ordinal(_client),
    do: nif_error()

  def get_device_count(_client),
    do: nif_error()

  def get_computation_hlo_proto(_computation),
    do: nif_error()

  def get_computation_hlo_text(_computation),
    do: nif_error()

  def build(_builder, _root),
    do: nif_error()

  def compile(_client, _computation, _argument_layouts, _options),
    do: nif_error()

  def run(_executable, _arguments, _run_options),
    do: nif_error()

  def literal_to_shaped_buffer(_client, _literal, _device_ordinal, _allocator),
    do: nif_error()

  def shaped_buffer_to_literal(_client, _shaped_buffer),
    do: nif_error()
end
