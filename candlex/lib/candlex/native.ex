defmodule Candlex.Native do
  @moduledoc false

  use Rustler, otp_app: :candlex, crate: "candlex"

  # Rustler will override all the below stub functions with real NIFs
  def from_binary(_binary, _dtype, _shape, _device), do: error()
  def to_binary(_tensor), do: error()
  def all(_tensor), do: error()
  def where_cond(_tensor, _on_true, _on_false), do: error()
  def narrow(_tensor, _dim, _start, _length), do: error()
  def squeeze(_tensor, _dim), do: error()
  def arange(_start, _end, _dtype, _shape, _device), do: error()
  def broadcast_to(_tensor, _shape), do: error()
  def reshape(_tensor, _shape), do: error()
  def to_type(_tensor, _dtype), do: error()
  def concatenate(_tensors, _axis), do: error()

  for op <- [
        :abs,
        :acos,
        :asin,
        :atan,
        :ceil,
        :cos,
        :exp,
        :floor,
        :log,
        :negate,
        :sin,
        :sqrt,
        :tan,
        :tanh
      ] do
    def unquote(op)(_tensor), do: error()
  end

  for op <- [
        :add,
        :equal,
        :greater_equal,
        :less,
        :less_equal,
        :matmul,
        :max,
        :min,
        :multiply,
        :subtract
      ] do
    def unquote(op)(_left, _right), do: error()
  end

  for op <- [:argmax, :argmin] do
    def unquote(op)(_tensor, _dim, _keep_dim), do: error()
  end

  def is_cuda_available(), do: error()

  defp error(), do: :erlang.nif_error(:nif_not_loaded)
end
