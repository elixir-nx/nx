defmodule Nx.Tensor do
  @moduledoc """
  The tensor data structure.

  `Nx.Tensor` is a generic container for multidimensional data structures.

  Its data field can be made by any struct that implements the behaviour
  defined in this module. The behaviour is mostly callback implementations
  for the functions in the `Nx` module where the tensor outgoing shape is
  given as first argument.

  `Nx` ships with two implementations of `Nx.Tensor`:

    * `Nx.BinaryTensor` - a binary tensor implementation with
      support for multiple devices

    * `Nx.Defn.Expr` - a tensor implementation that builds a
      graph of all invoked operations. Typically used in
      conjection with `defn` and custom compilers

  """

  @type data :: struct
  @type type :: Nx.Type.t()
  @type shape :: tuple()
  @type axis :: name | integer
  @type axes :: [axis]
  @type name :: atom
  @type t :: %Nx.Tensor{data: data, type: type, shape: shape, names: [name]}

  @enforce_keys [:type, :shape, :names]
  defstruct [:data, :type, :shape, :names]

  @callback to_binary(t) :: binary
  @callback device_read(t) :: t
  @callback device_deallocate(t) :: t
  @callback device_transfer(t, module, keyword) :: t

  @callback inspect(t, Inspect.Opts.t()) :: t
  @callback from_binary(out :: t, binary) :: t
  @callback as_type(out :: t, t) :: t
  @callback reshape(out :: t, t, shape) :: t
  @callback squeeze(out :: t, t, axes) :: t
  @callback broadcast(out :: t, t, shape, axes) :: t
  @callback transpose(out :: t, t, keyword) :: t
  @callback pad(out :: t, t, pad_value :: t, padding_config :: list()) :: t
  @callback reverse(out :: t, t, keyword) :: t
  @callback sort(out :: t, t, keyword) :: t

  @callback dot(out :: t, t, axes, t, axes) :: t
  @callback conv(out :: t, t, kernel :: t, keyword) :: t
  @callback clip(out :: t, t, min :: t, max :: t) :: t
  @callback slice(out :: t, t, list, list, list) :: t
  @callback concatenate(out :: t, t, keyword) :: t
  @callback select(out :: t, t, t, t) :: t

  @callback all?(out :: t, t, keyword) :: t
  @callback any?(out :: t, t, keyword) :: t
  @callback sum(out :: t, t, keyword) :: t
  @callback argmax(out :: t, t, keyword) :: t
  @callback argmin(out :: t, t, keyword) :: t
  @callback reduce(out :: t, t, acc :: t, keyword, fun) :: t
  @callback reduce_window(out :: t, t, acc :: t, list, keyword, fun) :: t
  @callback map(out :: t, t, fun) :: t

  @callback cholesky(out :: t, t) :: t

  binary_ops =
    [:add, :subtract, :multiply, :power, :remainder, :divide, :arctan2, :min, :max] ++
      [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift] ++
      [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal] ++
      [:logical_and, :logical_or, :logical_xor] ++
      [:outer]

  for binary_op <- binary_ops do
    @callback unquote(binary_op)(out :: t, t, t) :: t
  end

  unary_ops =
    Enum.map(Nx.Shared.unary_math_funs(), &elem(&1, 0)) ++
      [:abs, :bitwise_not, :ceil, :floor, :negate, :round, :sign] ++
      [:count_leading_zeros, :population_count]

  for unary_op <- unary_ops do
    @callback unquote(unary_op)(out :: t, t) :: t
  end

  defimpl Inspect do
    import Inspect.Algebra

    def inspect(tensor, opts) do
      inner = tensor.data.__struct__.inspect(tensor, opts)

      color("#Nx.Tensor<", :map, opts)
      |> concat(nest(concat(line(), inner), 2))
      |> concat(color("\n>", :map, opts))
    end
  end
end
