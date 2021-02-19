defmodule Nx.Backend do
  @moduledoc """
  The behaviour for tensor backends.

  Each backend is module that defines a struct and implements the callbacks
  defined in this module. The callbacks are mostly implementations of the
  functions in the `Nx` module with the tensor output shape given as first
  argument.

  `Nx` backends come in two flavors: opaque backends, of which you should
  not access its data directly except through the functions in the `Nx`
  module, and public ones, of which its data can be directly accessed and
  visited. The former typically have the `Backend` suffix.

  `Nx` ships with the following backends:

    * `Nx.BinaryBackend` - an opaque backend written in pure Elixir
      that stores the data in Elixir's binaries. This is the default
      backend used by the `Nx` module. The backend itself (and its
      data) is private and must not be accessed directly.

    * `Nx.Defn.Expr` - a public backend used by `defn` to build
      expression graphs that are traversed by custom compilers.
  """

  @type t :: %{__struct__: atom()}

  @type tensor :: Nx.Tensor.t()
  @type shape :: Nx.Tensor.shape()
  @type axis :: Nx.Tensor.axis()
  @type axes :: Nx.Tensor.axes()

  @callback iota(tensor, axis | nil) :: tensor
  @callback random_uniform(tensor, number, number) :: tensor
  @callback random_normal(tensor, mu :: float, sigma :: float) :: tensor

  @callback to_batched_list(out :: tensor, tensor) :: [tensor]
  @callback to_binary(tensor, keyword) :: binary
  @callback backend_deallocate(tensor) :: :ok | :already_deallocated
  @callback backend_transfer(tensor, module, keyword) :: tensor

  @callback tensor(tensor) :: tensor
  @callback inspect(tensor, Inspect.Opts.tensor()) :: tensor
  @callback from_binary(out :: tensor, binary, keyword) :: tensor
  @callback as_type(out :: tensor, tensor) :: tensor
  @callback reshape(out :: tensor, tensor, shape) :: tensor
  @callback squeeze(out :: tensor, tensor, axes) :: tensor
  @callback broadcast(out :: tensor, tensor, shape, axes) :: tensor
  @callback transpose(out :: tensor, tensor, axes) :: tensor
  @callback pad(out :: tensor, tensor, pad_value :: tensor, padding_config :: list()) :: tensor
  @callback reverse(out :: tensor, tensor, axes) :: tensor

  @callback dot(out :: tensor, tensor, axes, tensor, axes) :: tensor
  @callback clip(out :: tensor, tensor, min :: tensor, max :: tensor) :: tensor
  @callback slice(out :: tensor, tensor, list, list, list) :: tensor
  @callback concatenate(out :: tensor, tensor, axis) :: tensor
  @callback select(out :: tensor, tensor, tensor, tensor) :: tensor

  @callback conv(out :: tensor, tensor, kernel :: tensor, keyword) :: tensor
  @callback all?(out :: tensor, tensor, keyword) :: tensor
  @callback any?(out :: tensor, tensor, keyword) :: tensor
  @callback sum(out :: tensor, tensor, keyword) :: tensor
  @callback product(out :: tensor, tensor, keyword) :: tensor
  @callback reduce_max(out :: tensor, tensor, keyword) :: tensor
  @callback reduce_min(out :: tensor, tensor, keyword) :: tensor
  @callback argmax(out :: tensor, tensor, keyword) :: tensor
  @callback argmin(out :: tensor, tensor, keyword) :: tensor
  @callback reduce(out :: tensor, tensor, acc :: tensor, keyword, fun) :: tensor
  @callback reduce_window(out :: tensor, tensor, acc :: tensor, shape, keyword, fun) :: tensor
  @callback window_sum(out :: tensor, tensor, shape, keyword) :: tensor
  @callback window_product(out :: tensor, tensor, shape, keyword) :: tensor
  @callback window_max(out :: tensor, tensor, shape, keyword) :: tensor
  @callback window_min(out :: tensor, tensor, shape, keyword) :: tensor
  @callback map(out :: tensor, tensor, fun) :: tensor
  @callback sort(out :: tensor, tensor, keyword) :: tensor

  @callback cholesky(out :: tensor, tensor) :: tensor


  binary_ops =
    [:add, :subtract, :multiply, :power, :remainder, :divide, :arctan2, :min, :max, :quotient] ++
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

end
