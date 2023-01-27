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
  traversed. The former typically have the `Backend` suffix.

  `Nx` ships with the following backends:

    * `Nx.BinaryBackend` - an opaque backend written in pure Elixir
      that stores the data in Elixir's binaries. This is the default
      backend used by the `Nx` module. The backend itself (and its
      data) is private and must not be accessed directly.

    * `Nx.TemplateBackend` - an opaque backend written that works as
      a template in APIs to declare the type, shape, and names of
      tensors to be expected in the future.

    * `Nx.Defn.Expr` - a public backend used by `defn` to build
      expression trees that are traversed by custom compilers.

  This module also includes functions that are meant to be shared
  across backends.
  """

  @type t :: %{__struct__: atom()}

  @type tensor :: Nx.Tensor.t()
  @type shape :: Nx.Tensor.shape()
  @type axis :: Nx.Tensor.axis()
  @type axes :: Nx.Tensor.axes()
  @type backend_options :: term()

  @callback init(keyword()) :: backend_options

  @callback constant(out :: tensor, number | Complex.t(), backend_options) :: tensor
  @callback from_binary(out :: tensor, binary, backend_options) :: tensor
  @callback eye(tensor, backend_options) :: tensor
  @callback iota(tensor, axis | nil, backend_options) :: tensor
  @callback random_uniform(tensor, tensor, tensor, backend_options) :: tensor
  @callback random_normal(tensor, mu :: tensor, sigma :: tensor, backend_options) :: tensor

  @callback backend_deallocate(tensor) :: :ok | :already_deallocated
  @callback backend_copy(tensor, module, backend_options) :: tensor
  @callback backend_transfer(tensor, module, backend_options) :: tensor
  @callback to_batched(out :: tensor, tensor, keyword) :: [tensor]
  @callback to_binary(tensor, limit :: non_neg_integer) :: binary
  @callback inspect(tensor, Inspect.Opts.t()) :: tensor

  @callback as_type(out :: tensor, tensor) :: tensor
  @callback bitcast(out :: tensor, tensor) :: tensor
  @callback reshape(out :: tensor, tensor) :: tensor
  @callback squeeze(out :: tensor, tensor, axes) :: tensor
  @callback broadcast(out :: tensor, tensor, shape, axes) :: tensor
  @callback transpose(out :: tensor, tensor, axes) :: tensor
  @callback pad(out :: tensor, tensor, pad_value :: tensor, padding_config :: list()) :: tensor
  @callback reverse(out :: tensor, tensor, axes) :: tensor

  @callback dot(out :: tensor, tensor, axes, axes, tensor, axes, axes) :: tensor
  @callback clip(out :: tensor, tensor, min :: tensor, max :: tensor) :: tensor
  @callback slice(out :: tensor, tensor, list, list, list) :: tensor
  @callback put_slice(out :: tensor, tensor, tensor, list) :: tensor
  @callback take(out :: tensor, input :: tensor, indices :: tensor, axis) :: tensor
  @callback take_along_axis(out :: tensor, input :: tensor, indices :: tensor, axis) :: tensor
  @callback gather(out :: tensor, input :: tensor, indices :: tensor) :: tensor
  @callback concatenate(out :: tensor, tensor, axis) :: tensor
  @callback select(out :: tensor, tensor, tensor, tensor) :: tensor

  @callback conv(out :: tensor, tensor, kernel :: tensor, keyword) :: tensor
  @callback all(out :: tensor, tensor, keyword) :: tensor
  @callback any(out :: tensor, tensor, keyword) :: tensor
  @callback sum(out :: tensor, tensor, keyword) :: tensor
  @callback product(out :: tensor, tensor, keyword) :: tensor
  @callback reduce_max(out :: tensor, tensor, keyword) :: tensor
  @callback reduce_min(out :: tensor, tensor, keyword) :: tensor
  @callback argmax(out :: tensor, tensor, keyword) :: tensor
  @callback argmin(out :: tensor, tensor, keyword) :: tensor
  @callback reduce(out :: tensor, tensor, acc :: tensor, keyword, fun) :: tensor
  @callback window_reduce(out :: tensor, tensor, acc :: tensor, shape, keyword, fun) :: tensor
  @callback window_sum(out :: tensor, tensor, shape, keyword) :: tensor
  @callback window_product(out :: tensor, tensor, shape, keyword) :: tensor
  @callback window_max(out :: tensor, tensor, shape, keyword) :: tensor
  @callback window_min(out :: tensor, tensor, shape, keyword) :: tensor
  @callback map(out :: tensor, tensor, keyword, fun) :: tensor
  @callback sort(out :: tensor, tensor, keyword) :: tensor
  @callback argsort(out :: tensor, tensor, keyword) :: tensor
  @callback window_scatter_max(out :: tensor, tensor, tensor, tensor, shape, keyword) :: tensor
  @callback window_scatter_min(out :: tensor, tensor, tensor, tensor, shape, keyword) :: tensor
  @callback indexed_add(out :: tensor, target :: tensor, indices :: tensor, updates :: tensor) ::
              tensor
  @callback indexed_put(out :: tensor, target :: tensor, indices :: tensor, updates :: tensor) ::
              tensor

  @callback cholesky(out :: tensor, tensor) :: tensor
  @callback lu({p :: tensor, l :: tensor, u :: tensor}, tensor, keyword) :: tensor
  @callback qr({q :: tensor, r :: tensor}, tensor, keyword) :: tensor
  @callback triangular_solve(out :: tensor, a :: tensor, b :: tensor, keyword) :: tensor
  @callback eigh({eigenvals :: tensor, eigenvecs :: tensor}, tensor, keyword) :: tensor
  @callback svd({u :: tensor, s :: tensor, v :: tensor}, tensor, keyword) :: tensor

  @callback fft(out :: tensor, tensor, keyword) :: tensor
  @callback ifft(out :: tensor, tensor, keyword) :: tensor

  binary_ops =
    [:add, :subtract, :multiply, :power, :remainder, :divide, :atan2, :min, :max, :quotient] ++
      [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift] ++
      [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal] ++
      [:logical_and, :logical_or, :logical_xor]

  for binary_op <- binary_ops do
    @callback unquote(binary_op)(out :: tensor, tensor, tensor) :: tensor
  end

  unary_ops =
    Enum.map(Nx.Shared.unary_math_funs(), &elem(&1, 0)) ++
      [:abs, :bitwise_not, :ceil, :conjugate, :floor, :negate, :round, :sign] ++
      [:count_leading_zeros, :population_count, :real, :imag, :is_nan, :is_infinity]

  for unary_op <- unary_ops do
    @callback unquote(unary_op)(out :: tensor, tensor) :: tensor
  end

  ## Optional Callbacks

  @doc """
  Invoked for execution of optional callbacks with a default implementation.

  First we will attempt to call the optional callback itself
  (one of the many callbacks defined below), then we attempt
  to call this callback (which is also optional), then we
  fallback to the default iomplementation.
  """
  @callback optional(atom, [term], fun) :: tensor

  @callback solve(out :: tensor, a :: tensor, b :: tensor) :: tensor
  @callback determinant(out :: tensor, t :: tensor) :: tensor
  @callback logical_not(out :: tensor, t :: tensor) :: tensor
  @callback phase(out :: tensor, t :: tensor) :: tensor

  @callback cumulative_sum(out :: tensor, t :: tensor, keyword) :: tensor
  @callback cumulative_product(out :: tensor, t :: tensor, keyword) :: tensor
  @callback cumulative_min(out :: tensor, t :: tensor, keyword) :: tensor
  @callback cumulative_max(out :: tensor, t :: tensor, keyword) :: tensor

  @callback all_close(out :: tensor, tensor, tensor, keyword) :: tensor

  @optional_callbacks [
    optional: 3,
    solve: 3,
    determinant: 2,
    logical_not: 2,
    phase: 2,
    cumulative_sum: 3,
    cumulative_product: 3,
    cumulative_min: 3,
    cumulative_max: 3,
    all_close: 4,
    svd: 3
  ]

  ## Inspect implementation

  require Nx.Shared
  alias Inspect.Algebra, as: IA

  @doc """
  Inspects the given tensor given by `binary`.

  Note the `binary` may have fewer elements than the
  tensor size but, in such cases, it must strictly have
  more elements than `inspect_opts.limit`
  """
  def inspect(%{shape: shape, type: type}, binary, inspect_opts) do
    open = IA.color("[", :list, inspect_opts)
    sep = IA.color(",", :list, inspect_opts)
    close = IA.color("]", :list, inspect_opts)

    dims = Tuple.to_list(shape)
    {data, _rest, _limit} = chunk(dims, binary, type, inspect_opts.limit, {open, sep, close})
    data
  end

  defp chunk([], data, type, limit, _docs) do
    {doc, tail} =
      Nx.Shared.match_types [type] do
        <<match!(head, 0), tail::binary>> = data
        {inspect_value(read!(head, 0)), tail}
      end

    if limit == :infinity, do: {doc, tail, limit}, else: {doc, tail, limit - 1}
  end

  defp chunk([dim | dims], data, type, limit, {open, sep, close} = docs) do
    {acc, rest, limit} =
      chunk_each(dim, data, [], limit, fn chunk, limit ->
        chunk(dims, chunk, type, limit, docs)
      end)

    {open, sep, close, nest} =
      if dims == [] do
        {open, IA.concat(sep, " "), close, 0}
      else
        {IA.concat(open, IA.line()), IA.concat(sep, IA.line()), IA.concat(IA.line(), close), 2}
      end

    doc =
      open
      |> IA.concat(IA.concat(Enum.intersperse(acc, sep)))
      |> IA.nest(nest)
      |> IA.concat(close)

    {doc, rest, limit}
  end

  defp chunk_each(0, data, acc, limit, _fun) do
    {Enum.reverse(acc), data, limit}
  end

  defp chunk_each(_dim, data, acc, 0, _fun) do
    {Enum.reverse(["..." | acc]), data, 0}
  end

  defp chunk_each(dim, data, acc, limit, fun) do
    {doc, rest, limit} = fun.(data, limit)
    chunk_each(dim - 1, rest, [doc | acc], limit, fun)
  end

  defp inspect_value(%Complex{} = val), do: Complex.to_string(val)
  defp inspect_value(integer) when is_integer(integer), do: Integer.to_string(integer)
  defp inspect_value(float) when is_float(float), do: Float.to_string(float)
  defp inspect_value(:neg_infinity), do: "-Inf"
  defp inspect_value(:infinity), do: "Inf"
  defp inspect_value(:nan), do: "NaN"
end
