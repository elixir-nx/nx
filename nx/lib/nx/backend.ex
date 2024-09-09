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

  @callback backend_deallocate(tensor) :: :ok | :already_deallocated
  @callback backend_copy(tensor, module, backend_options) :: tensor
  @callback backend_transfer(tensor, module, backend_options) :: tensor
  @callback to_batched(out :: tensor, tensor, keyword) :: [tensor]
  @callback to_binary(tensor, limit :: non_neg_integer) :: binary
  @callback inspect(tensor, Inspect.Opts.t()) :: tensor
  @callback from_pointer(
              opaque_pointer :: term(),
              type :: tuple(),
              shape :: tuple(),
              backend_opts :: keyword(),
              opts :: keyword()
            ) :: {:ok, tensor} | {:error, term()}
  @callback to_pointer(tensor, opts :: keyword) :: {:ok, term()} | {:error, term()}

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
  @callback gather(out :: tensor, input :: tensor, indices :: tensor, keyword) :: tensor
  @callback concatenate(out :: tensor, tensor, axis) :: tensor
  @callback stack(out :: tensor, tensor, axis) :: tensor
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
  @callback sort(out :: tensor, tensor, keyword) :: tensor
  @callback argsort(out :: tensor, tensor, keyword) :: tensor
  @callback window_scatter_max(out :: tensor, tensor, tensor, tensor, shape, keyword) :: tensor
  @callback window_scatter_min(out :: tensor, tensor, tensor, tensor, shape, keyword) :: tensor
  @callback indexed_add(out :: tensor, tensor, indices :: tensor, updates :: tensor, keyword) ::
              tensor
  @callback indexed_put(out :: tensor, tensor, indices :: tensor, updates :: tensor, keyword) ::
              tensor

  @callback lu({p :: tensor, l :: tensor, u :: tensor}, tensor, keyword) :: tensor
  @callback triangular_solve(out :: tensor, a :: tensor, b :: tensor, keyword) :: tensor
  @callback svd({u :: tensor, s :: tensor, v :: tensor}, tensor, keyword) :: tensor

  @callback fft(out :: tensor, tensor, keyword) :: tensor
  @callback ifft(out :: tensor, tensor, keyword) :: tensor
  @callback fft2(out :: tensor, tensor, keyword) :: tensor
  @callback ifft2(out :: tensor, tensor, keyword) :: tensor

  binary_ops =
    [:add, :subtract, :multiply, :pow, :remainder, :divide, :atan2, :min, :max, :quotient] ++
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

  @callback qr({q :: tensor, r :: tensor}, tensor, keyword) :: tensor
  @callback cholesky(out :: tensor, tensor) :: tensor
  @callback eigh({eigenvals :: tensor, eigenvecs :: tensor}, tensor, keyword) :: tensor
  @callback solve(out :: tensor, a :: tensor, b :: tensor) :: tensor
  @callback determinant(out :: tensor, t :: tensor) :: tensor
  @callback logical_not(out :: tensor, t :: tensor) :: tensor
  @callback phase(out :: tensor, t :: tensor) :: tensor

  @callback cumulative_sum(out :: tensor, t :: tensor, keyword) :: tensor
  @callback cumulative_product(out :: tensor, t :: tensor, keyword) :: tensor
  @callback cumulative_min(out :: tensor, t :: tensor, keyword) :: tensor
  @callback cumulative_max(out :: tensor, t :: tensor, keyword) :: tensor

  @callback all_close(out :: tensor, tensor, tensor, keyword) :: tensor
  @callback top_k(out :: tensor, tensor, keyword) :: tensor
  @callback take(out :: tensor, input :: tensor, indices :: tensor, keyword) :: tensor
  @callback take_along_axis(out :: tensor, input :: tensor, indices :: tensor, keyword) :: tensor

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
    svd: 3,
    top_k: 3,
    fft2: 3,
    ifft2: 3,
    qr: 3,
    cholesky: 2,
    eigh: 3,
    take: 4,
    take_along_axis: 4
  ]

  ## Inspect implementation

  require Nx.Shared
  alias Inspect.Algebra, as: IA

  @doc """
  Inspects the given tensor given by `binary`.

  Note the `binary` may have fewer elements than the
  tensor size but, in such cases, it must strictly have
  more elements than `inspect_opts.limit`

  ## Options

  The following must be passed through `Inspect` `:custom_options`

    * `:nx_precision` - Configures the floating-point number printing precision.
      If set, will print floating-point numbers in scientific notation using the
      specified number of significant digits. Otherwise, default Elixir printing
      rules are applied.
  """
  def inspect(%{shape: shape, type: type}, binary, inspect_opts) do
    open = IA.color("[", :list, inspect_opts)
    sep = IA.color(",", :list, inspect_opts)
    close = IA.color("]", :list, inspect_opts)

    # TODO: This is a palliative accessibility-related solution
    precision = inspect_opts.custom_options[:nx_precision]

    dims = Tuple.to_list(shape)

    {data, _rest, _limit} =
      chunk(dims, binary, type, inspect_opts.limit, precision, {open, sep, close})

    data
  end

  defp chunk([], data, type, limit, precision, _docs) do
    {doc, tail} =
      Nx.Shared.match_types [type] do
        <<match!(head, 0), tail::binary>> = data
        {inspect_value(read!(head, 0), precision), tail}
      end

    if limit == :infinity, do: {doc, tail, limit}, else: {doc, tail, limit - 1}
  end

  defp chunk([dim | dims], data, type, limit, precision, {open, sep, close} = docs) do
    {acc, rest, limit} =
      chunk_each(dim, data, [], limit, fn chunk, limit ->
        chunk(dims, chunk, type, limit, precision, docs)
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

  defp inspect_value(integer, _) when is_integer(integer), do: Integer.to_string(integer)
  defp inspect_value(:neg_infinity, _), do: "-Inf"
  defp inspect_value(:infinity, _), do: "Inf"
  defp inspect_value(:nan, _), do: "NaN"
  defp inspect_value(%Complex{} = val, precision), do: complex_to_string(val, precision)

  defp inspect_value(float, precision), do: float_to_string(float, precision)

  defp float_to_string(float, precision) do
    [integer_part, decimal_part, exponent_part] =
      case String.split(Float.to_string(float), [".", "e"], parts: 3) do
        [i, d] -> [i, d, ""]
        [i, d, e] -> [i, d, "e" <> e]
      end

    # We'll now prune decimal_part to ensure we have at most `precision`
    # digits there.

    decimal_part =
      decimal_part
      |> binary_part(0, min(byte_size(decimal_part), precision))

    #  We also prune trailing zeros. Only for more than 1 digit because that single
    # digit always needs to stay put.
    decimal_part =
      if byte_size(decimal_part) > 1 do
        String.trim_trailing(decimal_part, "0")
      else
        decimal_part
      end

    integer_part <> "." <> decimal_part <> exponent_part
  end

  def complex_to_string(%Complex{re: re, im: im}, precision) do
    re_str = inspect_value(re, precision)
    im_str = inspect_value(im, precision)

    im_str =
      case im_str do
        "-" <> _ -> im_str
        s -> "+" <> s
      end

    re_str <> im_str <> "i"
  end
end
