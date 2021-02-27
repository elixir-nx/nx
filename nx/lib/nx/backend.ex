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

    * `Nx.TemplateBackend` - an opaque backend written that works as
      a template in APIs to declare the type, shape, and names of
      tensors to be expected in the future.

    * `Nx.Defn.Expr` - a public backend used by `defn` to build
      expression graphs that are traversed by custom compilers.

  This module also includes functions that are meant to be shared
  across backends.
  """

  @type t :: %{__struct__: atom()}

  @type tensor :: Nx.Tensor.t()
  @type shape :: Nx.Tensor.shape()
  @type axis :: Nx.Tensor.axis()
  @type axes :: Nx.Tensor.axes()

  @callback eye(tensor) :: tensor
  @callback iota(tensor, axis | nil) :: tensor
  @callback random_uniform(tensor, number, number) :: tensor
  @callback random_normal(tensor, mu :: float, sigma :: float) :: tensor

  @callback to_batched_list(out :: tensor, tensor) :: [tensor]
  @callback to_binary(tensor, limit :: non_neg_integer) :: binary
  @callback backend_deallocate(tensor) :: :ok | :already_deallocated
  @callback backend_transfer(tensor, module, keyword) :: tensor

  @callback inspect(tensor, Inspect.Opts.t()) :: tensor
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
  @callback qr({q :: tensor, r :: tensor}, tensor, keyword) :: tensor

  binary_ops =
    [:add, :subtract, :multiply, :power, :remainder, :divide, :atan2, :min, :max, :quotient] ++
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

  defp chunk([], data, {kind, size}, limit, _docs) do
    # TODO: Simplify inspection once nonfinite are officially supported in the VM
    {doc, tail} =
      case kind do
        :s ->
          <<head::size(size)-signed-native, tail::binary>> = data
          {Integer.to_string(head), tail}

        :u ->
          <<head::size(size)-unsigned-native, tail::binary>> = data
          {Integer.to_string(head), tail}

        :f ->
          <<head::size(size)-bitstring, tail::binary>> = data
          {inspect_float(head, size), tail}

        :bf ->
          <<head::16-bitstring, tail::binary>> = data
          {inspect_bf16(head), tail}
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

  defp inspect_bf16(<<0xFF80::16-native>>), do: "-Inf"
  defp inspect_bf16(<<0x7F80::16-native>>), do: "Inf"
  defp inspect_bf16(<<0xFFC1::16-native>>), do: "NaN"
  defp inspect_bf16(<<0xFF81::16-native>>), do: "NaN"

  if System.endianness() == :little do
    defp inspect_bf16(bf16) do
      <<x::float-little-32>> = <<0::16, bf16::binary>>
      Float.to_string(x)
    end
  else
    defp inspect_bf16(bf16) do
      <<x::float-big-32>> = <<bf16::binary, 0::16>>
      Float.to_string(x)
    end
  end

  defp inspect_float(data, 32) do
    case data do
      <<0xFF800000::32-native>> -> "-Inf"
      <<0x7F800000::32-native>> -> "Inf"
      <<0xFF800001::32-native>> -> "NaN"
      <<0xFFC00001::32-native>> -> "NaN"
      <<x::float-32-native>> -> Float.to_string(x)
    end
  end

  defp inspect_float(data, 64) do
    case data do
      <<0xFFF0000000000000::64-native>> -> "-Inf"
      <<0x7FF0000000000000::64-native>> -> "Inf"
      <<0x7FF0000000000001::64-native>> -> "NaN"
      <<0x7FF8000000000001::64-native>> -> "NaN"
      <<x::float-64-native>> -> Float.to_string(x)
    end
  end
end
