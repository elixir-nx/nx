defmodule Nx.Tensor do
  @moduledoc """
  The tensor struct and the behaviour for backends.

  `Nx.Tensor` is a generic container for multidimensional data structures.
  It contains the tensor type, shape, and names. The data itself is a
  struct that points to a backend responsible for controlling the data.
  The backend must implement the behaviour defined by this module.

  The behaviour is mostly callback implementations of the functions in
  the `Nx` module with the tensor output shape given as first argument.

  `Nx` ships with the following backends for `Nx.Tensor`:

    * `Nx.BinaryBackend` - a pure Elixir backend built on top
      of Elixir's binaries. This is the default backend used
      by the `Nx` module

  Note that `Nx.Defn.Expr` is also a tensor backend. It is used
  by `defn` to build expression graphs that are traversed by
  custom compilers.
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

  @callback iota(t, axis | nil) :: t
  @callback random_uniform(t, number, number) :: t
  @callback random_normal(t, mu :: float, sigma :: float) :: t

  @callback to_batch(out :: t, t) :: [t]
  @callback to_binary(t) :: binary
  @callback device_read(t) :: t
  @callback device_deallocate(t) :: t
  @callback device_transfer(t, module, keyword) :: t

  @callback tensor(t) :: t
  @callback inspect(t, Inspect.Opts.t()) :: t
  @callback from_binary(out :: t, binary) :: t
  @callback as_type(out :: t, t) :: t
  @callback reshape(out :: t, t, shape) :: t
  @callback squeeze(out :: t, t, axes) :: t
  @callback broadcast(out :: t, t, shape, axes) :: t
  @callback transpose(out :: t, t, axes) :: t
  @callback pad(out :: t, t, pad_value :: t, padding_config :: list()) :: t
  @callback reverse(out :: t, t, axes) :: t

  @callback dot(out :: t, t, axes, t, axes) :: t
  @callback clip(out :: t, t, min :: t, max :: t) :: t
  @callback slice(out :: t, t, list, list, list) :: t
  @callback concatenate(out :: t, t, axis) :: t
  @callback select(out :: t, t, t, t) :: t

  @callback conv(out :: t, t, kernel :: t, keyword) :: t
  @callback all?(out :: t, t, keyword) :: t
  @callback any?(out :: t, t, keyword) :: t
  @callback sum(out :: t, t, keyword) :: t
  @callback product(out :: t, t, keyword) :: t
  @callback reduce_max(out :: t, t, keyword) :: t
  @callback reduce_min(out :: t, t, keyword) :: t
  @callback argmax(out :: t, t, keyword) :: t
  @callback argmin(out :: t, t, keyword) :: t
  @callback reduce(out :: t, t, acc :: t, keyword, fun) :: t
  @callback reduce_window(out :: t, t, acc :: t, shape, keyword, fun) :: t
  @callback window_sum(out :: t, t, shape, keyword) :: t
  @callback window_product(out :: t, t, shape, keyword) :: t
  @callback window_max(out :: t, t, shape, keyword) :: t
  @callback window_min(out :: t, t, shape, keyword) :: t
  @callback map(out :: t, t, fun) :: t
  @callback sort(out :: t, t, keyword) :: t

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

  ## Access

  @behaviour Access

  @impl true
  def fetch(%Nx.Tensor{shape: {}} = tensor, _index) do
    raise ArgumentError,
          "cannot use the tensor[index] syntax on scalar tensor #{inspect(tensor)}"
  end

  def fetch(tensor, index) when is_integer(index),
    do: {:ok, fetch_axes(tensor, [{0, index}])}

  def fetch(tensor, _.._ = range),
    do: {:ok, fetch_axes(tensor, [{0, range}])}

  def fetch(tensor, []),
    do: {:ok, tensor}

  def fetch(%{names: names} = tensor, [{_, _} | _] = keyword),
    do: {:ok, fetch_axes(tensor, with_names(keyword, names, []))}

  def fetch(tensor, [_ | _] = list),
    do: {:ok, fetch_axes(tensor, with_index(list, 0, []))}

  def fetch(_tensor, value) do
    raise """
    tensor[slice] expects slice to be one of:

      * an integer representing a zero-based index
      * a first..last range representing inclusive start-stop indexes
      * a list of integers and ranges
      * a keyword list of integers and ranges

    Got #{inspect(value)}
    """
  end

  defp with_index([h | t], i, acc), do: with_index(t, i + 1, [{i, h} | acc])
  defp with_index([], _i, acc), do: acc

  defp with_names([{k, v} | t], names, acc),
    do: with_names(t, names, [{Nx.Shape.find_name!(names, k), v} | acc])

  defp with_names([], _names, acc),
    do: acc

  defp fetch_axes(%Nx.Tensor{shape: shape} = tensor, axes) do
    rank = Nx.rank(shape)
    impl = Nx.Shared.impl!(tensor)
    {start, lengths, squeeze} = fetch_axes(rank - 1, axes, shape, [], [], [])

    %{tensor | shape: List.to_tuple(lengths)}
    |> impl.slice(tensor, start, lengths, List.duplicate(1, rank))
    |> Nx.squeeze(axes: squeeze)
  end

  defp fetch_axes(axis, axes, shape, start, lengths, squeeze) when axis >= 0 do
    case List.keytake(axes, axis, 0) do
      {{^axis, index}, axes} when is_integer(index) ->
        index = normalize_index(index, axis, shape)
        fetch_axes(axis - 1, axes, shape, [index | start], [1 | lengths], [axis | squeeze])

      {{^axis, first..last}, axes} ->
        first = normalize_index(first, axis, shape)
        last = normalize_index(last, axis, shape)

        if last < first do
          raise ArgumentError,
                "slicing a tensor requires an increasing range, got: #{inspect(first..last)}"
        end

        len = last - first + 1
        fetch_axes(axis - 1, axes, shape, [first | start], [len | lengths], squeeze)

      {{^axis, value}, _} ->
        raise ArgumentError,
              "slicing a tensor on an axis requires an integer or a range, got: #{inspect(value)}"

      nil ->
        fetch_axes(axis - 1, axes, shape, [0 | start], [elem(shape, axis) | lengths], squeeze)
    end
  end

  defp fetch_axes(_axis, [{axis, _} | _], shape, _start, _lengths, _squeeze) do
    raise ArgumentError,
          "unknown or duplicate axis #{axis} found when slicing shape #{inspect(shape)}"
  end

  defp fetch_axes(_axis, [], _shape, start, lengths, squeeze) do
    {start, lengths, squeeze}
  end

  defp normalize_index(index, axis, shape) do
    dim = elem(shape, axis)
    norm = if index < 0, do: dim + index, else: index

    if norm < 0 or norm >= dim do
      raise ArgumentError,
            "index #{index} is out of bounds for axis #{axis} in shape #{inspect(shape)}"
    end

    norm
  end

  @impl true
  def get_and_update(_tensor, _index, _update) do
    raise "Access.get_and_update/3 is not yet supported by Nx.Tensor"
  end

  @impl true
  def pop(_tensor, _index) do
    raise "Access.pop/2 is not yet supported by Nx.Tensor"
  end

  defimpl Inspect do
    import Inspect.Algebra

    def inspect(%{shape: shape, names: names, type: type} = tensor, opts) do
      open = color("[", :list, opts)
      close = color("]", :list, opts)
      type = color(Nx.Type.to_string(type), :atom, opts)
      shape = Nx.Shape.to_algebra(shape, names, open, close)
      data = tensor.data.__struct__.inspect(tensor, opts)
      inner = concat([line(), type, shape, line(), data])

      color("#Nx.Tensor<", :map, opts)
      |> concat(nest(inner, 2))
      |> concat(color("\n>", :map, opts))
    end
  end
end
