defmodule Nx.BinaryBackend.WeightedShape do
  @moduledoc """
  Weighted shapes are lists of dim-and-weight tuples used
  for traversing binaries.
  """

  @doc """
  Build a list of {dim, weight} pairs.

  By default the weight multiplier is 1, there are no limits,
  and the dilation factor is 1.

  ## Examples

      iex> WeightedShape.build({5})
      [{5, 1}]

      iex> WeightedShape.build({2, 2, 2, 2})
      [{2, 8}, {2, 4}, {2, 2}, {2, 1}]

      iex> WeightedShape.build({2, 2, 2, 2}, 8)
      [{2, 64}, {2, 32}, {2, 16}, {2, 8}]

      iex> WeightedShape.build({10, 10, 10, 10}, 1, {2, 2, 2, 2})
      [{2, 1000}, {2, 100}, {2, 10}, {2, 1}]

      iex> WeightedShape.build({3, 4, 5}, 1, :none, 3)
      [{3, 60}, {4, 15}, {5, 3}]
  """
  def build(shape, weight \\ 1, limits \\ :none, dilations \\ 1) do
    rank = tuple_size(shape)

    dilations =
      if is_list(dilations),
        do: Enum.reverse(dilations),
        else: List.duplicate(dilations, rank)

    do_build(shape, rank, weight, limits, dilations, [])
  end

  defp do_build(_shape, 0, _weight, _limits, [], acc), do: acc

  defp do_build(shape, pos, weight, limits, [dilation | dilations], acc) do
    shape_elem = :erlang.element(pos, shape)

    element =
      if limits == :none, do: shape_elem, else: min(:erlang.element(pos, limits), shape_elem)

    acc = [{element, dilation_factor(element, dilation) * weight} | acc]
    do_build(shape, pos - 1, weight * shape_elem, limits, dilations, acc)
  end

  defp dilation_factor(1, _dilation), do: 1
  defp dilation_factor(_, dilation), do: dilation

  @doc """
  Split the weighted shape into the paths for aggregating.

  ## Examples

      iex> WeightedShape.aggregate(WeightedShape.build({2, 3}), [1])
      {[], [{3, 1}]}

      iex> WeightedShape.aggregate(WeightedShape.build({2, 3}), [0])
      {[{3, 1}], [{2, 3}]}

      iex> WeightedShape.aggregate(WeightedShape.build({2, 3}), [])
      {[{2, 3}, {3, 1}], []}

      iex> WeightedShape.aggregate(WeightedShape.build({2, 3, 3}), [2])
      {[], [{3, 1}]}

      iex> WeightedShape.aggregate(WeightedShape.build({2, 3, 4, 5, 6}), [1, 2])
      {[{5, 6}, {6, 1}], [{3, 120}, {4, 30}]}
  """
  def aggregate(weighted_shape, []) do
    {weighted_shape, []}
  end

  def aggregate(weighted_shape, axes) do
    axes = Enum.sort(axes)
    min = List.first(axes) || 0

    # drops = Enum.take(weighted_shape, min)

    {ws_rev, rev_aggs} =
      weighted_shape
      |> Enum.drop(min)
      |> aggregate(axes, min, [], [])

    {Enum.reverse(ws_rev), Enum.reverse(rev_aggs)}
  end

  defp aggregate([pair | rest], [i | axes], i, pre, pos) do
    aggregate(rest, axes, i + 1, pre, [pair | pos])
  end

  defp aggregate([pair | rest], axes, i, pre, pos) do
    aggregate(rest, axes, i + 1, [pair | pre], pos)
  end

  defp aggregate([], [], _i, pre, pos) do
    {pre, pos}
  end

  @doc """
  Rearrange the dim-weight pairs of the weights.

  ## Examples

      iex> WeightedShape.transpose(WeightedShape.build({2, 3}), [0, 1])
      [{2, 3}, {3, 1}]

      iex> WeightedShape.transpose(WeightedShape.build({2, 3}), [1, 0])
      [{3, 1}, {2, 3}]
  """
  def transpose(weighted_shape, axes) do
    map =
      weighted_shape
      |> Enum.with_index()
      |> Map.new(fn {item, i} -> {i, item} end)

    Enum.map(axes, fn i -> Map.fetch!(map, i) end)
  end

  @doc """
  Multiplies the current weights by the given weight.

      iex> WeightedShape.with_weight(WeightedShape.build({2, 3}), 64)
      [{2, 192}, {3, 64}]

      iex> WeightedShape.with_weight(WeightedShape.build({3, 10}), 8)
      [{3, 80}, {10, 8}]
  """
  def with_weight(weighted_shape, weight) do
    Enum.map(weighted_shape, fn
      {d, w} -> {d, w * weight}
      {d, w, tags} -> {d, w * weight, tags}
    end)
  end

  @doc """
  Limits each dim of the weighted shape to a maximum dimensional size given
  in limits.

      iex> WeightedShape.limit(WeightedShape.build({3, 3}), {2, 2})
      [{2, 3}, {2, 1}]

      iex> WeightedShape.limit(WeightedShape.build({3, 3}), {2, 3})
      [{2, 3}, {3, 1}]

      iex> WeightedShape.limit(WeightedShape.build({3, 3, 3}), {1, 1, 1})
      [{1, 9}, {1, 3}, {1, 1}]

      iex> WeightedShape.limit(WeightedShape.build({1, 3}), {3, 1})
      [{1, 3}, {1, 1}]
  """
  def limit(weighted_shape, limits) when is_tuple(limits) do
    do_limit(weighted_shape, Tuple.to_list(limits))
  end

  defp do_limit([], []) do
    []
  end

  defp do_limit([{dim, weight} | weighted_shape], [lim | limits]) do
    [{min(dim, lim), weight} | do_limit(weighted_shape, limits)]
  end

  @doc """
  Dilates a weighted shape with an integer or list of integers.

  ## Examples

      iex> WeightedShape.dilate(WeightedShape.build({2, 3}), 10)
      [{2, 30}, {3, 10}]

      iex> WeightedShape.dilate(WeightedShape.build({1, 1, 3}), 10)
      [{1, 3}, {1, 3}, {3, 10}]

      iex> WeightedShape.dilate(WeightedShape.build({2, 3}), [7, 5])
      [{2, 21}, {3, 5}]

  """
  def dilate(weighted_shape, dilation) when is_integer(dilation) do
    dilate_int(weighted_shape, dilation)
  end

  def dilate(weighted_shape, dilations) when is_list(dilations) do
    dilate_list(weighted_shape, dilations)
  end

  defp dilate_int([], _), do: []

  defp dilate_int([{dim, w} | rest], dil) do
    [{dim, w * dilation_factor(dim, dil)} | dilate_int(rest, dil)]
  end

  defp dilate_list([], []), do: []

  defp dilate_list([{dim, w} | rest], [dil | dilations]) do
    [{dim, w * dilation_factor(dim, dil)} | dilate_list(rest, dilations)]
  end

  @doc """
  Tags the dims at the given axes as reversed.

  ## Examples

      iex> WeightedShape.reverse(WeightedShape.build({2, 3}), [1])
      [{2, 3}, {3, 1, [:reverse]}]
  """
  def reverse(weighted_shape, axes) do
    map_with_axis(weighted_shape, fn {dim, axis} ->
      if axis in axes do
        tag_dim(dim, :reverse)
      else
        dim
      end
    end)
  end

  @doc """
  Turns a weighted shape into a list used for traversal.

  ## Examples

      iex> WeightedShape.traversal_list(WeightedShape.build({3, 2}))
      [[0, 1], [2, 3], [4, 5]]

      iex> ws = WeightedShape.build({3, 2})
      iex> {_, pos} = WeightedShape.aggregate(ws, [1])
      iex> WeightedShape.traversal_list(pos)
      [0, 1]

      iex> ws = WeightedShape.build({3, 2})
      iex> {pre, _} = WeightedShape.aggregate(ws, [1])
      iex> WeightedShape.traversal_list(pre)
      [0]
  """
  def traversal_list(weighted_shape) do
    weighted_shape
    |> expand()
    |> traversal_list(0)
    |> List.wrap()
  end

  defp traversal_list([], i) do
    i
  end

  defp traversal_list([:reverse | dims], i) do
    dims
    |> traversal_list(i)
    |> Enum.reverse()
  end

  defp traversal_list([{dim, size} | dims], i) do
    traversal_list(dim, size, dims, i)
  end

  defp traversal_list(dim, dim_size, dims, i) do
    head = traversal_list(dims, i)

    case dim do
      1 ->
        [head]

      _ ->
        i = i + dim_size
        [head | traversal_list(dim - 1, dim_size, dims, i)]
    end
  end

  # Helpers

  defp tag_dim({d, w}, tag) do
    {d, w, [tag]}
  end

  defp tag_dim({d, w, tags}, tag) do
    {d, w, [tag | tags]}
  end

  defp map_with_axis(weighted_shape, fun) do
    map_with_axis(weighted_shape, 0, fun)
  end

  defp map_with_axis([], _axis, _fun) do
    []
  end

  defp map_with_axis([head | tail], axis, fun) do
    [fun.({head, axis}) | map_with_axis(tail, axis + 1, fun)]
  end

  # unravels tagged dims into a flat list of dim-weight pairs and ops.
  defp expand([]) do
    []
  end

  defp expand([{d, _} = head | tail]) when is_integer(d) do
    [head | expand(tail)]
  end

  defp expand([{d, w, tags} | tail]) do
    expand_dim(d, w, tags, tail)
  end

  defp expand_dim(d, w, [], tail) when is_integer(d) do
    [{d, w} | expand(tail)]
  end

  defp expand_dim(d, w, [:reverse | tags], tail) do
    [:reverse | expand_dim(d, w, tags, tail)]
  end
end
