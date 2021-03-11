defmodule Nx.BinaryBackend.Traverser do
  alias Nx.BinaryBackend.Traverser

  defstruct [:ctx]

  def range(count) when is_integer(count) and count >= 0 do
    %Traverser{ctx: {{1, count, [0]}, Enum.to_list(0..(count - 1))}}
  end

  def build(shape, opts \\ []) do
    agg_axes = Keyword.get(opts, :aggregate, [])
    limits = Keyword.get(opts, :limits, :none)
    dilations = Keyword.get(opts, :dilations, 1)
    transpose = Keyword.get(opts, :transpose, :none)
    reverse? = Keyword.get(opts, :reverse, false)
    weight = Keyword.get(opts, :weight, 1)
    do_build(shape, weight, agg_axes, limits, dilations, transpose, reverse?)
  end

  def size(%Traverser{ctx: {{n_cycles, cycle_size, _}, _}}) do
    n_cycles * cycle_size
  end

  def zip_reduce_aggregates(
        %Traverser{ctx: {o1, r1}},
        %Traverser{ctx: {o2, r2}},
        outer_acc,
        inner_acc,
        inner_fn,
        outer_fn
      ) do
    reduce(0, o1, r1, outer_acc, fn {offset1, readers1}, outer_acc2 ->
      reduce(0, o2, r2, outer_acc2, fn {offset2, readers2}, outer_acc3 ->
        result = zip_reduce_readers(offset1, readers1, offset2, readers2, inner_acc, inner_fn)
        outer_fn.(result, outer_acc3)
      end)
    end)
  end

  def reduce(%Traverser{ctx: {o, r}}, acc, reducer) do
    reduce(0, o, r, acc, fn {offset, readers}, acc2 ->
      reduce_readers(offset, readers, acc2, reducer)
    end)
  end

  def to_flat_list(trav) do
    trav
    |> reduce([], fn i, acc -> [i | acc] end)
    |> Enum.reverse()
  end

  def reduce_aggregates(%Traverser{ctx: {o, r}}, acc, fun) do
    reduce(0, o, r, acc, fn {offset, reads}, acc ->
      fun.(Enum.map(reads, fn r -> offset + r end), acc)
    end)
  end

  def map(%Traverser{ctx: {o, r}}, fun) do
    reduce(0, o, r, [], fn {offset, readers}, acc2 ->
      reduce_readers(offset, readers, acc2, fn o, acc3 ->
        [fun.(o) | acc3]
      end)
    end)
  end

  defp do_build(shape, weight, agg_axes, limits, dilations, transpose, reverse?) do
    size = Nx.size(shape) * weight

    {offsets_path, readers_path} =
      do_build_paths(shape, weight, agg_axes, limits, dilations, transpose)

    offsets = expand_path(offsets_path)
    readers = expand_path(readers_path)

    cycle_size = length(offsets) * length(readers) * weight

    do_build_struct(size, cycle_size, offsets, readers, reverse?)
  end

  defp do_build_struct(size, cycle_size, offsets, readers, reverse?) do
    readers = reverse_list(readers, reverse?)

    offsets = reverse_list(offsets, reverse?)
    n_cycles = div(size, cycle_size)
    offset_ctx = {n_cycles, cycle_size, offsets}

    %Traverser{ctx: {offset_ctx, readers}}
  end

  defp do_build_paths(shape, weight, [_ | _] = agg_axes, limits, dilations, transpose) do
    agg_axes = Enum.sort(agg_axes)
    min = hd(agg_axes)

    shape
    |> weighted_shape(weight, limits, dilations)
    |> transpose_weighted_shape(transpose)
    |> Enum.drop(min)
    |> aggregate_paths(agg_axes, min, [], [])
  end

  defp do_build_paths(shape, weight, [], limits, dilations, transpose) do
    shape
    |> weighted_shape(weight, limits, dilations)
    |> transpose_weighted_shape(transpose)
    |> aggregate_paths([], 0, [], [])
  end

  defp reduce(cycle, {n_cycles, cycle_size, offsets}, readers, acc, fun) do
    reduce(cycle, n_cycles, cycle_size, offsets, readers, acc, fun)
  end

  defp reduce(cycle, n_cycles, _cycle_size, _ot, _readers, acc, _fun) when cycle >= n_cycles do
    acc
  end

  defp reduce(cycle, n_cycles, cycle_size, offsets, readers, acc, fun) do
    acc = reduce_cycle(cycle * cycle_size, offsets, readers, acc, fun)
    reduce(cycle + 1, n_cycles, cycle_size, offsets, readers, acc, fun)
  end

  @compile {:inline, reduce_cycle: 5, reduce_readers: 4, zip_reduce_readers: 6}

  defp reduce_cycle(_, [], _, acc, _) do
    acc
  end

  defp reduce_cycle(cycle_offset, [o | offsets], readers, acc, fun) do
    reduce_cycle(cycle_offset, offsets, readers, fun.({cycle_offset + o, readers}, acc), fun)
  end

  defp reduce_readers(_offset, [], acc, _fun) do
    acc
  end

  defp reduce_readers(offset, [reader | rest], acc, fun) do
    reduce_readers(offset, rest, fun.(offset + reader, acc), fun)
  end

  defp zip_reduce_readers(_o1, [], _o2, [], inner_acc, _inner_fn) do
    inner_acc
  end

  defp zip_reduce_readers(o1, [r1 | rest1], o2, [r2 | rest2], inner_acc, inner_fn) do
    zip_reduce_readers(o1, rest1, o2, rest2, inner_fn.(o1 + r1, o2 + r2, inner_acc), inner_fn)
  end

  defp weighted_shape(shape, weight, limits, dilations) do
    rank = tuple_size(shape)

    dilations =
      if is_list(dilations),
        do: Enum.reverse(dilations),
        else: List.duplicate(dilations, rank)

    weighted_shape(shape, rank, weight, limits, dilations, [])
  end

  defp weighted_shape(_shape, 0, _weight, _limits, [], acc), do: acc

  defp weighted_shape(shape, pos, weight, limits, [dilation | dilations], acc) do
    shape_elem = :erlang.element(pos, shape)

    element =
      if limits == :none, do: shape_elem, else: min(:erlang.element(pos, limits), shape_elem)

    dilation_factor =
      if element == 1,
        do: 1,
        else: dilation

    acc = [{element, dilation_factor * weight} | acc]
    weighted_shape(shape, pos - 1, weight * shape_elem, limits, dilations, acc)
  end

  defp aggregate_paths([pair | shape], [i | axes], i, pre, pos),
    do: aggregate_paths(shape, axes, i + 1, pre, [pair | pos])

  defp aggregate_paths([pair | shape], axes, i, pre, pos),
    do: aggregate_paths(shape, axes, i + 1, [pair | pre], pos)

  defp aggregate_paths([], [], _i, pre, pos), do: {pre, pos}

  defp reverse_list(list, true), do: Enum.reverse(list)
  defp reverse_list(list, false), do: list

  defp expand_path(path) do
    path
    |> Enum.reverse()
    |> weighted_traverse(0)
    |> List.wrap()
    |> List.flatten()
  end

  defp weighted_traverse([], i) do
    i
  end

  defp weighted_traverse([{dim, size} | dims], i) do
    weighted_traverse(dim, size, dims, i)
  end

  defp weighted_traverse(dim, dim_size, dims, i) do
    head = weighted_traverse(dims, i)

    case dim do
      1 ->
        [head]

      _ ->
        i = i + dim_size
        [head | weighted_traverse(dim - 1, dim_size, dims, i)]
    end
  end

  defp transpose_weighted_shape(ws, :none) do
    ws
  end

  defp transpose_weighted_shape(ws, order) do
    map =
      ws
      |> Enum.with_index()
      |> Map.new(fn {item, i} -> {i, item} end)

    Enum.map(order, fn i -> Map.fetch!(map, i) end)
  end
end
