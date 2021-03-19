defmodule Nx.BinaryBackend.Traverser do
  alias Nx.BinaryBackend.Traverser
  alias Nx.BinaryBackend.WeightedShape

  defstruct [:count, :ctx]

  def range(count) when is_integer(count) and count >= 0 do
    readers = Enum.to_list(0..(count - 1))
    %Traverser{
      count: count,
      ctx: {[0], [0], readers}
    }
  end

  def range(count, {_, sizeof}) when is_integer(count) and count >= 0 do
    readers = Enum.map(0..(count - 1), fn o -> o * sizeof end)
    %Traverser{
      count: count,
      ctx: {[0], [0], readers}
    }
  end

  @doc """
  Build a traverser from weighted shape or an aggregate-weighted-shape-tuple.
  """
  def build(weighted_shape) when is_list(weighted_shape) do
    weighted_shape
    |> WeightedShape.aggregate([])
    |> build()
  end

  def build({cycles_ws, offsets_ws, readers_ws}) do
    c_count = WeightedShape.size(cycles_ws)
    o_count = WeightedShape.size(offsets_ws)
    r_count = WeightedShape.size(readers_ws)
  
    cycles = expand_weighted_shape(cycles_ws)
    offsets = expand_weighted_shape(offsets_ws)
    readers = expand_weighted_shape(readers_ws)

    %Traverser{
      count: c_count * o_count * r_count,
      ctx: {cycles, offsets, readers}
    }
  end

  def build(weighted_shape, {_, sizeof}) when is_list(weighted_shape) do
    weighted_shape
    |> WeightedShape.with_weight(sizeof)
    |> build()
  end

  def build({c, o, r}, {_, sizeof}) do
    c = WeightedShape.with_weight(c, sizeof)
    o = WeightedShape.with_weight(o, sizeof)
    r = WeightedShape.with_weight(r, sizeof)
    build({c, o, r})
  end

  def count(%Traverser{count: c}), do: c

  def zip_reduce_aggregates(
        %Traverser{ctx: {c1, o1, r1}},
        %Traverser{ctx: {c2, o2, r2}},
        outer_acc,
        inner_acc,
        inner_fn,
        outer_fn
      ) do
    reduce_co(c1, o1, outer_acc, fn offset1, outer_acc2 ->
      reduce_co(c2, o2, outer_acc2, fn offset2, outer_acc3 ->
        result = zip_reduce_readers(offset1, r1, offset2, r2, inner_acc, inner_fn)
        outer_fn.(result, outer_acc3)
      end)
    end)
  end

  def reduce(%Traverser{ctx: {c, o, r}}, acc, reducer) do
    reduce_co(c, o, acc, fn offset, acc2 ->
      reduce_readers(offset, r, acc2, reducer)
    end)
  end

  defp reduce_co([], _offsets, acc, _fun) do
    acc
  end

  defp reduce_co([c | c_rest], offsets, acc, fun) do
    acc = reduce_o(c, offsets, acc, fun)
    reduce_co(c_rest, offsets, acc, fun)
  end

  defp reduce_o(_, [], acc, _) do
    acc
  end

  defp reduce_o(c, [o | offsets], acc, fun) do
    reduce_o(c, offsets, fun.(c + o, acc), fun)
  end

  def to_flat_list(trav) do
    trav
    |> reduce([], fn i, acc -> [i | acc] end)
    |> Enum.reverse()
  end

  def reduce_aggregates(%Traverser{ctx: {c, o, reads}}, acc, fun) do
    reduce_co(c, o, acc, fn offset, acc ->
      fun.({offset, reads}, acc)
    end)
  end

  @compile {:inline, reduce_readers: 4, zip_reduce_readers: 6}

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

  defp expand_weighted_shape(path) do
    path
    |> WeightedShape.traversal_list()
    |> List.flatten()
  end
end
