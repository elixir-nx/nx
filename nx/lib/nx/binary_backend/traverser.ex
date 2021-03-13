defmodule Nx.BinaryBackend.Traverser do
  alias Nx.BinaryBackend.Traverser
  alias Nx.BinaryBackend.WeightedShape

  defstruct [:ctx]

  def range(count) when is_integer(count) and count >= 0 do
    %Traverser{ctx: {{1, count, [0]}, Enum.to_list(0..(count - 1))}}
  end

  # def build(_shape, _opts \\ []) do
  #   raise "build is removed"
  #   # agg_axes = Keyword.get(opts, :aggregate, [])
  #   # limits = Keyword.get(opts, :limits, :none)
  #   # dilations = Keyword.get(opts, :dilations, 1)
  #   # transpose = Keyword.get(opts, :transpose, :none)
  #   # reverse = Keyword.get(opts, :reverse, :none)
  #   # weight = Keyword.get(opts, :weight, 1)

  #   # size = Nx.size(shape)

  #   # {offsets_ws, readers_ws} =
  #   #   build_weighted_shape(shape, weight, agg_axes, limits, dilations, transpose, reverse)

  #   # build_from_parts(size, offsets_ws, readers_ws, weight)

  
  # end

  def build(size, weight, weighted_shape) when is_list(weighted_shape) do
    build(size, weight, WeightedShape.aggregate(weighted_shape, []))
  end

  @doc """
  Build a traverser from weighted shape.
  """
  def build(size, weight, {offsets_ws, readers_ws}) do    
    size = size * weight
    offsets = expand_path(offsets_ws)
    readers = expand_path(readers_ws)

    cycle_size = length(offsets) * length(readers) * weight

    do_build_struct(size, cycle_size, offsets, readers)
  end

  defp do_build_struct(size, cycle_size, offsets, readers) do
    n_cycles = div(size, cycle_size)
    offset_ctx = {n_cycles, cycle_size, offsets}

    %Traverser{ctx: {offset_ctx, readers}}
  end

  # defp build_weighted_shape(shape, weight, agg_axes, limits, dilations, transpose, reverse) do
  #   shape
  #   |> WeightedShape.build(weight, limits, dilations)
  #   |> case do
  #     ws when transpose == :none ->
  #       ws

  #     ws when is_list(transpose) ->
  #       WeightedShape.transpose(ws, transpose)
  #   end
  #   |> WeightedShape.aggregate(agg_axes)
  # end

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

  @doc """
  Element-wise reduce over 2 traversals.
  """
  def elem_wise_reduce2(trav1, trav2, acc, fun) do
    n = size(trav1)

    # TODO: remove this check - for debugging while WIP
    if n != size(trav2) do
      raise ArgumentError, "elem_wise_reduce2 sizes were different, got: #{n} and #{size(trav2)} "
    end

    %Traverser{ctx: {{_nc1, cs1, o1}, r1}} = trav1
    %Traverser{ctx: {{_nc2, cs2, o2}, r2}} = trav2

    ox1 = expand_offsets_and_readers(o1, r1)
    ox2 = expand_offsets_and_readers(o2, r2)

    elem_reduce(n, cs1, 0, ox1, ox1, cs2, 0, ox2, ox2, acc, fun)
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

  defp elem_reduce(0, _cs1, _co1, _ox1, _imm_ox1, _cs2, _co2, _ox2, _imm_ox2, acc, _fun) do
    acc
  end

  defp elem_reduce(n, cs1, co1, [], imm_ox1, cs2, co2, ox2, imm_ox2, acc, fun) do
    elem_reduce(n, cs1, co1 + cs1, imm_ox1, imm_ox1, cs2, co2, ox2, imm_ox2, acc, fun)
  end

  defp elem_reduce(n, cs1, co1, ox1, imm_ox1, cs2, co2, [], imm_ox2, acc, fun) do
    elem_reduce(n, cs1, co1, ox1, imm_ox1, cs2, co2 + cs2, imm_ox2, imm_ox2, acc, fun)
  end

  defp elem_reduce(n, cs1, co1, [o1 | ox1], imm_ox1, cs2, co2, [o2 | ox2], imm_ox2, acc, fun) do
    elem_reduce(
      n - 1,
      cs1,
      co1,
      ox1,
      imm_ox1,
      cs2,
      co2,
      ox2,
      imm_ox2,
      fun.(co1 + o1, co2 + o2, acc),
      fun
    )
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

  defp expand_path(path) do
    path
    |> Enum.reverse()
    |> WeightedShape.traversal_list()
    |> List.flatten()
  end

  defp expand_offsets_and_readers([], []) do
    []
  end

  defp expand_offsets_and_readers([o | o_rest], r) do
    expand_offsets_and_readers(o_rest, o, r, r)
  end

  defp expand_offsets_and_readers(o_rest, o, [r | r_rest], imm_r) do
    [o + r | expand_offsets_and_readers(o_rest, o, r_rest, imm_r)]
  end

  defp expand_offsets_and_readers([], _, [], _) do
    []
  end

  defp expand_offsets_and_readers([o | o_rest], _, [], imm_r) do
    expand_offsets_and_readers(o_rest, o, imm_r, imm_r)
  end
end
