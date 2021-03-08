defmodule Nx.BinaryBackend.Traverser do
  alias Nx.BinaryBackend.Traverser
  alias Nx.BinaryBackend.Index
  alias Nx.BinaryBackend.Weights

  defstruct size: nil,
            offset: 0,
            seq_template: [],
            seq: [],
            offsets: []

  def build(shape, weights \\ nil, [_|_] = axes) do
    weights = weights || Weights.build(shape)
    size = Weights.size(weights)
    axes = Enum.sort(axes)
    min_axis = hd(axes)
    
    first_axis..last_axis = 
      shape
      |> Nx.rank()
      |> Index.range()
      |> Index.range_shift(min_axis..0)

    groups = group_axes(first_axis..last_axis, axes)

    {pivot_offset, minor, major_nested} =
      Enum.reduce(groups, {0, [], []}, fn range, {pivot_offset, minor_acc, major_acc} ->
        {minor_axis, majors_range} = Index.range_pop_left(range)
        {minor_d, minor_o} = axis_to_pair(shape, weights, minor_axis)
        major_pairs = Enum.map(majors_range, fn major_axis ->
          axis_to_pair(shape, weights, major_axis)
        end)
        
        {pivot_offset + minor_o, [{minor_d, minor_o} | minor_acc], [major_acc, major_pairs]}
      end)

    major = List.flatten(major_nested)
    pivot = List.last(major)
    [offset | offsets] = offsets_to_list(major)

    {pivot_range, _} = pivot
    minor_pivot = {pivot_range, pivot_offset}

    template = minor ++ [minor_pivot]

    %Traverser{
      size: size,
      seq_template: template,
      seq: template,
      offsets: offsets,
      offset: offset,
    }
  end


  def next(%Traverser{seq: [], offsets: []}) do
    :done
  end

  def next(%Traverser{seq: [_ | _] = state, offset: offset} = trav) do
    {i, seq2} = pop_seq(state)
    {:cont, offset + i, put_seq(trav, seq2)}
  end

  def next(%Traverser{seq: [], offsets: [offset | offsets], seq_template: seq} = trav) do
    next(%Traverser{trav | seq: seq, offset: offset, offsets: offsets})
  end
  
  defp put_seq(trav, seq) do
    %Traverser{trav | seq: seq}
  end

  defp pop_seq([{range, o} | rest]) do
    case Index.range_pop_left(range) do
      {i, []} ->
        {i + o, rest}
      {i, range2} ->
        {i + o, [{range2, o} | rest]}
    end
  end

  defp group_axes(range, axes, acc \\ [])

  defp group_axes(range, [], grouped) do
    Enum.reverse([range | grouped])
  end

  defp group_axes([], _, grouped) do
    Enum.reverse(grouped)
  end

  defp group_axes(range, [axis | rest], acc) do
    case Index.range_split_right(range, axis) do
      {[], range} ->
        group_axes(range, rest, acc)

      {left, new_range} ->
        group_axes(new_range, rest, [left | acc])
    end 
  end

  defp axis_to_pair(shape, weights, axis) do
    o = Weights.offset_of_axis(weights, axis)
    d = elem(shape, axis)
    {Index.range(d), o}
  end

  defp offsets_to_list(range_offsets) do
    range_offsets
    |> offsets_to_list(0)
    |> List.flatten()
  end

  defp offsets_to_list([], total) do
    [total]
  end

  defp offsets_to_list([{range, o} | rest], total) do
    Enum.map(range, fn i ->
      offsets_to_list(rest, total + i * o)
    end)
  end

  defimpl Enumerable do
    def count(%Traverser{size: size}) do
      {:ok, size}
    end

    def member?(_, _) do
      {:error, __MODULE__}
    end

    def slice(_) do
      {:error, __MODULE__}
    end

    def reduce(_, {:halt, acc}, _fun) do
      {:halted, acc}
    end

    def reduce(trav, {:suspend, acc}, fun) do
      {:suspended, acc, fn acc2 -> reduce(trav, acc2, fun) end}
    end

    def reduce(trav, {:cont, acc}, fun) do
      case Traverser.next(trav) do
        {:cont, i, trav2} ->
          reduce(trav2, fun.(i, acc), fun)

        :done ->
          {:done, acc}
      end
    end
  end
end