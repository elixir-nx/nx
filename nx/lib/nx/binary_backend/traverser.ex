defmodule Nx.BinaryBackend.Traverser do
  alias Nx.BinaryBackend.Traverser
  alias Nx.BinaryBackend.ViewIter

  defstruct [
    :offsets,
    :offsets_template,
    :readers_template,
    :readers,
    :cycle_size,
    :size,
    cycle: 0
  ]

  def build(shape, axes, opts \\ []) do
    limits = Keyword.get(opts, :limits, :none)
    dilations = Keyword.get(opts, :dilations, 1)
    transpose = Keyword.get(opts, :transpose, :none)
    reverse? = Keyword.get(opts, :reverse, false)
    do_build(shape, axes, limits, dilations, transpose, reverse?)
  end

  defp do_build(shape, axes, limits, dilations, transpose, reverse?) do
    size = Nx.size(shape)
    {offsets_path, readers_path} = do_build_paths(shape, axes, limits, dilations, transpose)

    offsets = expand_path(offsets_path)
    readers = expand_path(readers_path)

    cycle_size = length(offsets) * length(readers)

    do_build_struct(size, cycle_size, offsets, readers, reverse?)
  end

  defp do_build_struct(size, cycle_size, offsets, readers, reverse? \\ false) do
    offsets = reverse_list(offsets, reverse?)
    readers = reverse_list(readers, reverse?)

    %Traverser{
      size: size,
      cycle_size: cycle_size,
      offsets_template: offsets,
      offsets: offsets,
      readers_template: readers,
      readers: readers
    }
  end

  defp do_build_paths(shape, [_ | _] = axes, limits, dilations, transpose) do
    axes = Enum.sort(axes)
    min = hd(axes)

    shape
    |> weighted_shape(limits, dilations)
    |> transpose_weighted_shape(transpose)
    |> Enum.drop(min)
    |> aggregate_paths(axes, min, [], [])
  end

  defp do_build_paths(shape, [], limits, dilations, transpose) do
    shape
    |> weighted_shape(limits, dilations)
    |> transpose_weighted_shape(transpose)
    |> aggregate_paths([], 0, [], [])
  end

  def next(%Traverser{offsets: []} = trav) do
    case next_cycle(trav) do
      :done ->
        :done

      trav ->
        next(trav)
    end
  end

  def next(%Traverser{offsets: [_ | offsets], readers: [], readers_template: rt} = trav) do
    next(%Traverser{trav | offsets: offsets, readers: rt})
  end

  def next(%Traverser{offsets: [o | _], readers: readers} = trav) do
    {:cont, r, rest} = next_item(readers)
    {:cont, o + r, %Traverser{trav | readers: rest}}
  end

  def next_view(
        %Traverser{
          offsets: [head_o | offsets],
          readers: readers,
          readers_template: readers_template
        } = trav
      ) do
    size = length(readers_template)
    view = do_build_struct(size, size, [head_o], readers)
    trav = %Traverser{trav | offsets: offsets, readers: readers_template}
    {:cont, view, trav}
  end

  def next_view(%Traverser{offsets: []} = trav) do
    case next_cycle(trav) do
      :done ->
        :done

      trav ->
        next_view(trav)
    end
  end

  def iter_views(%Traverser{} = trav) do
    ViewIter.build(trav)
  end

  def flatten(%Traverser{size: size} = trav) do
    readers =
      trav
      |> iter_views()
      |> Enum.flat_map(fn %Traverser{offsets: [o], readers: readers} ->
        Enum.map(readers, fn r -> r + o end)
      end)
    %Traverser{
      readers_template: readers,
      readers: readers,
      offsets_template: [0],
      offsets: [0],
      size: size,
      cycle_size: size,
      cycle: 0,
    }
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

  defp next_cycle(trav) do
    %Traverser{
      cycle: cycle_prev,
      size: size,
      cycle_size: cycle_size,
      offsets_template: offsets_template,
      readers_template: readers_template
    } = trav

    cycle = cycle_prev + 1

    if div(size, cycle_size) == cycle do
      :done
    else
      cycle_offset = cycle * cycle_size
      offsets = Enum.map(offsets_template, fn o -> o + cycle_offset end)
      %Traverser{trav | cycle: cycle, offsets: offsets, readers: readers_template}
    end
  end

  defp next_item([]) do
    :done
  end

  defp next_item(start..stop) when start == stop do
    {:cont, start, []}
  end

  defp next_item(start..stop) when start < stop do
    {:cont, start, (start + 1)..stop}
  end

  defp next_item([first | rest]) do
    {:cont, first, rest}
  end

  defp weighted_shape(shape, limits, dilations) do
    rank = tuple_size(shape)

    dilations =
      if is_list(dilations),
        do: Enum.reverse(dilations),
        else: List.duplicate(dilations, rank)

    weighted_shape(shape, rank, 1, limits, dilations, [])
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
    # expanded path comes out backward so we just flatten
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
    check_transpose!(ws, order)

    map =
      ws
      |> Enum.with_index()
      |> Map.new(fn {item, i} -> {i, item} end)

    Enum.map(order, fn i -> Map.fetch!(map, i) end)
  end

  defp check_transpose!(ws, order) do
    len = length(order)

    if length(ws) != len do
      raise ArgumentError,
            "expected length of permutation list" <>
              " #{inspect(order)} to match rank of weighted" <>
              " shape #{length(ws)}"
    end

    if Enum.sort(order) != Enum.to_list(0..(len - 1)) do
      raise ArgumentError,
            "expected each axis of the permutation list to appear exactly" <>
              " 1 time, got: #{inspect(order)}"
    end
  end
end
