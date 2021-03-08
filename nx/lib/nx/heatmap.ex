defmodule Nx.Heatmap do
  @moduledoc """
  Provides a heatmap that is printed using ANSI colors
  in the terminal.
  """

  @doc false
  defstruct [:tensor, opts: []]

  @behaviour Access

  @impl true
  def fetch(%Nx.Heatmap{tensor: tensor} = hm, value) do
    case Access.fetch(tensor, value) do
      {:ok, %Nx.Tensor{shape: {}} = tensor} -> {:ok, tensor}
      {:ok, tensor} -> {:ok, put_in(hm.tensor, tensor)}
      :error -> :error
    end
  end

  @impl true
  def get_and_update(hm, key, fun) do
    {get, tensor} = Access.get_and_update(hm.tensor, key, fun)
    {get, put_in(hm.tensor, tensor)}
  end

  @impl true
  def pop(hm, key) do
    {pop, tensor} = Access.pop(hm.tensor, key)
    {pop, put_in(hm.tensor, tensor)}
  end

  defimpl Inspect do
    import Inspect.Algebra

    @mono265 Enum.to_list(232..255)

    def inspect(%{tensor: tensor, opts: heatmap_opts}, opts) do
      %{shape: shape, names: names, type: type} = tensor

      open = color("[", :list, opts)
      sep = color(",", :list, opts)
      close = color("]", :list, opts)

      data = data(tensor, heatmap_opts, opts, {open, sep, close})
      type = color(Nx.Type.to_string(type), :atom, opts)
      shape = Nx.Shape.to_algebra(shape, names, open, close)

      color("#Nx.Heatmap<", :map, opts)
      |> concat(nest(concat([line(), type, shape, line(), data]), 2))
      |> concat(color("\n>", :map, opts))
    end

    defp data(tensor, heatmap_opts, opts, doc) do
      whitespace = Keyword.get(heatmap_opts, :ansi_whitespace, "\u3000")

      {entry_fun, line_fun} =
        if Keyword.get_lazy(heatmap_opts, :ansi_enabled, &IO.ANSI.enabled?/0) do
          scale = length(@mono265) - 1

          entry_fun = fn range ->
            index = range |> Kernel.*(scale) |> round()
            color = Enum.fetch!(@mono265, index)
            [IO.ANSI.color_background(color), whitespace]
          end

          {entry_fun, &IO.iodata_to_binary([&1 | IO.ANSI.reset()])}
        else
          {&Integer.to_string(&1 |> Kernel.*(9) |> round()), &IO.iodata_to_binary/1}
        end

      render(tensor, opts, doc, entry_fun, line_fun)
    end

    defp render(%{shape: {size}} = tensor, _opts, _doc, entry_fun, line_fun) do
      data = Nx.to_flat_list(tensor)
      {data, [], min, max} = take_min_max(data, size)
      base = max - min

      data
      |> Enum.map(fn elem -> entry_fun.((elem - min) / base) end)
      |> line_fun.()
    end

    defp render(%{shape: shape} = tensor, opts, doc, entry_fun, line_fun) do
      {dims, [rows, cols]} = shape |> Tuple.to_list() |> Enum.split(-2)

      limit = opts.limit
      list_opts = if limit == :infinity, do: [], else: [limit: rows * cols * limit + 1]
      data = Nx.to_flat_list(tensor, list_opts)

      {data, _rest, _limit} = chunk(dims, data, limit, {rows, cols, entry_fun, line_fun}, doc)
      data
    end

    defp take_min_max([head | tail], count),
      do: take_min_max(tail, count - 1, head, head, [head])

    defp take_min_max(rest, 0, min, max, acc),
      do: {Enum.reverse(acc), rest, min, max}

    defp take_min_max([head | tail], count, min, max, acc),
      do: take_min_max(tail, count - 1, min(min, head), max(max, head), [head | acc])

    defp chunk([], acc, limit, {rows, cols, entry_fun, line_fun}, _docs) do
      {acc, rest, min, max} = take_min_max(acc, rows * cols)
      base = max - min

      {[], doc} =
        Enum.reduce(1..rows, {acc, empty()}, fn _, {acc, doc} ->
          {line, acc} =
            Enum.map_reduce(1..cols, acc, fn _, [elem | acc] ->
              {entry_fun.((elem - min) / base), acc}
            end)

          doc = concat(doc, concat(line(), line_fun.(line)))
          {acc, doc}
        end)

      if limit == :infinity, do: {doc, rest, limit}, else: {doc, rest, limit - 1}
    end

    defp chunk([dim | dims], data, limit, rcw, {open, sep, close} = docs) do
      {acc, rest, limit} =
        chunk_each(dim, data, [], limit, fn chunk, limit ->
          chunk(dims, chunk, limit, rcw, docs)
        end)

      doc =
        if(dims == [], do: open, else: concat(open, line()))
        |> concat(concat(Enum.intersperse(acc, concat(sep, line()))))
        |> nest(2)
        |> concat(line())
        |> concat(close)

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
  end
end
