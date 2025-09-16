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

    @min_mono256 232
    @max_mono256 255
    @mono256 @min_mono256..@max_mono256

    def inspect(%{tensor: tensor, opts: heatmap_opts}, opts) do
      %{shape: shape, names: names, type: type, vectorized_axes: vectorized_axes} = tensor

      open = color("[", :list, opts)
      sep = color(",", :list, opts)
      close = color("]", :list, opts)

      data = data(Nx.devectorize(tensor), heatmap_opts, opts, {open, sep, close})
      type = color(Nx.Type.to_string(type), :atom, opts)

      {vectorized_names, vectorized_sizes} = Enum.unzip(vectorized_axes)
      vectorized_shape_tuple = List.to_tuple(vectorized_sizes)

      vectorized_shape =
        if vectorized_axes == [] do
          empty()
        else
          concat([
            "vectorized",
            Nx.Shape.to_algebra(vectorized_shape_tuple, vectorized_names, open, close),
            line()
          ])
        end

      shape = Nx.Shape.to_algebra(shape, names, open, close)

      color("#Nx.Heatmap<", :map, opts)
      |> concat(nest(concat([line(), vectorized_shape, type, shape, line(), data]), 2))
      |> concat(color("\n>", :map, opts))
    end

    defp data(tensor, heatmap_opts, opts, doc) do
      whitespace = Keyword.get(heatmap_opts, :ansi_whitespace, "\u3000")

      {entry_fun, line_fun, nf_fun} =
        if Keyword.get_lazy(heatmap_opts, :ansi_enabled, &IO.ANSI.enabled?/0) do
          scale = Enum.count(@mono256) - 1

          entry_fun = fn range ->
            index = range |> Kernel.*(scale) |> round()
            color = Enum.fetch!(@mono256, index)
            [IO.ANSI.color_background(color), whitespace]
          end

          {entry_fun, &IO.iodata_to_binary([&1 | IO.ANSI.reset()]),
           &non_finite_ansi(&1, whitespace)}
        else
          {&Integer.to_string(&1 |> Kernel.*(9) |> round()), &IO.iodata_to_binary/1,
           &non_finite_ascii/1}
        end

      render(tensor, opts, doc, entry_fun, line_fun, nf_fun)
    end

    defp non_finite_ascii(:infinity), do: ?+
    defp non_finite_ascii(:neg_infinity), do: ?-
    defp non_finite_ascii(:nan), do: ?x

    defp non_finite_ansi(:infinity, _ws), do: [IO.ANSI.color_background(@max_mono256), "∞"]
    defp non_finite_ansi(:neg_infinity, _ws), do: [IO.ANSI.color_background(@min_mono256), "∞"]
    defp non_finite_ansi(:nan, ws), do: [IO.ANSI.red_background(), ws]

    defp render(%{shape: {size}, type: type} = tensor, _opts, _doc, entry_fun, line_fun, nf_fun) do
      data = Nx.to_flat_list(tensor)
      min = type |> Nx.Constants.min_finite() |> Nx.to_number()
      max = type |> Nx.Constants.max_finite() |> Nx.to_number()
      {data, [], min, max} = take_min_max(data, size, max, min, [])
      base = if max == min, do: 1, else: max - min

      data
      |> Enum.map(fn
        elem when is_number(elem) -> entry_fun.((elem - min) / base)
        elem when is_atom(elem) -> nf_fun.(elem)
      end)
      |> line_fun.()
    end

    defp render(%{shape: shape, type: type} = tensor, opts, doc, entry_fun, line_fun, nf_fun) do
      {dims, [rows, cols]} = shape |> Tuple.to_list() |> Enum.split(-2)

      limit = opts.limit
      list_opts = if limit == :infinity, do: [], else: [limit: rows * cols * limit + 1]
      data = Nx.to_flat_list(tensor, list_opts)

      min = type |> Nx.Constants.min_finite() |> Nx.to_number()
      max = type |> Nx.Constants.max_finite() |> Nx.to_number()
      state = {rows, cols, entry_fun, line_fun, nf_fun, min, max}
      {data, _rest, _limit} = chunk(dims, data, limit, state, doc)
      data
    end

    defp take_min_max(rest, 0, min, max, acc),
      do: {Enum.reverse(acc), rest, min, max}

    defp take_min_max([head | tail], count, min, max, acc) when is_atom(head),
      do: take_min_max(tail, count - 1, min, max, [head | acc])

    defp take_min_max([head | tail], count, min, max, acc),
      do: take_min_max(tail, count - 1, min(min, head), max(max, head), [head | acc])

    defp chunk([], acc, limit, {rows, cols, entry_fun, line_fun, nf_fun, min, max}, _docs) do
      {acc, rest, min, max} = take_min_max(acc, rows * cols, max, min, [])
      base = if max == min, do: 1, else: max - min

      {[], doc} =
        Enum.reduce(1..rows, {acc, empty()}, fn _, {acc, doc} ->
          {line, acc} =
            Enum.map_reduce(1..cols, acc, fn
              _, [elem | acc] when is_number(elem) -> {entry_fun.((elem - min) / base), acc}
              _, [elem | acc] when is_atom(elem) -> {nf_fun.(elem), acc}
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
