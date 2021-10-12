defprotocol Nx.Defn.Container do
  @fallback_to_any true
  def decompose(container)
end

defimpl Nx.Defn.Container, for: Tuple do
  def decompose(tuple) do
    decompose_impl(Tuple.to_list(tuple), {[], fn _ -> {} end})
  end

  defp decompose_impl([], {leaves, fun}), do: {List.flatten(Enum.reverse(leaves)), fun}

  defp decompose_impl([obj | rest], {leaves, fun}) do
    {child_leaves, child_fun} = Nx.Defn.Container.decompose(obj)

    fun = fn x ->
      child = child_fun.(child_leaves)
      Tuple.append(fun.(x), child)
    end

    decompose_impl(rest, {[child_leaves | leaves], fun})
  end
end

defimpl Nx.Defn.Container, for: Map do
  def decompose(map) do
    {leaves, fun} =
      map
      |> Enum.reduce({[], fn _ -> %{} end}, fn {k, obj}, {leaves, fun} ->
        {child_leaves, child_fun} = Nx.Defn.Container.decompose(obj)

        fun = fn x ->
          child = child_fun.(child_leaves)
          Map.put(fun.(x), k, child)
        end

        {[child_leaves | leaves], fun}
      end)

    {List.flatten(Enum.reverse(leaves)), fun}
  end
end

# Fallback for structs and leaf values, this is really so
# we can have a generic struct implementation which matches
# how we handle structs now. It doesn't feel like the best way
# though.
defimpl Nx.Defn.Container, for: Any do
  def decompose(map) when is_struct(map) do
    {leaves, fun} =
      map
      |> Map.from_struct()
      |> Nx.Defn.Container.decompose()

    {leaves, fn x -> struct(map.__struct__, fun.(x)) end}
  end

  def decompose(any) do
    {any, & &1}
  end
end
