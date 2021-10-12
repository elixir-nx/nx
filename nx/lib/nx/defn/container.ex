defprotocol Nx.Defn.Container do
  def decompose(container)
end

defimpl Nx.Defn.Container, for: Tuple do
  def decompose(tuple) do
    decompose_impl(Tuple.to_list(tuple), {[], fn _ -> {} end})
  end

  defp decompose_impl([], {leaves, fun}), do: {List.flatten(Enum.reverse(leaves)), fun}

  defp decompose_impl([obj | rest], {leaves, fun}) do
    case Nx.Defn.Container.impl_for(obj) do
      nil ->
        # The object is a leaf, so no nesting
        {leaves, fun} = {[obj | leaves], fn x -> Tuple.append(fun.(x), obj) end}

        decompose_impl(rest, {leaves, fun})

      _ ->
        # Otherwise, we need to get leaves of nested obj
        {child_leaves, child_fun} = Nx.Defn.Container.decompose(obj)

        fun = fn x ->
          child = child_fun.(child_leaves)
          Tuple.append(fun.(x), child)
        end

        decompose_impl(rest, {[child_leaves | leaves], fun})
    end
  end
end

defimpl Nx.Defn.Container, for: Map do
  def decompose(map) do
    {leaves, fun} =
      map
      |> Enum.reduce({[], fn _ -> %{} end}, fn {k, obj}, {leaves, fun} ->
        case Nx.Defn.Container.impl_for(obj) do
          nil ->
            # The object is a leaf, no nesting
            {[obj | leaves], fn x -> Map.put(fun.(x), k, obj) end}

          _ ->
            # Otherwise, we need to get leaves of nested obj
            {child_leaves, child_fun} = Nx.Defn.Container.decompose(obj)

            fun = fn x ->
              child = child_fun.(child_leaves)
              Map.put(fun.(x), k, child)
            end

            {[child_leaves | leaves], fun}
        end
      end)

    {List.flatten(Enum.reverse(leaves)), fun}
  end
end
