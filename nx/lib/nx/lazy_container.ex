defprotocol Nx.LazyContainer do
  @moduledoc """
  Converts a data structure to a lazy container.

  Sometimes building tensors for a container is an expensive
  operation, so we want to allow that to happen lazily.

  This module provides a single traverse implementation
  that emits the tensor template and their contents as two
  distinct values.

  If a data structures does not implement this protocol,
  a default implementation that converts eager to lazy is
  done by default.
  """

  @fallback_to_any true

  @doc """
  Traverses recursively tensors in a data structure with `acc` and `fun`.

  For each tensor in the container, `fun` receives a tensor
  template, an anonymous function to build the actual tensor,
  and the accumulator . It returns a two element tuple with
  the updated container and the accumulator.

  This function returns the updated container and the accumulator.

  Note this function is recursive by default. Therefore if you
  are implementing this function and one of your arguments may
  be containers, you must call `Nx.Defn.Composite.lazy_traverse/3`
  on said arguments so they are recursively traversed.
  """
  @spec traverse(t(), acc, (Nx.template(), (-> Nx.Tensor.t()), acc -> {term(), acc})) :: acc when acc: term()
  def traverse(data, acc, fun)
end

defimpl Nx.LazyContainer, for: Tuple do
  def traverse(tuple, acc, fun) do
    tuple
    |> Tuple.to_list()
    |> Enum.map_reduce(acc, &Nx.Defn.Composite.lazy_traverse(&1, &2, fun))
    |> then(fn {list, acc} -> {List.to_tuple(list), acc} end)
  end
end

defimpl Nx.LazyContainer, for: Map do
  def traverse(map, acc, fun) do
    map
    |> Map.to_list()
    |> Enum.sort()
    |> Enum.map_reduce(acc, fn {k, v}, acc ->
      {v, acc} = Nx.Defn.Composite.lazy_traverse(v, acc, fun)
      {{k, v}, acc}
    end)
    |> then(fn {list, acc} -> {Map.new(list), acc} end)
  end
end

defimpl Nx.LazyContainer, for: Any do
  def traverse(data, acc, fun) do
    Nx.Container.traverse(data, acc, &Nx.Defn.Composite.lazy_traverse(&1, &2, fun))
  end
end

