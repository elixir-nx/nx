defprotocol Nx.LazyContainer do
  @moduledoc """
  Converts a data structure to a container lazily and recursively.

  Sometimes building tensors for a container is an expensive
  operation, so we want to allow that to happen lazily.

  This module provides a single traverse implementation
  that emits the tensor template and a function that computes
  the tensor as two distinct values. Then a tensor is only
  allocated if necessary.

  This protocol is used throughout `Nx.Defn` API. This means
  compilation, jitting, and streaming will only realize lazy
  tensors when necessary.

  If a data structure does not implement this protocol,
  a default implementation converts eager to lazy using
  `Nx.Container`. When a value is given to `defn`, it is
  first converted to tensors and containers via `Nx.LazyContainer`.
  Inside `defn`, there are no lazy containers, only containers.
  """

  @fallback_to_any true

  @doc """
  Traverses recursively tensors in a data structure with `acc` and `fun`.

  For each tensor in the container, `fun` receives a tensor
  template, an anonymous function to build the actual tensor,
  and the accumulator . It returns a two element tuple with
  a non-lazy Nx.Container and the accumulator.

  This function returns the updated container and the accumulator.

  Note this function is recursive by default. Therefore if you
  are implementing this function and one of your arguments may
  be containers, you must call `Nx.LazyContainer.traverse/3`
  on said arguments so they are recursively traversed.
  """
  @spec traverse(t(), acc, (Nx.template(), (() -> Nx.Tensor.t()), acc -> {term(), acc})) ::
          {Nx.Container.t(), acc}
        when acc: term()
  def traverse(data, acc, fun)
end

defimpl Nx.LazyContainer, for: Nx.Tensor do
  def traverse(tensor, acc, fun) do
    fun.(%{tensor | data: %Nx.TemplateBackend{}}, fn -> tensor end, acc)
  end
end

defimpl Nx.LazyContainer, for: [Integer, Float, Complex] do
  def traverse(number, acc, fun) do
    tensor = Nx.to_tensor(number)
    fun.(%{tensor | data: %Nx.TemplateBackend{}}, fn -> tensor end, acc)
  end
end

# Implement to speed up fallback to container.
defimpl Nx.LazyContainer, for: Tuple do
  def traverse(tuple, acc, fun) do
    tuple
    |> Tuple.to_list()
    |> Enum.map_reduce(acc, &Nx.LazyContainer.traverse(&1, &2, fun))
    |> then(fn {list, acc} -> {List.to_tuple(list), acc} end)
  end
end

# Implement to speed up fallback to container.
defimpl Nx.LazyContainer, for: Map do
  def traverse(map, acc, fun) do
    map
    |> Map.to_list()
    |> Enum.sort()
    |> Enum.map_reduce(acc, fn {k, v}, acc ->
      {v, acc} = Nx.LazyContainer.traverse(v, acc, fun)
      {{k, v}, acc}
    end)
    |> then(fn {list, acc} -> {Map.new(list), acc} end)
  end
end

defimpl Nx.LazyContainer, for: Atom do
  def traverse(bool, _acc, _fun) when is_boolean(bool) do
    raise Protocol.UndefinedError,
      protocol: @protocol,
      value: bool,
      description:
        "booleans are not valid tensors (and therefore not supported as defn inputs). " <>
          "However, you can convert them to tensors using Nx.tensor/1"
  end

  def traverse(atom, _acc, _fun) do
    raise Protocol.UndefinedError,
      protocol: @protocol,
      value: atom
  end
end

defimpl Nx.LazyContainer, for: List do
  def traverse(list, _acc, _fun) do
    raise Protocol.UndefinedError,
      protocol: @protocol,
      value: list,
      description:
        "lists are not valid tensors (and therefore not supported as defn inputs). " <>
          "However, you can convert them to tensors using Nx.tensor/1"
  end
end

defimpl Nx.LazyContainer, for: Any do
  def traverse(data, acc, fun) do
    Nx.Container.traverse(data, acc, &Nx.LazyContainer.traverse(&1, &2, fun))
  end
end
