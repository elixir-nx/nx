defmodule Nx.Defn.Donated do
  @moduledoc false

  # Marks a value (tensor or container) so its leaves may be donated at the
  # next JIT/compile boundary. See `Nx.Defn.donate/1`.

  defstruct [:value]

  @donate_depth_key {__MODULE__, :depth}

  @doc false
  def depth_key, do: @donate_depth_key

  @doc false
  def enter do
    Process.put(@donate_depth_key, Process.get(@donate_depth_key, 0) + 1)
    :ok
  end

  @doc false
  def leave do
    case Process.get(@donate_depth_key, 0) do
      n when n > 1 -> Process.put(@donate_depth_key, n - 1)
      _ -> Process.delete(@donate_depth_key)
    end

    :ok
  end

  @doc false
  def donating? do
    Process.get(@donate_depth_key, 0) > 0
  end
end

defimpl Nx.LazyContainer, for: Nx.Defn.Donated do
  def traverse(%Nx.Defn.Donated{value: value}, acc, fun) do
    Nx.Defn.Donated.enter()

    try do
      Nx.LazyContainer.traverse(value, acc, fun)
    after
      Nx.Defn.Donated.leave()
    end
  end
end

defimpl Nx.Container, for: Nx.Defn.Donated do
  def traverse(%Nx.Defn.Donated{value: value}, acc, fun) do
    Nx.Container.traverse(value, acc, fun)
  end

  def reduce(%Nx.Defn.Donated{value: value}, acc, fun) do
    Nx.Container.reduce(value, acc, fun)
  end

  def serialize(%Nx.Defn.Donated{}), do: raise("cannot serialize donated containers")
  def deserialize(_, _), do: raise("cannot deserialize donated containers")
end
