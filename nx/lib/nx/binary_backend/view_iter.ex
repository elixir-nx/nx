defmodule Nx.BinaryBackend.ViewIter do
  alias Nx.BinaryBackend.ViewIter
  alias Nx.BinaryBackend.Traverser

  defstruct [:trav]

  def build(trav), do: %ViewIter{trav: trav}

  defimpl Enumerable do
    def count(%ViewIter{trav: %Traverser{size: size, cycle_size: cycle_size}}) do
      {:ok, div(size, cycle_size)}
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

    def reduce(view, {:suspend, acc}, fun) do
      {:suspended, acc, fn acc2 -> reduce(view, acc2, fun) end}
    end

    def reduce(%ViewIter{trav: trav1} = view, {:cont, acc}, fun) do
      case Traverser.next_view(trav1) do
        {:cont, view_i, trav2} ->
          reduce(%ViewIter{view | trav: trav2}, fun.(view_i, acc), fun)

        :done ->
          {:done, acc}
      end
    end
  end
end
