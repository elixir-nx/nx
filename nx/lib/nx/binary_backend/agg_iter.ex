defmodule Nx.BinaryBackend.AggIter do
  alias Nx.BinaryBackend.AggIter
  alias Nx.BinaryBackend.Traverser

  defstruct [:trav]

  def build(trav), do: %AggIter{trav: trav}

  defimpl Enumerable do
    def count(%AggIter{trav: %Traverser{size: size, cycle_size: cycle_size}}) do
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

    def reduce(agg_list, {:suspend, acc}, fun) do
      {:suspended, acc, fn acc2 -> reduce(agg_list, acc2, fun) end}
    end

    def reduce(%AggIter{trav: trav1} = a, {:cont, acc}, fun) do
      case Traverser.next_agg(trav1) do
        {:cont, agg_trav, trav2} ->
          reduce(%AggIter{a | trav: trav2}, fun.(agg_trav, acc), fun)

        :done ->
          {:done, acc}
      end
    end
  end
end
