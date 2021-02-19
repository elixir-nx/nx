defmodule Softmax do
  import Nx.Defn

  defn softmax(n), do: Nx.exp(n) / Nx.sum(Nx.exp(n)) |> inspect_expr()
end

Nx.Defn.aot(&Softmax.softmax/1, [Nx.tensor([1, 2, 3, 4])], EXLA)

defmodule Softmax do
  import Nx.Defn

  @on_load :__on_load__

  def __on_load__ do
    :erlang.load_nif('libnif', 0)
  end

  def softmax(_) do
    :ok
  end
end

Softmax.softmax(Nx.tensor([1, 2, 3, 4]))
