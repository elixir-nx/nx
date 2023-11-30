defmodule M do
    import Nx.Defn
  defn f(t, x) do
    pred = t ==  1
    cond do
        pred ->
            t + 10 + pred
        true ->
            x - 20
    end
  end
end

# EXLA.jit(&M.f/1, compiler_mode: :xla).(Nx.tensor(1)) |> IO.inspect()
EXLA.jit(&M.f/2, compiler_mode: :mlir).(Nx.tensor(1), Nx.tensor(10)) |> IO.inspect()
