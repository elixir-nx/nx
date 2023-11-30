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

  defn g(t, x) do
    cond do
        t == 1  ->
          t + x

        t == 2 ->
          -t

        true ->
            x - 20
    end
  end
end

# EXLA.jit(&M.f/1, compiler_mode: :xla).(Nx.tensor(1)) |> IO.inspect()
f = EXLA.jit(&M.f/2, compiler_mode: :mlir)


f.(Nx.tensor(1), Nx.tensor(10))|> IO.inspect(label: "f")
f.(Nx.tensor(0), Nx.tensor(10))|> IO.inspect(label: "f")


g = EXLA.jit(&M.g/2, compiler_mode: :mlir)

g.(Nx.tensor(1), Nx.tensor(10))|> IO.inspect(label: "g")
g.(Nx.tensor(2), Nx.tensor(10))|> IO.inspect(label: "g")
g.(Nx.tensor(3), Nx.tensor(10))|> IO.inspect(label: "g")
