defmodule M do
    import Nx.Defn

end

# EXLA.jit(&M.f/1, compiler_mode: :xla).(Nx.tensor(1)) |> IO.inspect()


g = EXLA.jit(&M.g/2, compiler_mode: :mlir)

g.(Nx.tensor(1), Nx.tensor(10))|> IO.inspect(label: "g")
g.(Nx.tensor(2), Nx.tensor(10))|> IO.inspect(label: "g")
g.(Nx.tensor(3), Nx.tensor(10))|> IO.inspect(label: "g")
