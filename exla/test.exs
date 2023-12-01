:os.getpid() |> IO.inspect(label: "PID: ")
IO.gets("Press enter")

defmodule M do
  import Nx.Defn

  defn f(x) do
    while {i = 0, x}, i < 10 do
      {i + 1, x + x}
    end
  end
end

EXLA.jit(&M.f/1, compiler_mode: :mlir).(Nx.tensor([-1, 0, 1]))
|> IO.inspect()
