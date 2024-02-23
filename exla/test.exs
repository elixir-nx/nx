Nx.Defn.default_options(compiler: EXLA, compiler_mode: :mlir)
t = Nx.tensor([-1, 0, 1])

defmodule M do
  import Nx.Defn

  defn cond3(a, b, c) do
    d = Nx.sum(a)

    cond do
      Nx.all(Nx.greater(a, 0)) -> b * c * d
      Nx.all(Nx.less(a, 0)) -> b + c + d
      true -> -b - c - d
    end
  end

  defn cond2(a, b, c) do
    cond do
      Nx.all(Nx.greater(a, 0)) -> b * c
      true -> b + c
    end
  end
end

IO.gets("enter to continue -- PID: #{System.pid()}")

IO.puts("=== Cond 2 === ")
M.cond2(t, Nx.tensor(2), Nx.tensor(3.0)) |> IO.inspect()
M.cond2(Nx.tensor([1, 2, 3]), Nx.tensor(2), Nx.tensor(3.0)) |> IO.inspect()

IO.puts("=== Cond 3 === ")
M.cond3(t, Nx.tensor(2), Nx.tensor(3.0)) |> IO.inspect()
M.cond3(Nx.tensor([-1, -2, -3]), Nx.tensor(2), Nx.tensor(3.0)) |> IO.inspect()
M.cond3(Nx.tensor([1, 2, 3]), Nx.tensor(2), Nx.tensor(3.0)) |> IO.inspect()
