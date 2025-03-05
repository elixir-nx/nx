defmodule Nx.PartitionedExecutorTest.CompiledFunctions do
  import Nx.Defn

  defn f({x, y}), do: {x + y, x - y}
  defn g({z}), do: {0, z - 1}
  defn h({f0, f1, g1}), do: {f0, f1, g1}

  def run_defn_expr(args, expr) do
    f = fn _ -> expr end

    Nx.Defn.jit_apply(f, [args])
  end
end
