# Minimal test to debug hook + cond cache issue
Mix.install([])

Code.require_file("nx/lib/nx.ex", __DIR__)
Code.require_file("nx/lib/nx/defn.ex", __DIR__)
Code.require_file("nx/lib/nx/defn/evaluator.ex", __DIR__)

defmodule DebugHookCond do
  import Nx.Defn

  defn simple_hook_cond(bool, a, b) do
    # This should create one expression with a hook
    res = hook(a + b, :example, fn x -> IO.inspect(x, label: "Hook called") end)
    
    # Use res in a cond
    if bool do
      res
    else
      -res
    end
  end
end

# Test it
IO.puts("\n=== Testing with bool=1 ===")
result1 = DebugHookCond.simple_hook_cond(Nx.tensor(1), Nx.tensor(4), Nx.tensor(5))
IO.inspect(result1, label: "Result1")

IO.puts("\n=== Testing with bool=0 ===")
result2 = DebugHookCond.simple_hook_cond(Nx.tensor(0), Nx.tensor(4), Nx.tensor(5))
IO.inspect(result2, label: "Result2")
