arg0 =
  Nx.tensor([
    [1, 2, 3],
    [4, 5, 6]
  ])

arg1 =
  Nx.tensor([
    [1, 2],
    [3, 4],
    [5, 6]
  ])

fun = fn arg0, arg1 ->
  x = Nx.add(arg0, 1)
  y = Nx.subtract(arg1, 2)

  z = Nx.dot(x, y)
  # Nx.add(z, arg2)
end

Nx.Defn.jit(fun, compiler: Nx.Defn.ShardingCompiler, sharding_config: [%{0 => 1, 1 => 3}, %{0 => 3, 1 => 1}]).(arg0, arg1)
|> dbg()

dbg(fun.(arg0, arg1))
