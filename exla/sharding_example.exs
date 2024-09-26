arg0_sharding = %{0 => [0..0, 1..1, 2..2]}
arg1_sharding = %{1 => [0..0, 1..2]}

Nx.default_backend(EXLA.Backend)

fun = &Nx.add(Nx.cos(&1), Nx.sin(&2))

inputs = [
  Nx.iota({3, 1}, type: :f32),
  Nx.add(Nx.iota({1, 3}), 10)
]

{output_holder, shards} =
  Nx.Defn.jit_apply(
    fun,
    inputs,
    compiler: Nx.Defn.ShardingCompiler,
    sharding_config: [arg0_sharding, arg1_sharding],
    sharding_compiler: EXLA,
    sharding_compiler_options: []
  )

sharded_result =
  shards
  |> Task.async_stream(fn {arg, fun, caster} ->
    dbg(self())
    {fun.(arg), caster}
  end)
  |> Enum.reduce(output_holder, fn {:ok, {result, caster}}, acc ->
    caster.(result, acc)
  end)
  |> IO.inspect()

IO.inspect(Nx.equal(sharded_result, apply(fun, inputs)) |> Nx.all() |> Nx.to_number() |> Kernel.==(1))
