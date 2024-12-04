Mix.install([{:nx, path: ".", override: true}, :axon])

model =
  Axon.input("x", shape: {1, 10})
  |> Axon.dense(10)
  |> Axon.relu()
  |> Axon.dense(10)
  |> Axon.relu()
  |> Axon.nx(&Nx.Defn.Expr.metadata(&1, %{split: true}))
  |> Axon.dense(10)
  |> Axon.sigmoid()

{init_fn, predict_fn} = Axon.build(model)

params = Nx.Defn.jit_apply(init_fn, [Nx.template({1, 10}, :f32), %{}])

expr = Nx.Defn.debug_expr(predict_fn).(params, Nx.iota({1, 10}, type: :f32))

split_fn = fn
  :metadata, [_expr, %{split: split}], _shards -> split
  _, _, _ -> false
end

alias Nx.Defn.ShardingCompiler.Passes.GraphSplitter

combine_fn = fn shards, _, _ -> shards end

split_rules =
  Map.put(
    %{},
    :metadata,
    {split_fn, combine_fn, :none}
  )

{chain, _cache, _state} = GraphSplitter.traverse(expr, %{}, split_rules)

alias Nx.Defn.ShardingCompiler

Task.async(fn ->
  result_1 = Nx.Defn.jit_apply(predict_fn, [params, Nx.iota({1, 10}, type: :f32)], compiler: ShardingCompiler, sharding_config: :disable, ops_split_rules: split_rules)
  result_2 = Nx.Defn.jit_apply(predict_fn, [params, Nx.iota({1, 10}, type: :f32)], compiler: Nx.Defn.Evaluator)

  dbg({result_1, result_2})
end)
