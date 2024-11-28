defmodule Nx.Defn.ShardingCompiler do
  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  alias Nx.Defn.ShardingCompiler.Shard

  alias Nx.Defn.ShardingCompiler.Passes.ShardPropagation
  alias Nx.Defn.ShardingCompiler.Passes.GraphSplitter
  alias Nx.Defn.ShardingCompiler.ShardExecution
  @behaviour Nx.Defn.Compiler

  @impl true
  def __jit__(_key, vars, fun, args, opts) do
    opts =
      Keyword.validate!(opts, [
        :sharding_config,
        sharding_compiler: Nx.Defn.Evaluator,
        sharding_compiler_options: []
      ])

    [args] = args

    # TODO: support containers here
    {%T{data: %ShardPropagation{expr: sharded_expr}} = ans, expr_shards} =
      propagate_shards(vars, fun, opts[:sharding_config])

    args_by_idx = Enum.with_index(args, fn arg, idx -> {idx, arg} end) |> Map.new()

    {[first_stage | _] = stages, _cache, _state} =
      GraphSplitter.traverse(%T{ans | data: sharded_expr}, expr_shards)

    Enum.flat_map(first_stage.arguments, fn {_id, expr} ->
      start_shard_providers(expr, args_by_idx)
    end)

    {last_stage, _last_stage_pid} =
      for stage <- stages, reduce: nil do
        _ ->
          {:ok, pid} = ShardExecution.Supervisor.start_link(stage)
          {stage, pid}
      end

    {:ok, output_collector_pid} =
      ShardExecution.OutputCollector.start_link(ans, last_stage.id, self())

    receive do
      {ShardExecution.OutputCollector, :done, ^output_collector_pid, result} ->
        [result]
    end
  end

  defp start_shard_providers(sharded_expr, arg_data) do
    case sharded_expr do
      %T{
        shape: {},
        data: %Expr{op: :metadata, args: [%T{data: %Expr{args: [idx]}}, %{shards: shards}]}
      }
      when map_size(shards) == 0 ->
        [
          ShardExecution.ArgumentProvider.start_link([
            arg_data[idx].(),
            idx,
            :scalar
          ])
        ]

      %T{data: %Expr{op: :metadata, args: [%T{data: %Expr{args: [idx]}}, %{shards: shards}]}} ->
        shards
        |> Enum.sort_by(fn {axis, _} -> axis end)
        |> Enum.map(fn {axis, shard} -> {shard, axis} end)
        |> cartesian_product()
        |> Enum.map(fn sections ->
          {starts, lengths} =
            sections
            |> Enum.map(fn {shard, _axis} -> {shard.start, shard.length} end)
            |> Enum.unzip()

          data_section_id = Enum.map(sections, fn {shard, _axis} -> shard.id end)

          data = arg_data[idx].()

          data_slice = Nx.slice(data, starts, lengths)

          ShardExecution.ArgumentProvider.start_link([
            data_slice,
            idx,
            data_section_id
          ])
        end)
    end
  end

  defp cartesian_product([{data, meta} | rest]) do
    for x <- data, y <- cartesian_product(rest), do: [{x, meta} | y]
  end

  defp cartesian_product([]), do: [[]]

  @impl true
  def __compile__(_key, _vars, _fun, _opts) do
    raise "Not implemented yet"
  end

  def propagate_shards(vars, fun, sharding_config) do
    expr = fun.(vars)

    arity = length(vars)

    sharding_config = sharding_config || List.duplicate(%{}, arity)

    if length(sharding_config) != arity do
      raise "Expected sharding config for function with #{arity} arguments to have the same length"
    end

    tensor_shardings =
      sharding_config
      |> Enum.zip_with(vars, fn config, var ->
        Shard.from_config(var, config)
      end)
      |> Enum.with_index(fn x, idx -> {idx, x} end)
      |> Map.new()

    {container, _cache, %{expr_shards: expr_shards}} =
      ShardPropagation.traverse(expr, tensor_shardings)

    {container, expr_shards}
  end

  @impl true
  def __partitions_options__(_keyword) do
    raise "__partitions_options__ not supported"
  end

  @impl true
  def __to_backend__(_keyword) do
    raise "__to_backend__ not supported"
  end

  def init(opts), do: opts
end
