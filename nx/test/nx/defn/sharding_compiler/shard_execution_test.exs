defmodule Nx.Defn.ShardingCompiler.ShardExecutionTest do
  use ExUnit.Case, async: true

  alias Nx.Defn.ShardingCompiler.Passes.GraphSplitter
  alias Nx.Defn.ShardingCompiler.Passes.GraphSplitter.Stage
  alias Nx.Defn.ShardingCompiler.Passes.ShardPropagation
  alias Nx.Defn.ShardingCompiler.ShardExecution
  alias Nx.Defn.ShardingCompiler.Shard

  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  test "Creates all the necessary children for each stage" do
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

    expr =
      Nx.Defn.debug_expr(fn arg0, arg1 ->
        x = Nx.add(arg0, 1)
        y = Nx.subtract(arg1, 2)
        z = Nx.dot(x, y)
        w = Nx.multiply(z, 3)
        Nx.divide(w, 4)
      end).(arg0, arg1)

    {sharded_expr, _cache, %{expr_shards: expr_shards}} =
      ShardPropagation.traverse(expr, %{
        0 => Shard.from_config(arg0, %{0 => [0..0, 1..1], 1 => [0..2]}),
        1 => Shard.from_config(arg1, %{})
      })

    assert {[stage0, stage1], _cache, _state} =
             GraphSplitter.traverse(expr, expr_shards)

    assert {:ok, pid} = ShardExecution.Supervisor.start_link(stage0)

    flunk("incomplete test")
  end
end
