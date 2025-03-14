# defmodule Nx.Defn.ShardingCompiler.ShardExecutionTest do
#   use ExUnit.Case, async: true

#   alias Nx.Defn.ShardingCompiler.Passes.GraphSplitter
#   alias Nx.Defn.ShardingCompiler.Passes.GraphSplitter.Stage
#   alias Nx.Defn.ShardingCompiler.Passes.ShardPropagation
#   alias Nx.Defn.ShardingCompiler.ShardExecution
#   alias Nx.Defn.ShardingCompiler.Shard

#   alias Nx.Tensor, as: T
#   alias Nx.Defn.Expr

#   @moduletag :skip
#   test "Creates all the necessary children for each stage" do
#     arg0 =
#       Nx.tensor([
#         [1, 2, 3],
#         [4, 5, 6]
#       ])

#     arg1 =
#       Nx.tensor([
#         [1, 2],
#         [3, 4],
#         [5, 6]
#       ])

#     fun = fn arg0, arg1 ->
#       x = Nx.add(arg0, 1)
#       y = Nx.subtract(arg1, 2)

#       Nx.multiply(x, Nx.transpose(y))
#     end

#     expected_output = fun.(arg0, arg1)

#     expr =
#       Nx.Defn.debug_expr(fun).(arg0, arg1)

#     {%T{data: %ShardPropagation{expr: sharded_expr}} = ans, _cache, %{expr_shards: expr_shards}} =
#       ShardPropagation.traverse(expr, %{
#         0 => Shard.from_config(arg0, %{0 => 1, 1 => 3}, debug_id: "arg 0"),
#         1 => Shard.from_config(arg1, %{0 => 3}, debug_id: "arg 1")
#       })

#     assert {[%Stage{} = stage0], _cache, _state} =
#              GraphSplitter.traverse(%T{ans | data: sharded_expr}, expr_shards)

#     args_by_idx = %{0 => arg0, 1 => arg1}

#     arg_providers =
#       Enum.flat_map(stage0.arguments, fn {_id, expr} ->
#         start_shard_providers(expr, args_by_idx)
#       end)

#     assert Enum.count(arg_providers, &match?({:ok, _}, &1)) == 4

#     assert {:ok, pid} = ShardExecution.Supervisor.start_link(stage0)

#     children = Supervisor.which_children(pid)

#     states =
#       Enum.map(children, fn {key, pid, :worker, [ShardExecution]} ->
#         {key, :sys.get_state(pid)}
#       end)
#       |> Enum.sort_by(fn {_, state} -> {state.output_entry_index, state.output_starts} end)

#     assert [executor0, executor1] = states

#     assert {_key0, executor0_state} = executor0
#     assert {_key1, executor1_state} = executor1

#     idx_to_id =
#       Map.new(stage0.arguments, fn
#         {id, %T{data: %Expr{op: :parameter, args: [idx]}}} ->
#           {idx, {id, nil}}

#         {id, %T{data: %Expr{op: :metadata, args: [expr, %{shards: shards}]}}} ->
#           %T{data: %Expr{op: :parameter, args: [idx]}} = expr
#           {idx, {id, shards}}
#       end)

#     assert %ShardExecution{
#              input_data_sections: [{0, input_section0}, {1, input_section1}],
#              output_starts: [0, 0],
#              output_lengths: [1, 3]
#            } = executor0_state

#     {id0, %{0 => [shard0, shard1], 1 => [shard2]}} = idx_to_id[0]
#     {id1, %{0 => [shard3], 1 => [shard4, shard5]}} = idx_to_id[1]

#     assert {id0, [shard0.id, shard2.id]} == input_section0
#     assert {id1, [shard3.id, shard4.id]} == input_section1

#     assert %ShardExecution{
#              input_data_sections: [{0, input_section0}, {1, input_section1}],
#              output_starts: [1, 0],
#              output_lengths: [1, 3]
#            } = executor1_state

#     assert {id0, [shard1.id, shard2.id]} == input_section0
#     assert {id1, [shard3.id, shard5.id]} == input_section1

#     assert executor0_state.output ==
#              Nx.tensor([
#                [-2, 3, 12]
#              ])

#     assert executor1_state.output ==
#              Nx.tensor([
#                [0, 12, 28]
#              ])

#     {:ok, output_collector_pid} =
#       ShardExecution.OutputCollector.start_link(ans, stage0.id, self())

#     assert_receive {ShardExecution.OutputCollector, :done, ^output_collector_pid, result}

#     assert expected_output == result
#   end

#   defp start_shard_providers(sharded_expr, arg_data) do
#     case sharded_expr do
#       %T{data: %Expr{op: :parameter, args: [idx]}} ->
#         [
#           ShardExecution.ArgumentProvider.start_link([
#             sharded_expr,
#             idx,
#             arg_data[idx]
#           ])
#         ]

#       %T{data: %Expr{op: :metadata, args: [%T{data: %Expr{args: [idx]}}, %{shards: shards}]}} ->
#         shards
#         |> Enum.sort_by(fn {axis, _} -> axis end)
#         |> Enum.map(fn {axis, shard} -> {shard, axis} end)
#         |> cartesian_product()
#         |> Enum.map(fn sections ->
#           {starts, lengths} =
#             sections
#             |> Enum.map(fn {shard, _axis} -> {shard.start, shard.length} end)
#             |> Enum.unzip()

#           data_section_id = Enum.map(sections, fn {shard, _axis} -> shard.id end)

#           ShardExecution.ArgumentProvider.start_link([
#             Nx.slice(arg_data[idx], starts, lengths),
#             idx,
#             data_section_id
#           ])
#         end)
#     end
#   end

#   defp cartesian_product([{data, meta} | rest]) do
#     for x <- data, y <- cartesian_product(rest), do: [{x, meta} | y]
#   end

#   defp cartesian_product([]), do: [[]]
# end
