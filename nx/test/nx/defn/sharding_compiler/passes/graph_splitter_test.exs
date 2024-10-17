defmodule Nx.Defn.ShardingCompiler.Passes.GraphSplitterTest do
  use ExUnit.Case, async: true

  alias Nx.Defn.ShardingCompiler.Passes.GraphSplitter
  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  describe "traverse/1" do
    test "simple expression with 1 split and no common nodes" do
      expr =
        Nx.Defn.debug_expr(fn arg0, arg1 ->
          x = Nx.add(arg0, arg1)
          y = Nx.subtract(arg0, arg1)
          z = Nx.dot(x, y)
          w = Nx.multiply(z, 2)
          Nx.divide(w, 4)
        end).(Nx.tensor([1, 2]), Nx.tensor([3, 4]))

      {chain, state, cache} = GraphSplitter.traverse(expr)

      assert [
               {stage_0_id, :gather, stage_0_expr, stage_0_argument_sources},
               {_stage_1_id, :none, stage_1_expr, stage_1_argument_sources}
             ] = chain

      assert Enum.all?(stage_0_argument_sources, fn {_id, source} -> source == nil end)

      assert [{2, arg_2_original_node_id, arg_2_id}, {3, arg_3_original_node_id, arg_3_id}] =
               state.nodes_to_replace
               |> Enum.map(fn {original_node_id,
                               %T{data: %Expr{id: id, op: :parameter, args: [idx]}}} ->
                 {idx, original_node_id, id}
               end)
               |> Enum.sort()

      # ensure that arg2 and arg3 map to the correct stage and output container position
      assert %{
               arg_2_id => {stage_0_id, 0},
               arg_3_id => {stage_0_id, 1}
             } ==
               stage_1_argument_sources

      # ensure that arg2 and arg3 are replacing the correct nodes
      {_dot_node_id, %T{data: %Expr{args: [dot_arg_0, _, _, dot_arg_1, _, _]}}} =
        Enum.find(cache, fn
          {_, %T{data: %Expr{op: :dot}}} -> true
          _ -> false
        end)

      assert dot_arg_0.data.id == arg_2_id
      assert dot_arg_1.data.id == arg_3_id

      # ensure that the output of the first stage contains the original nodes from dot(x, y)
      # also assert on the rough shape for the expression
      assert {%T{data: %Expr{id: ^arg_2_original_node_id}} = left,
              %T{data: %Expr{id: ^arg_3_original_node_id}} = right} = stage_0_expr

      assert %T{
               data: %Expr{
                 op: :add,
                 args: [
                   %T{data: %Expr{op: :parameter, args: [0]}},
                   %T{data: %Expr{op: :parameter, args: [1]}}
                 ]
               }
             } = left

      assert %T{
               data: %Expr{
                 op: :subtract,
                 args: [
                   %T{data: %Expr{op: :parameter, args: [0]}},
                   %T{data: %Expr{op: :parameter, args: [1]}}
                 ]
               }
             } = right

      assert %T{
               data: %Expr{
                 op: :divide,
                 args: [
                   %T{
                     data: %Expr{
                       op: :multiply,
                       args: [
                         %T{data: %Expr{op: :constant, args: [2]}},
                         %T{
                           data: %Expr{
                             op: :dot,
                             args: [
                               %T{data: %Expr{op: :parameter, args: [0]}},
                               [0],
                               [],
                               %T{data: %Expr{op: :parameter, args: [1]}},
                               [0],
                               []
                             ]
                           }
                         }
                       ]
                     }
                   },
                   %T{data: %Expr{op: :constant, args: [4]}}
                 ]
               }
             } = stage_1_expr
    end

    test "expression with 2 splits, common nodes and argument separation" do
      expr =
        Nx.Defn.debug_expr(fn arg0, arg1, arg2 ->
          x = Nx.add(arg0, arg1)
          y = Nx.subtract(arg0, arg1)
          z = Nx.dot(x, y)
          w = Nx.multiply(z, 2)
          a = Nx.sum(w)

          a
          |> Nx.add(w)
          |> Nx.subtract(arg2)
        end).(Nx.tensor([[1, 2]]), Nx.tensor([[3], [4]]), Nx.tensor([5, 6]))

      {chain, state, cache} = GraphSplitter.traverse(expr)

      assert [
               {stage_0_id, :gather, stage_0_expr, stage_0_argument_sources},
               {stage_1_id, :reduce, stage_1_expr, stage_1_argument_sources},
               {_stage_2_id, :none, stage_2_expr, stage_2_argument_sources}
             ] = chain

      assert Enum.all?(stage_0_argument_sources, fn {_id, source} -> source == nil end)

      assert map_size(state.args) == 6

      original_args =
        Enum.reduce(state.args, [], fn {id, _}, acc ->
          if node = cache[id] do
            [{hd(node.data.args), id} | acc]
          else
            acc
          end
        end)
        |> Enum.sort()
        |> Enum.map(fn {_, id} -> id end)

      [arg_0_id, arg_1_id, arg_2_id] = original_args

      assert [
               {2, arg_3_original_node_id, arg_3_id},
               {3, arg_4_original_node_id, arg_4_id},
               {4, arg_5_original_node_id, arg_5_id}
             ] =
               state.nodes_to_replace
               |> Enum.map(fn {original_node_id,
                               %T{data: %Expr{id: id, op: :parameter, args: [idx]}}} ->
                 {idx, original_node_id, id}
               end)
               |> Enum.sort()

      assert arg_3_id not in original_args
      assert arg_4_id not in original_args
      assert arg_5_id not in original_args

      # ensure that arg3 and arg4 map to the correct stage and output container position
      assert %{
               arg_3_id => {stage_0_id, 0},
               arg_4_id => {stage_0_id, 1}
             } ==
               stage_1_argument_sources

      # ensure that arg3 and arg4 are replacing the correct nodes
      {_dot_node_id, %T{data: %Expr{args: [dot_arg_0, _, _, dot_arg_1, _, _]}}} =
        Enum.find(cache, fn
          {_, %T{data: %Expr{op: :dot}}} -> true
          _ -> false
        end)

      assert dot_arg_0.data.id == arg_3_id
      assert dot_arg_1.data.id == arg_4_id

      # ensure that the output of the first stage contains the original nodes from dot(x, y)
      # also assert on the rough shape for the expression
      assert {%T{data: %Expr{id: ^arg_3_original_node_id}} = left,
              %T{data: %Expr{id: ^arg_4_original_node_id}} = right} = stage_0_expr

      assert %T{
               data: %Expr{
                 op: :add,
                 args: [
                   %T{data: %Expr{id: ^arg_0_id, op: :parameter, args: [0]}},
                   %T{data: %Expr{id: ^arg_1_id, op: :parameter, args: [1]}}
                 ]
               }
             } = left

      assert %T{
               data: %Expr{
                 op: :subtract,
                 args: [
                   %T{data: %Expr{id: ^arg_0_id, op: :parameter, args: [0]}},
                   %T{data: %Expr{id: ^arg_1_id, op: :parameter, args: [1]}}
                 ]
               }
             } = right

      assert {%T{
                data: %Expr{
                  id: ^arg_5_original_node_id,
                  op: :multiply,
                  args: [
                    %T{data: %Expr{op: :constant, args: [2]}},
                    %T{
                      data: %Expr{
                        op: :dot,
                        args: [
                          %T{data: %Expr{op: :parameter, args: [0]}},
                          [1],
                          [],
                          %T{data: %Expr{op: :parameter, args: [1]}},
                          [0],
                          []
                        ]
                      }
                    }
                  ]
                }
              }} = stage_1_expr

      assert %T{data: %Expr{op: :subtract, args: [c, d]}} = stage_2_expr
      assert %T{data: %Expr{op: :add, args: [b, a]}} = c
      assert %T{data: %Expr{id: ^arg_2_id, op: :parameter, args: [0]}} = d
      assert %T{data: %Expr{op: :sum, args: [^a, [axes: nil, keep_axes: false]]}} = b
      assert %T{data: %Expr{id: ^arg_5_id, op: :parameter, args: [1]}} = a

      assert %{arg_2_id => nil, arg_5_id => {stage_1_id, 0}} == stage_2_argument_sources
    end
  end
end
