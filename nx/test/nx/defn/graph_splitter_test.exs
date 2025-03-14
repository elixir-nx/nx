defmodule Nx.Defn.GraphSplitterTest do
  use ExUnit.Case, async: true

  alias Nx.Defn.GraphSplitter
  alias Nx.Defn.GraphSplitter.Stage

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

      split_fn = fn
        %T{data: %Expr{op: :dot}} -> true
        _ -> false
      end

      {chain, cache, state} = GraphSplitter.traverse_and_return_cache(expr, split_fn)

      assert [
               %Stage{
                 id: stage_0_id,
                 expr: stage_0_expr,
                 argument_sources: stage_0_argument_sources
               },
               %Stage{
                 id: _stage_1_id,
                 expr: stage_1_expr,
                 argument_sources: stage_1_argument_sources
               }
             ] = chain

      assert Enum.all?(stage_0_argument_sources, fn {_id, {source_id, _idx}} ->
               source_id == nil
             end)

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

      split_fn = fn
        %T{data: %Expr{op: :dot}} -> true
        %T{data: %Expr{op: :sum}} -> true
        _ -> false
      end

      {chain, cache, state} = GraphSplitter.traverse_and_return_cache(expr, split_fn)

      assert [
               %Stage{
                 id: stage_0_id,
                 expr: stage_0_expr,
                 argument_sources: stage_0_argument_sources
               },
               %Stage{
                 id: stage_1_id,
                 expr: stage_1_expr,
                 argument_sources: stage_1_argument_sources
               },
               %Stage{
                 id: _stage_2_id,
                 expr: stage_2_expr,
                 argument_sources: stage_2_argument_sources
               }
             ] = chain

      assert Enum.all?(stage_0_argument_sources, fn {_id, {source_id, _idx}} ->
               source_id == nil
             end)

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

      assert %{arg_2_id => {nil, 2}, arg_5_id => {stage_1_id, 0}} == stage_2_argument_sources
    end

    test "supports optional callbacks" do
      arg0 =
        Nx.u8([
          [1, 0, 1],
          [1, 1, 1]
        ])

      expr =
        Nx.Defn.debug_expr(fn a, b ->
          x = Nx.add(b, 1)
          y = Nx.sum(x, axes: [1])
          z = Nx.logical_not(y)
          Nx.subtract(z, a)
        end).(1, arg0)

      split_fn = fn
        %T{data: %Expr{op: :sum}} -> true
        _ -> false
      end

      assert [%Stage{} = stage_0, %Stage{} = stage_1] = GraphSplitter.traverse(expr, split_fn)

      [{arg1_id, %T{shape: {2, 3}, type: {:u, 8}, data: %Expr{args: [0]}}}] =
        Enum.to_list(stage_0.arguments)

      assert stage_0.argument_sources == %{arg1_id => {nil, 1}}

      stage_1_args =
        Enum.sort_by(stage_1.arguments, fn {_id, %T{data: %Expr{op: :parameter, args: [idx]}}} ->
          idx
        end)

      assert [
               {arg_0_id, %T{shape: {}, type: {:s, 32}}},
               {arg_1_id, %T{shape: {2, 3}, type: {:u, 8}}}
             ] =
               stage_1_args

      assert stage_1.argument_sources == %{arg_0_id => {nil, 0}, arg_1_id => {stage_0.id, 0}}

      assert %T{data: %Expr{op: :subtract, args: [c, d]}} = stage_1.expr
      assert %T{data: %Expr{op: :optional, args: [call, subexpr, _fun]}} = c

      assert %T{data: %Expr{id: ^arg_0_id, op: :parameter, args: [0]}} = d

      assert %T{data: %Expr{op: :logical_not, args: [b]}} = call
      assert %T{data: %Expr{op: :sum, args: [a, [axes: [1], keep_axes: false]]}} = b
      assert %T{data: %Expr{id: ^arg_1_id, op: :parameter, args: [1]}} = a

      assert %T{
               data: %Expr{
                 op: :equal,
                 args: [
                   %T{data: %Expr{id: subexpr_arg_0_id, op: :parameter, args: [0]}},
                   %T{data: %Expr{op: :constant, args: [0]}}
                 ]
               }
             } = subexpr

      # ensure subexpr is hermetic
      assert subexpr_arg_0_id != arg_0_id
      assert subexpr_arg_0_id != arg_1_id
    end

    test "supports in-line anonymous functions" do
      arg0 =
        Nx.u8([
          [1, 0, 1],
          [1, 1, 1]
        ])

      expr =
        Nx.Defn.debug_expr(fn a, b ->
          x = Nx.add(b, 1)
          y = Nx.sum(x, axes: [1])
          f = fn a -> Nx.equal(a, 0) end
          z = f.(y)
          Nx.subtract(z, a)
        end).(1, arg0)

      split_fn = fn
        %T{data: %Expr{op: :sum}} -> true
        _ -> false
      end

      assert [%Stage{} = stage_0, %Stage{} = stage_1] = GraphSplitter.traverse(expr, split_fn)

      [{arg1_id, %T{shape: {2, 3}, type: {:u, 8}, data: %Expr{args: [0]}}}] =
        Enum.to_list(stage_0.arguments)

      assert stage_0.argument_sources == %{arg1_id => {nil, 1}}

      stage_1_args =
        Enum.sort_by(stage_1.arguments, fn {_id, %T{data: %Expr{op: :parameter, args: [idx]}}} ->
          idx
        end)

      assert [
               {arg_0_id, %T{shape: {}, type: {:s, 32}}},
               {arg_1_id, %T{shape: {2, 3}, type: {:u, 8}}}
             ] =
               stage_1_args

      assert stage_1.argument_sources == %{arg_0_id => {nil, 0}, arg_1_id => {stage_0.id, 0}}

      assert %T{data: %Expr{op: :subtract, args: [c, d]}} = stage_1.expr

      assert %T{
               data: %Expr{
                 op: :equal,
                 args: [
                   left,
                   %T{data: %Expr{op: :constant, args: [0]}}
                 ]
               }
             } = c

      assert %T{data: %Expr{id: ^arg_0_id, op: :parameter, args: [0]}} = d

      assert %T{data: %Expr{op: :sum, args: [a, [axes: [1], keep_axes: false]]}} = left
      assert %T{data: %Expr{id: ^arg_1_id, op: :parameter, args: [1]}} = a
    end
  end
end
