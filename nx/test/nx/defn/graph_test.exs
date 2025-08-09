defmodule Nx.Defn.GraphTest do
  use ExUnit.Case, async: true

  alias Nx.Defn.Graph
  alias Nx.Defn.Graph.Stage

  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  doctest Nx.Defn.Graph

  describe "split/2" do
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
        %T{data: %Expr{op: :dot}}, acc -> {true, acc}
        _, acc -> {false, acc}
      end

      {chain, cache, state} = Graph.__split__(expr, nil, split_fn)

      assert [
               %Stage{
                 id: stage_0_id,
                 expr: stage_0_expr,
                 arguments: stage_0_arguments
               },
               %Stage{
                 id: _stage_1_id,
                 expr: stage_1_expr,
                 arguments: stage_1_arguments
               }
             ] = chain

      assert [%{source: {nil, 0}}, %{source: {nil, 1}}] == stage_0_arguments

      assert [{2, arg_2_original_node_id, arg_2_id}, {3, arg_3_original_node_id, arg_3_id}] =
               state.nodes_to_replace
               |> Enum.map(fn {original_node_id,
                               %T{data: %Expr{id: id, op: :parameter, args: [idx]}}} ->
                 {idx, original_node_id, id}
               end)
               |> Enum.sort()

      # ensure that arg2 and arg3 map to the correct stage and output container position
      assert [%{source: {stage_0_id, 0}}, %{source: {stage_0_id, 1}}] == stage_1_arguments

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
        %T{data: %Expr{op: :dot}}, acc -> {true, acc}
        %T{data: %Expr{op: :sum}}, acc -> {true, acc}
        _, acc -> {false, acc}
      end

      {chain, cache, state} = Graph.__split__(expr, nil, split_fn)

      assert [
               %Stage{
                 id: stage_0_id,
                 expr: stage_0_expr,
                 arguments: stage_0_arguments
               },
               %Stage{
                 id: stage_1_id,
                 expr: stage_1_expr,
                 arguments: stage_1_arguments
               },
               %Stage{
                 id: _stage_2_id,
                 expr: stage_2_expr,
                 arguments: stage_2_arguments
               }
             ] = chain

      assert [%{source: {nil, 0}}, %{source: {nil, 1}}] == stage_0_arguments

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
      assert [%{source: {stage_0_id, 0}}, %{source: {stage_0_id, 1}}] == stage_1_arguments

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

      assert [%{source: {nil, 2}}, %{source: {stage_1_id, 0}}] == stage_2_arguments
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

      assert [%Stage{} = stage_0, %Stage{} = stage_1] = Graph.split(expr, split_fn)

      assert stage_0.arguments == [%{source: {nil, 1}}]
      assert stage_1.arguments == [%{source: {nil, 0}}, %{source: {stage_0.id, 0}}]

      assert %T{data: %Expr{op: :subtract, args: [c, d]}} = stage_1.expr
      assert %T{data: %Expr{op: :optional, args: [call, subexpr, _fun]}} = c

      assert %T{data: %Expr{id: arg_0_id, op: :parameter, args: [0]}} = d

      assert %T{data: %Expr{op: :logical_not, args: [b]}} = call
      assert %T{data: %Expr{op: :sum, args: [a, [axes: [1], keep_axes: false]]}} = b
      assert %T{data: %Expr{id: arg_1_id, op: :parameter, args: [1]}} = a

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

      assert [%Stage{} = stage_0, %Stage{} = stage_1] = Graph.split(expr, split_fn)

      assert [%{source: {nil, 1}}] == stage_0.arguments

      assert [%{source: {nil, 0}}, %{source: {stage_0.id, 0}}] == stage_1.arguments
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

      assert %T{data: %Expr{op: :parameter, args: [0]}} = d

      assert %T{data: %Expr{op: :sum, args: [a, [axes: [1], keep_axes: false]]}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}} = a
    end

    test "supports splitting on tuples with metadata" do
      expr =
        Nx.Defn.debug_expr(fn x ->
          y = Nx.add(x, 1)
          z = Nx.add(x, 2)
          w = {Nx.add(y, 3), Nx.add(z, 4)}
          {a, b} = Nx.Defn.Expr.metadata(w, %{split: true})
          Nx.add(a, b)
        end).(Nx.tensor([1, 2, 3]))

      split_fn = fn
        %T{data: %Expr{op: :metadata, args: [_expr, %{split: true}]}} -> true
        _ -> false
      end

      assert [%Stage{} = stage_0, %Stage{} = stage_1] = Graph.split(expr, split_fn)

      assert [%{source: {nil, 0}}] = stage_0.arguments
      assert {add_y, add_z} = stage_0.expr

      assert %T{data: %Expr{op: :add, args: [%T{data: %Expr{op: :constant, args: [4]}}, y]}} =
               add_y

      assert %T{data: %Expr{op: :parameter, args: [0]}} = y

      assert %T{data: %Expr{op: :add, args: [%T{data: %Expr{op: :constant, args: [6]}}, ^y]}} =
               add_z

      assert stage_1.arguments == [%{source: {stage_0.id, 0}}, %{source: {stage_0.id, 1}}]
      assert %T{data: %Expr{op: :add, args: [add_y, add_z]}} = stage_1.expr

      assert %T{data: %Expr{op: :parameter, args: [0]}} = add_y
      assert %T{data: %Expr{op: :parameter, args: [1]}} = add_z
    end
  end

  describe "split/3" do
    test "splits with accumulator" do
      expr =
        Nx.Defn.debug_expr(fn x0, x1, x2, x3, x4 ->
          x10 = Nx.add(x0, Nx.add(x1, x2))
          x20 = Nx.add(x10, Nx.add(x10, x3))
          x30 = Nx.add(x20, Nx.add(x20, x4))
          {x10, x20, x30}
        end).(1, 2, 3, 4, 5)

      split_fn = fn
        _node, acc ->
          {acc > 0 and rem(acc, 2) == 0, acc + 1}
      end

      chain = Graph.split(expr, 0, split_fn)

      assert [stage_0, stage_1, stage_2] = chain

      assert stage_0.arguments == [%{source: {nil, 0}}, %{source: {nil, 1}}, %{source: {nil, 2}}]

      assert {
               %T{
                 data: %Expr{
                   op: :add,
                   args: [
                     %T{data: %Expr{op: :parameter, args: [0]}},
                     %T{
                       data: %Expr{
                         args: [
                           %T{data: %Expr{op: :parameter, args: [1]}},
                           %T{data: %Expr{op: :parameter, args: [2]}}
                         ],
                         op: :add
                       }
                     }
                   ]
                 }
               }
             } = stage_0.expr

      assert stage_1.arguments == [
               %{source: {nil, 3}},
               %{source: {stage_0.id, 0}}
             ]

      assert {
               %T{
                 data: %Expr{
                   op: :add,
                   args: [
                     %T{data: %Expr{op: :parameter, args: [1]}},
                     %T{
                       data: %Expr{
                         args: [
                           %T{data: %Expr{op: :parameter, args: [1]}},
                           %T{data: %Expr{op: :parameter, args: [0]}}
                         ],
                         op: :add
                       }
                     }
                   ]
                 }
               }
             } = stage_1.expr

      assert stage_2.arguments == [
               %{source: {nil, 4}},
               %{source: {stage_0.id, 0}},
               %{source: {stage_1.id, 0}}
             ]

      assert {%T{data: %Expr{op: :parameter, args: [1]}},
              %T{data: %Expr{op: :parameter, args: [2]}},
              %T{
                data: %Expr{
                  op: :add,
                  args: [
                    %T{data: %Expr{op: :parameter, args: [2]}},
                    %T{
                      data: %Expr{
                        args: [
                          %T{data: %Expr{op: :parameter, args: [2]}},
                          %T{data: %Expr{op: :parameter, args: [0]}}
                        ],
                        op: :add
                      }
                    }
                  ]
                }
              }} = stage_2.expr
    end
  end

  describe "run/2" do
    test "executes the stages chain and returns the correct result" do
      function = fn arg0, arg1 ->
        # root
        x = Nx.multiply(arg0, arg1) |> Nx.Defn.Expr.metadata(%{split: true})

        # left side
        w_left = Nx.multiply(x, arg1) |> Nx.Defn.Expr.metadata(%{split: true})

        # right side
        w_right = Nx.divide(x, arg1) |> Nx.Defn.Expr.metadata(%{split: true})

        # merge
        Nx.add(w_right, w_left)
      end

      args = [Nx.tensor([1, 2]), Nx.tensor([3, 4])]

      # This is used in the final assertion of this test
      expected_result = Nx.Defn.jit_apply(function, args)

      expr = apply(Nx.Defn.debug_expr(function), args)

      split_fn = fn
        %T{data: %Expr{op: :metadata, args: [_expr, %{split: true}]}} -> true
        _ -> false
      end

      chain = Graph.split(expr, split_fn)

      assert [root, right, left, merge] = chain

      assert {%T{data: %Expr{op: :multiply, args: [arg0, arg1]}}} = root.expr
      assert %T{data: %Expr{op: :parameter, args: [0]}} = arg0
      assert %T{data: %Expr{op: :parameter, args: [1]}} = arg1

      # left should depend on exactly the same parameters as the root, as it's pulling from
      # the global scope
      assert {%T{data: %Expr{op: :multiply, args: [x, arg1_left]}}} = left.expr

      assert %T{
               data: %Expr{
                 op: :metadata,
                 args: [
                   %T{data: %Expr{op: :parameter, args: [1]}},
                   %{split: true}
                 ]
               }
             } = x

      assert %T{data: %Expr{op: :parameter, args: [0]}} = arg1_left

      assert Enum.fetch!(left.arguments, 0).source == {nil, 1}
      assert Enum.fetch!(left.arguments, 1).source == {root.id, 0}

      # right should depend on the result of the root and on arg1, but arg1 will be reindexed
      # we should assert that the argument source for arg1_right is correct
      assert {%T{data: %Expr{op: :divide, args: [x, arg1_right]}}} = right.expr

      assert %T{
               data: %Expr{
                 op: :metadata,
                 args: [
                   %T{data: %Expr{op: :parameter, args: [1]}},
                   %{split: true}
                 ]
               }
             } = x

      assert %T{data: %Expr{op: :parameter, args: [0]}} = arg1_right

      assert Enum.fetch!(right.arguments, 0).source == {nil, 1}
      assert Enum.fetch!(right.arguments, 1).source == {root.id, 0}

      assert %T{data: %Expr{op: :add, args: [w_right, w_left]}} = merge.expr

      assert %T{
               data: %Expr{
                 op: :metadata,
                 args: [
                   %T{data: %Expr{op: :parameter, args: [0]}},
                   %{split: true}
                 ]
               }
             } = w_right

      assert %T{
               data: %Expr{
                 op: :metadata,
                 args: [
                   %T{data: %Expr{op: :parameter, args: [1]}},
                   %{split: true}
                 ]
               }
             } = w_left

      assert Enum.fetch!(merge.arguments, 0).source == {right.id, 0}
      assert Enum.fetch!(merge.arguments, 1).source == {left.id, 0}

      assert Graph.run(chain, args) == expected_result
    end
  end
end
