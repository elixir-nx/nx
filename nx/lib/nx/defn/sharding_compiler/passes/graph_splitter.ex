defmodule Nx.Defn.ShardingCompiler.Passes.GraphSplitter do
  alias Nx.Defn.Composite

  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr
  alias Nx.Defn.ShardingCompiler.Shard
  alias Nx.Defn.ShardingCompiler.Passes.GraphSplitter.Stage

  def default_operation_split_rules do
    %{
      dot: {&must_split_expr?/3, &argument_combine_shards/3, :gather},
      sum: {&must_split_expr?/3, &argument_combine_shards/3, :reduce}
    }
  end

  def traverse(expr, expr_shards \\ %{}, ops_split_rules \\ nil) do
    # expression_chain is going to be a reverse-accumulation of {category, subexpr}
    # that we can then compile and chain-execute elsewhere. category is either :gather, :reduce or :none
    state = %{
      expression_chain: [],
      nodes_to_replace: %{},
      ops_split_rules: ops_split_rules || default_operation_split_rules(),
      # contains the sharding configuration for each node by id
      shards: expr_shards,
      # args is a map of id -> {stage_id, output_container_position}
      args: %{}
    }

    cache = %{}
    {expr, {cache, state}} = composite_eval(expr, state, cache)

    expr_chain =
      Enum.reduce(
        [{make_ref(), :none, expr, state.nodes_to_replace} | state.expression_chain],
        [],
        fn {id, category, expr, nodes_to_replace}, acc ->
          # TO-DO: we need to also do a pass to avoid recalculating results that have been previously calculated.
          # For example:
          # x = arg0 + arg1
          # y = arg0 - arg1
          # z = x + y
          # -----
          # w = dot(z, arg1)
          # y + w <- here, we currently have to recalculate y given that only z, arg0 and arg1 will be passed as arguments.
          #          ideally, we should also pass y as a value to avoid recalculating it.
          #          We might be able to calculate this in the first traversal somehow.

          {expr, %{used_args: used_args}} =
            composite_rewrite_subtree(expr, %{state | nodes_to_replace: nodes_to_replace})

          arg_remapping =
            used_args
            |> Enum.sort_by(fn {_id, {%T{data: %Expr{op: :parameter, args: [idx]}}, _shards}} ->
              idx
            end)
            |> Enum.with_index(fn
              {id, {expr, nil}}, idx ->
                {id, put_in(expr.data.args, [idx])}

              {id, {expr, _shard_propagation}}, idx ->
                expr = put_in(expr.data.args, [idx])
                {id, expr}
            end)
            |> Map.new()

          {expr, _} =
            composite_rewrite_subtree(expr, %{state | nodes_to_replace: arg_remapping})

          # Traverse the expression to remap all shapes according to the sharding given
          expr = set_shard_metadata(expr, state.shards)

          arguments =
            Map.new(arg_remapping, fn {_id, arg_expr} ->
              {arg_expr.data.id, set_shard_metadata(arg_expr, state.shards)}
            end)

          argument_sources = Map.take(state.args, Map.keys(arguments))

          [
            %Stage{
              id: id,
              category: category,
              expr: expr,
              arguments: arguments,
              argument_sources: argument_sources
            }
            | acc
          ]
        end
      )

    remapped_chain = remap_chain(expr_chain)

    {remapped_chain, cache, Map.delete(state, :expression_chain)}
  end

  defp composite_eval(expr, state, cache) do
    Composite.traverse(expr, {cache, state}, &eval/2)
  end

  defp eval(%T{data: %Expr{id: id, op: op}} = ans, {cache, state}) do
    case {cache, state.nodes_to_replace, state.ops_split_rules} do
      {_, %{^id => res}, _} ->
        # Replace the node with the corresponding parameter
        {res, {Map.put(cache, id, res), state}}

      {%{^id => res}, _, _} ->
        {res, {cache, state}}

      {_, _, %{^op => {split_fn, combine_fn, category}}} ->
        rewrite_args(ans, split_fn, combine_fn, category, {cache, state})

      _ ->
        eval_apply(op, ans, {cache, state})
    end
  end

  defp eval(other, {cache, state}) do
    {other, {cache, state}}
  end

  defp rewrite_args(expr, split_fn, combine_fn, category, {cache, state}) do
    {args, {cache, state}} = Nx.Defn.Tree.apply_args(expr, {cache, state}, &eval/2)

    if split_fn.(expr.data.op, args, state.shards) do
      shards = combine_fn.(state.shards, expr.data.op, args)

      state = Map.put(state, :shards, shards)

      split_expr(expr, args, category, {cache, state})
    else
      {expr, {cache, state}}
    end
  end

  defp must_split_expr?(:dot, [t0, c0, _b0, t1, c1, _b1], shards) do
    left_shards =
      case shards[t0.data.id] do
        %{shards: shards} -> shards
        _ -> nil
      end

    left_valid =
      Enum.all?(c0, fn axis ->
        case left_shards[axis] do
          [%Shard{start: 0, length: length}] -> length == elem(t0.shape, axis)
          _ -> false
        end
      end)

    right_shards =
      case shards[t1.data.id] do
        %{shards: shards} -> shards
        _ -> nil
      end

    right_valid =
      Enum.all?(c1, fn axis ->
        case right_shards[axis] do
          [%Shard{start: 0, length: length}] -> length == elem(t1.shape, axis)
          _ -> false
        end
      end)

    not (left_valid and right_valid)
  end

  # default to true so that we can optimize this gradually
  defp must_split_expr?(_, _, _), do: true

  # This function is responsible for producing a valid list of arguments (same as the original)
  # but with the shards combined properly for the given operation.
  defp argument_combine_shards(shards, :dot, [t0, c0, _b0, t1, c1, _b1])
       when is_map_key(shards, t0.data.id) and is_map_key(shards, t1.data.id) do
    shard_propagation =
      Enum.reduce(c0, shards[t0.data.id], fn axis, shard_propagation ->
        axis_shards = shard_propagation.shards[axis]

        shard = %{hd(axis_shards) | start: 0, length: elem(t0.shape, axis)}

        axis_shards =
          Shard.make_child_shards([shard], axis,
            extra_parents: axis_shards,
            from_contraction?: true
          )

        put_in(shard_propagation.shards[axis], axis_shards)
      end)

    shards = put_in(shards[t0.data.id], shard_propagation)

    shard_propagation =
      Enum.reduce(c1, shards[t1.data.id], fn axis, shard_propagation ->
        axis_shards = shard_propagation.shards[axis]

        shard = %{hd(axis_shards) | start: 0, length: elem(t1.shape, axis)}

        axis_shards =
          Shard.make_child_shards([shard], axis,
            extra_parents: axis_shards,
            from_contraction?: true
            # keep_shard_as_parent: false
          )

        put_in(shard_propagation.shards[axis], axis_shards)
      end)

    put_in(shards[t1.data.id], shard_propagation)
  end

  defp argument_combine_shards(shards, _, _) do
    shards
  end

  defp split_expr(expr, args, category, {cache, state}) do
    # We need to save this so that each previous stage
    # isn't affected by following ones
    nodes_to_replace = state.nodes_to_replace

    stage_id = make_ref()

    {args, {tensor_args, _out_position, state}} =
      Enum.map_reduce(args, {[], 0, state}, fn
        %T{} = expr, {tensor_args, out_position, state} ->
          arg = Expr.parameter(expr, map_size(state.args))

          state = %{
            state
            | args: Map.put(state.args, arg.data.id, {stage_id, out_position}),
              nodes_to_replace: Map.put(state.nodes_to_replace, expr.data.id, arg),
              shards: Map.put(state.shards, arg.data.id, state.shards[expr.data.id])
          }

          {arg, {[expr | tensor_args], out_position + 1, state}}

        non_tensor_arg, acc ->
          {non_tensor_arg, acc}
      end)

    new_expr = put_in(expr.data.args, args)

    state =
      update_in(
        state.expression_chain,
        &[
          {stage_id, category, List.to_tuple(Enum.reverse(tensor_args)), nodes_to_replace}
          | &1
        ]
      )

    cache = Map.put(cache, new_expr.data.id, new_expr)

    {new_expr, {cache, state}}
  end

  defp eval_apply(:parameter, %T{data: %Expr{id: id, args: [idx]}} = expr, {cache, state}) do
    state = put_in(state.args[id], {nil, idx})
    {expr, {Map.put(cache, id, expr), state}}
  end

  defp eval_apply(:elem, %T{data: %Expr{id: id, args: [tuple, i]}}, {cache, state}) do
    {tuple, cache} = composite_eval(tuple, state, cache)
    res = elem(tuple, i)
    {res, {Map.put(cache, id, res), state}}
  end

  defp eval_apply(_op, %T{data: %Expr{id: id}} = ans, {cache, state}) do
    {args, {cache, state}} = Nx.Defn.Tree.apply_args(ans, {cache, state}, &eval/2)
    ans = put_in(ans.data.args, args)
    {ans, {Map.put(cache, id, ans), state}}
  end

  defp composite_rewrite_subtree(container, state, acc \\ %{used_args: %{}})

  defp composite_rewrite_subtree(container, state, acc) when is_list(container) do
    Enum.map_reduce(container, acc, fn
      %T{} = arg, acc ->
        composite_rewrite_subtree(arg, state, acc)

      arg, acc when is_list(arg) ->
        composite_rewrite_subtree(arg, state, acc)

      arg, acc ->
        {arg, acc}
    end)
  end

  defp composite_rewrite_subtree(container, state, acc) do
    Composite.traverse(container, acc, &rewrite_subtree(&1, state, &2))
  end

  defp rewrite_subtree(%T{data: %Expr{id: id, op: :parameter}} = expr, state, acc) do
    case state.nodes_to_replace do
      %{^id => res} ->
        {res, put_in(acc.used_args[id], {res, state.shards[id]})}

      _ ->
        {expr, put_in(acc.used_args[id], {expr, state.shards[id]})}
    end
  end

  defp rewrite_subtree(
         %T{data: %Expr{op: :optional, id: id, args: [call, subexpr, fun]}} = expr,
         state,
         acc
       ) do
    case state.nodes_to_replace do
      %{^id => res} ->
        {res, put_in(acc.used_args[id], {res, state.shards[id]})}

      _ ->
        {call, acc} = rewrite_subtree(call, state, acc)
        # `subexpr` is hermetic, in the sense that it is a self-contained scope
        # from which the arguments always come from `call`, so we can
        # keep it as is.

        {put_in(expr.data.args, [call, subexpr, fun]), acc}
    end
  end

  defp rewrite_subtree(%T{data: %Expr{id: id, args: args}} = expr, state, acc) do
    case state.nodes_to_replace do
      %{^id => res} ->
        # nodes_to_replace always contains a param
        {res, put_in(acc.used_args[id], {res, state.shards[id]})}

      _ ->
        {args, acc} = composite_rewrite_subtree(args, state, acc)
        {put_in(expr.data.args, args), acc}
    end
  end

  defp rewrite_subtree(other, _, acc), do: {other, acc}

  defp set_shard_metadata(expr, shards) when map_size(shards) == 0, do: expr

  defp set_shard_metadata(expr, shards) do
    Composite.traverse(expr, fn
      %T{data: %Expr{id: id}} = t ->
        if shard_propagation = shards[id] do
          shape =
            shard_propagation.shards
            |> Enum.sort()
            |> Enum.map(fn
              {_axis, [%Shard{length: length} | _]} -> length
            end)
            |> List.to_tuple()

          t = do_set_shard_metadata(%{t | shape: shape}, shards)
          Expr.metadata(t, %{shards: shard_propagation.shards})
        else
          do_set_shard_metadata(t, shards)
        end

      other ->
        other
    end)
  end

  defp do_set_shard_metadata(%T{data: %Expr{args: args}} = expr, shards) do
    args =
      Enum.map(args, fn
        %T{} = arg ->
          set_shard_metadata(arg, shards)

        arg when is_list(arg) ->
          Enum.map(arg, &do_set_shard_metadata(&1, shards))

        arg ->
          arg
      end)

    put_in(expr.data.args, args)
  end

  defp do_set_shard_metadata(other, _), do: other

  defp remap_chain(expr_chain) do
    Enum.flat_map(expr_chain, fn
      %Stage{category: :gather} = stage ->
        gather_stage(stage)

      %Stage{category: :reduce} = stage ->
        # not implemented yet
        [stage]

      %Stage{category: :none} = stage ->
        [stage]
    end)
  end

  defp gather_stage(%Stage{} = input_stage) do
    require IEx

    # TODO: we need to:
    # 1. get the shards for each argument (they will either be a normal [shards] list or
    #    a tuple of {[contracted], [parent_shards]})
    # 2. write a function that takes this list of shards and concatenates them along the contracted dimensions
    # 3. make it so that this new function is the argument source for the corresponding arguments in the input stage
    # 4. turn this function into a new intermediate collection stage

    {gather_stages, stage} =
      Enum.map_reduce(input_stage.arguments, input_stage, fn {arg_id, %T{type: type} = arg},
                                                             stage ->
        case arg.data do
          %Nx.Defn.Expr{op: :metadata, args: [_, %{shards: shards}]} ->
            any_contracted? =
              Enum.any?(shards, fn
                {_axis, [%Shard{from_contraction?: from_contraction?, parents: _}]} ->
                  from_contraction?

                _ ->
                  false
              end)

            if any_contracted? do
              output_shape =
                shards
                |> Enum.map(fn
                  {_axis, [shard | _]} -> shard.length
                end)
                |> List.to_tuple()

              gather_args =
                shards
                |> Enum.to_list()
                |> cartesian_product()
                |> Enum.flat_map(fn sections ->
                  {concatenation_arguments_set, _concatenation_axes_reverse} =
                    sections
                    |> Enum.map_reduce([], fn
                      {axis, %Shard{from_contraction?: true, parents: parents}}, acc ->
                        {{axis, parents}, [axis | acc]}

                      {axis, %Shard{} = shard}, acc ->
                        {{axis, [shard]}, acc}
                    end)

                  cartesian_product(concatenation_arguments_set)
                end)
                |> Enum.with_index(fn shards, index ->
                  arg_shape =
                    Enum.map(shards, fn {_axis, shard} -> shard.length end) |> List.to_tuple()

                  arg = Expr.parameter(%{arg | shape: arg_shape}, :root, index)

                  Expr.metadata(arg, %{
                    shards: Map.new(shards, fn {axis, shard} -> {axis, [shard]} end)
                  })
                end)

              expr =
                Nx.with_default_backend(Expr, fn ->
                  output_holder =
                    Nx.broadcast(Nx.tensor(0, type: type), output_shape, names: arg.names)

                  Enum.reduce(gather_args, output_holder, fn arg, acc ->
                    %T{data: %Expr{op: :metadata, args: [_slice, %{shards: shards}]}} = arg
                    starts = Enum.map(shards, fn {_axis, [shard]} -> shard.start end)
                    Nx.put_slice(acc, starts, arg)
                  end)
                end)

              gather_expr_shards =
                Map.new(shards, fn
                  {axis, shards} ->
                    {axis, shards}
                end)

              expr = Expr.metadata(expr, %{shards: gather_expr_shards})

              gather_stage = %Stage{
                id: make_ref(),
                category: :none,
                expr: expr,
                arguments: Map.new(gather_args, &{&1.data.id, &1}),
                argument_sources:
                  Map.new(gather_args, fn %T{data: %Nx.Defn.Expr{id: id}} ->
                    {id, stage.argument_sources[arg_id]}
                  end)
              }

              modified_stage =
                update_in(stage.expr, fn expr ->
                  Nx.Defn.Composite.traverse(expr, fn
                    %T{data: %Expr{op: :metadata, args: [subexpr, %{shards: shards}]}} = expr ->
                      put_in(expr.data.args, [
                        subexpr,
                        %{
                          shards:
                            Map.new(shards, fn {axis, shards} ->
                              {axis, shards}
                            end)
                        }
                      ])

                    other ->
                      other
                  end)
                end)

              modified_stage =
                put_in(modified_stage.argument_sources[arg_id], {gather_stage.id, 0})

              # modified_stage =
              #   update_in(modified_stage.arguments[arg_id], fn arg ->
              #     %T{data: %Expr{op: :metadata, args: [slice, %{shards: shards}]}} = arg

              #     put_in(arg.data.args, [slice, %{shards: shards}])
              #   end)

              {gather_stage, modified_stage}
            else
              {[], stage}
            end

          _ ->
            {[], stage}
        end
      end)

    List.flatten([stage | gather_stages])
  end

  defp cartesian_product([{axis, entries} | rest]) do
    for x <- entries, y <- cartesian_product(rest), do: [{axis, x} | y]
  end

  defp cartesian_product([]), do: [[]]
end
