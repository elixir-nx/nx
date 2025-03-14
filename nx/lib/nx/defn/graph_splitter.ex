defmodule Nx.Defn.GraphSplitter do
  alias Nx.Defn.Composite

  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr
  alias Nx.Defn.GraphSplitter.Stage

  @doc """
  Traverses the expression and splits it into stages.
  """
  def traverse(expr, expr_split_fn \\ fn _ -> false end) do
    {chain, _, _} = traverse_and_return_cache(expr, expr_split_fn)
    chain
  end

  @doc false
  def traverse_and_return_cache(expr, expr_split_fn) do
    # expression_chain is going to be a reverse-accumulation of {category, subexpr}
    # that we can then compile and chain-execute elsewhere. category is either :gather, :reduce or :none
    state = %{
      expression_chain: [],
      nodes_to_replace: %{},
      expr_split_fn: expr_split_fn,
      # args is a map of id -> {stage_id, output_container_position}
      args: %{}
    }

    cache = %{}
    {expr, {cache, state}} = composite_eval(expr, state, cache)

    expr_chain =
      Enum.reduce(
        [{make_ref(), expr, state.nodes_to_replace} | state.expression_chain],
        [],
        fn {id, expr, nodes_to_replace}, acc ->
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
            composite_rewrite_subtree(
              expr,
              %{state | nodes_to_replace: nodes_to_replace}
            )

          arg_remapping =
            used_args
            |> Enum.sort_by(fn {_id, %T{data: %Expr{op: :parameter, args: [idx]}}} -> idx end)
            |> Enum.with_index(fn
              {id, expr}, idx ->
                {id, put_in(expr.data.args, [idx])}
            end)
            |> Map.new()

          {expr, _} =
            composite_rewrite_subtree(expr, %{state | nodes_to_replace: arg_remapping})

          arguments =
            Map.new(arg_remapping, fn {_id, arg_expr} ->
              {arg_expr.data.id, arg_expr}
            end)

          argument_sources =
            state.args
            |> Map.take(Map.keys(arg_remapping))
            |> Map.new(fn {remap_id, v} ->
              {arg_remapping[remap_id].data.id, v}
            end)

          [
            %Stage{
              id: id,
              expr: expr,
              arguments: arguments,
              argument_sources: argument_sources
            }
            | acc
          ]
        end
      )

    # Apply topological sort to the expr_chain
    sorted_expr_chain = topological_sort(expr_chain)

    {sorted_expr_chain, cache, Map.delete(state, :expression_chain)}
  end

  defp composite_eval(expr, state, cache) do
    Composite.traverse(expr, {cache, state}, &eval/2)
  end

  defp eval(%T{data: %Expr{id: id, op: op}} = ans, {cache, state}) do
    case {cache, state.nodes_to_replace} do
      {_, %{^id => res}} ->
        # Replace the node with the corresponding parameter
        {res, {Map.put(cache, id, res), state}}

      {%{^id => res}, _} ->
        {res, {cache, state}}

      _ ->
        if state.expr_split_fn.(ans) do
          split_expr(ans, {cache, state})
        else
          eval_apply(op, ans, {cache, state})
        end
    end
  end

  defp eval(other, {cache, state}) do
    {other, {cache, state}}
  end

  defp split_expr(expr, {cache, state}) do
    {args, {cache, state}} = Nx.Defn.Tree.apply_args(expr, {cache, state}, &eval/2)
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
              nodes_to_replace: Map.put(state.nodes_to_replace, expr.data.id, arg)
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
          {stage_id, List.to_tuple(Enum.reverse(tensor_args)), nodes_to_replace}
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
        {res, put_in(acc.used_args[id], res)}

      _ ->
        {expr, put_in(acc.used_args[id], expr)}
    end
  end

  defp rewrite_subtree(
         %T{data: %Expr{op: :optional, id: id, args: [call, subexpr, fun]}} = expr,
         state,
         acc
       ) do
    case state.nodes_to_replace do
      %{^id => res} ->
        {res, put_in(acc.used_args[id], res)}

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
        {res, put_in(acc.used_args[id], res)}

      _ ->
        {args, acc} = composite_rewrite_subtree(args, state, acc)
        {put_in(expr.data.args, args), acc}
    end
  end

  defp rewrite_subtree(other, _, acc), do: {other, acc}

  defp topological_sort(expr_chain) do
    # Create a new directed graph
    graph = :digraph.new()

    # Add vertices for each stage output
    Enum.each(expr_chain, fn %Stage{id: id, arguments: arguments} ->
      Enum.with_index(arguments, fn _, idx ->
        :digraph.add_vertex(graph, {id, idx})
      end)
    end)

    # Add edges based on argument sources
    Enum.each(expr_chain, fn %Stage{id: id, argument_sources: sources} ->
      Enum.each(sources, fn {_arg_id, {source_stage_id, source_index}} ->
        if source_stage_id != nil do
          Enum.each(0..(map_size(sources) - 1)//1, fn idx ->
            :digraph.add_edge(graph, {source_stage_id, source_index}, {id, idx})
          end)
        end
      end)
    end)

    # Perform topological sort
    sorted_ids =
      graph
      |> :digraph_utils.topsort()
      |> Enum.map(fn {stage_id, _} -> stage_id end)
      |> Enum.uniq()

    # Clean up the graph
    :digraph.delete(graph)

    # Return the sorted stages based on sorted_ids
    Enum.map(sorted_ids, fn stage_id ->
      Enum.find(expr_chain, fn %Stage{id: id} -> id == stage_id end)
    end)
  end
end
