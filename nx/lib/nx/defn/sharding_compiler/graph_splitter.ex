defmodule Nx.Defn.ShardingCompiler.GraphSplitter do
  alias Nx.Defn.Composite

  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  @gather_ops [:dot]
  @reduction_ops [:sum]

  def traverse(expr) do
    # expression_chain is going to be a reverse-accumulation of {category, subexpr}
    # that we can then compile and chain-execute elsewhere. category is either :gather, :reduce or :root
    state = %{
      expression_chain: [],
      nodes_to_replace: %{},
      args: %{}
    }

    cache = %{}
    {expr, {cache, state}} = composite_eval(expr, state, cache)

    expr_chain =
      Enum.reduce(
        [{:none, expr, state.nodes_to_replace} | state.expression_chain],
        [],
        fn {category, expr, nodes_to_replace}, acc ->
          [
            {category,
             composite_rewrite_subtree(
               expr,
               {cache, %{state | nodes_to_replace: nodes_to_replace}}
             )}
            | acc
          ]
        end
      )

    {expr_chain, Map.delete(state, :expression_chain)}
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

      {_, _} ->
        cond do
          op in @gather_ops ->
            rewrite_args(ans, :gather, {cache, state})

          op in @reduction_ops ->
            rewrite_args(ans, :reduce, {cache, state})

          true ->
            eval_apply(op, ans, {cache, state})
        end
    end
  end

  defp eval(other, {cache, state}) do
    {other, {cache, state}}
  end

  defp rewrite_args(expr, category, {cache, state}) do
    {args, {cache, state}} = Nx.Defn.Tree.apply_args(expr, {cache, state}, &eval/2)

    # We need to save this so that each previous stage
    # isn't affected by following ones
    nodes_to_replace = state.nodes_to_replace

    {args, {tensor_args, state}} =
      Enum.map_reduce(args, {[], state}, fn
        %T{} = expr, {tensor_args, state} ->
          arg = Expr.parameter(expr, map_size(state.args))

          state = %{
            state
            | args: Map.put(state.args, arg.data.id, nil),
              nodes_to_replace: Map.put(state.nodes_to_replace, expr.data.id, arg)
          }

          {arg, {[expr | tensor_args], state}}

        non_tensor_arg, acc ->
          {non_tensor_arg, acc}
      end)

    new_expr = put_in(expr.data.args, args)

    state =
      update_in(
        state.expression_chain,
        &[{category, List.to_tuple(Enum.reverse(tensor_args)), nodes_to_replace} | &1]
      )

    cache = Map.put(cache, new_expr.data.id, new_expr)

    {new_expr, {cache, state}}
  end

  defp eval_apply(:parameter, %T{data: %Expr{id: id}} = expr, {cache, state}) do
    state = put_in(state.args[id], nil)
    {expr, {Map.put(cache, id, expr), state}}
  end

  defp eval_apply(:elem, %T{data: %Expr{id: id, args: [tuple, i]}}, {cache, state}) do
    {tuple, cache} = composite_eval(tuple, state, cache)
    res = elem(tuple, i)
    {res, {Map.put(cache, id, res), state}}
  end

  defp eval_apply(op, %T{data: %Expr{id: id}} = ans, {cache, state}) do
    {args, {cache, state}} = Nx.Defn.Tree.apply_args(ans, {cache, state}, &eval/2)

    # args = composite_rewrite_subtree(args, {cache, state})

    # if op == :multiply do
    #   dbg(args)
    #   dbg(state.nodes_to_replace)
    # end

    ans = put_in(ans.data.args, args)
    {ans, {Map.put(cache, id, ans), state}}
  end

  defp composite_rewrite_subtree(args, {cache, state}) when is_list(args) do
    Enum.map(args, fn
      arg when is_list(arg) ->
        arg

      arg ->
        composite_rewrite_subtree(arg, {cache, state})
    end)
  end

  defp composite_rewrite_subtree(arg, {cache, state}) do
    Composite.traverse(arg, &rewrite_subtree(&1, {cache, state}))
  end

  defp rewrite_subtree(%T{data: %Expr{id: id, args: args}} = expr, {cache, state}) do
    case state.nodes_to_replace do
      %{^id => res} ->
        res

      _ ->
        args = composite_rewrite_subtree(args, {cache, state})

        put_in(expr.data.args, args)
    end
  end

  defp rewrite_subtree(other, _), do: other
end
