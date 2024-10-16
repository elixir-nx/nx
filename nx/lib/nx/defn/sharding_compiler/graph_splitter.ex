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
      num_args: 0
    }

    cache = %{}
    {expr, {_cache, state}} = composite_eval(expr, state, cache)

    {expr, state}
  end

  defp composite_eval(expr, state, cache) do
    Composite.traverse(expr, {cache, state}, &eval/2)
  end

  defp eval(%T{data: %Expr{id: id, op: op}} = ans, {cache, state}) do
    dbg({op, Map.has_key?(cache, id), state.nodes_to_replace})

    case {cache, state.nodes_to_replace} do
      {%{^id => res}, _} ->
        {res, {cache, state}}

      {_, %{^id => res}} ->
        # Replace the node with the corresponding parameter
        {res, {cache, state}}

      {_, _} ->
        cond do
          op in @gather_ops ->
            rewrite_args(op, ans, :gather, {cache, state})

          op in @reduction_ops ->
            rewrite_args(op, ans, :reduce, {cache, state})

          true ->
            eval_apply(op, ans, {cache, state})
        end
    end
  end

  defp eval(other, {cache, state}) do
    {other, {cache, state}}
  end

  defp rewrite_args(op, expr, category, {cache, state}) when op in [:dot] do
    {args, {cache, state}} = Nx.Defn.Tree.apply_args(expr, {cache, state}, &eval/2)

    {args, {tensor_args, state}} =
      Enum.map_reduce(args, {[], state}, fn
        %T{} = expr, {tensor_args, state} ->
          arg = Expr.parameter(expr, state.num_args)

          state = %{
            state
            | num_args: state.num_args + 1,
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
        &[{category, List.to_tuple(Enum.reverse(tensor_args))} | &1]
      )

    cache = Map.put(cache, new_expr.data.id, new_expr)

    {new_expr, {cache, state}}
  end

  defp eval_apply(:parameter, %T{data: %Expr{id: id}} = expr, {cache, state}) do
    state = update_in(state.num_args, &(&1 + 1))
    {expr, {Map.put(cache, id, expr), state}}
  end

  defp eval_apply(:elem, %T{data: %Expr{id: id, args: [tuple, i]}}, {cache, state}) do
    {tuple, cache} = composite_eval(tuple, state, cache)
    res = elem(tuple, i)
    {res, {Map.put(cache, id, res), state}}
  end

  defp eval_apply(op, %T{data: %Expr{id: id}} = ans, {cache, state}) do
    {args, {cache, state}} = Nx.Defn.Tree.apply_args(ans, {cache, state}, &eval/2)

    args =
      Enum.map(args, fn
        %T{data: %Expr{id: id}} = arg ->
          case state.nodes_to_replace do
            %{^id => new_arg} -> new_arg
            _ -> arg
          end

        arg ->
          arg
      end)

    ans = put_in(ans.data.args, args)
    {ans, {Map.put(cache, id, ans), state}}
  end
end
