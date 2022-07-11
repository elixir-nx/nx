defmodule Nx.Defn.Evaluator do
  @moduledoc """
  The default implementation of a `Nx.Defn.Compiler`
  that evaluates the expression tree against the
  tensor backend.
  """

  @behaviour Nx.Defn.Compiler
  alias Nx.Defn.{Composite, Expr, Tree}

  @creation_ops [:constant, :eye, :iota, :from_binary]
  @random_ops [:random_uniform, :random_normal]
  @list_ops [:concatenate]

  @impl true
  def __stream__(_key, input, acc, vars, fun, [args], opts) do
    count = Nx.Defn.Composite.count(input) + Nx.Defn.Composite.count(acc)
    hooks = Keyword.get(opts, :hooks, %{})
    gc? = Keyword.get(opts, :garbage_collect, true)
    expr = fun.(vars)

    [
      Nx.Defn.Stream.start_link(input, acc, fn input, acc ->
        params = Nx.Defn.Composite.flatten_runtime_args([input, acc], Enum.drop(args, count))

        expr
        |> composite_eval(%{params: params, hooks: hooks, gc: gc?}, %{})
        |> elem(0)
      end)
    ]
  end

  @impl true
  def __jit__(key, vars, fun, args_list, opts) do
    __compile__(key, vars, fun, opts).(args_list)
  end

  @impl true
  def __compile__(_key, vars, fun, opts) do
    hooks = Keyword.get(opts, :hooks, %{})
    gc? = Keyword.get(opts, :garbage_collect, true)
    expr = fun.(vars)

    fn [params] ->
      [
        expr
        |> composite_eval(%{params: params, hooks: hooks, gc: gc?}, %{})
        |> elem(0)
      ]
    end
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :parameter, args: [i]}}, state, cache) do
    {Enum.fetch!(state.params, i), cache}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :tensor, args: [t]}}, _state, cache) do
    {t, cache}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :elem, args: args}}, state, cache) do
    [tuple, i] = args
    {tuple, cache} = composite_eval(tuple, state, cache)
    {elem(tuple, i), cache}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :attach_token, args: [token, expr]}}, state, cache) do
    {_, cache} = eval(token, state, cache)
    eval(expr, state, cache)
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :metadata, args: [expr, _meta]}}, state, cache) do
    eval(expr, state, cache)
  end

  defp eval(%Nx.Tensor{data: %Expr{op: op, id: id}} = ans, state, cache) do
    case cache do
      %{^id => res} ->
        {res, cache}

      %{} ->
        {res, cache} = eval_apply(op, ans, state, cache)
        state.gc && :erlang.garbage_collect(self())
        {res, Map.put(cache, id, res)}
    end
  end

  defp eval(other, _state, cache) do
    {other, cache}
  end

  defp eval_apply(:fun, %{data: %Expr{args: [args, expr, _mfa]}}, state, cache) do
    fun =
      case length(args) do
        1 ->
          fn arg1 ->
            params = [Nx.to_tensor(arg1)]
            {result, _cache} = composite_eval(expr, %{state | params: params}, %{})
            result
          end

        2 ->
          fn arg1, arg2 ->
            params = [Nx.to_tensor(arg1), Nx.to_tensor(arg2)]
            {result, _cache} = composite_eval(expr, %{state | params: params}, %{})
            result
          end
      end

    {fun, cache}
  end

  defp eval_apply(:cond, %{data: %Expr{args: [clauses, last]}}, state, cache) do
    {res, cache} = cond_clause(clauses, last, state, cache)
    composite_eval(res, state, cache)
  end

  defp eval_apply(:while, %{data: %Expr{args: args}}, state, cache) do
    [initial, _arg, condition, block] = args
    {initial, cache} = composite_eval(initial, state, cache)
    {while(initial, condition, block, state, cache), cache}
  end

  defp eval_apply(:token, %{data: %Expr{args: [token]}}, state, cache) do
    hooks = state.hooks

    cache =
      List.foldr(token.hooks, cache, fn %{callback: callback, expr: expr, name: name}, cache ->
        hook_fun = hooks[name] || callback

        cond do
          hook_fun ->
            {expr, cache} = composite_eval(expr, state, cache)
            hook_fun.(expr)
            cache

          Tree.has_hooks?(expr, hooks) ->
            {_expr, cache} = composite_eval(expr, state, cache)
            cache

          true ->
            cache
        end
      end)

    {{}, cache}
  end

  defp eval_apply(:optional, %{data: %Expr{args: [expr, default_impl_expr]}}, state, cache) do
    # The arguments are shared between expr and default_impl_expr nodes,
    # so we don't do extra work regardless of the branch we choose.
    {args, cache} = Tree.apply_args(expr, cache, &eval(&1, state, &2))
    backend = Nx.Shared.list_impl!(args)

    if function_exported?(backend, expr.data.op, length(args) + 1) do
      {apply(backend, expr.data.op, [expr | args]), cache}
    else
      eval(default_impl_expr, state, cache)
    end
  end

  defp eval_apply(op, ans, state, cache) do
    {args, cache} = Tree.apply_args(ans, cache, &eval(&1, state, &2))

    {mod, args} =
      cond do
        op in @creation_ops ->
          {backend, backend_options} = Nx.default_backend()
          {backend, [ans | args] ++ [backend_options]}

        op in @random_ops ->
          {_, backend_options} = Nx.default_backend()
          {Nx.Shared.list_impl!(args), [ans | args] ++ [backend_options]}

        op in @list_ops ->
          {Nx.Shared.list_impl!(hd(args)), [ans | args]}

        match?({:tuple, _}, ans.type) ->
          {Nx.Shared.list_impl!(args), args}

        true ->
          {Nx.Shared.list_impl!(args), [ans | args]}
      end

    {apply(mod, op, args), cache}
  end

  defp while(acc, condition, block, state, cache) do
    state = %{state | params: composite_to_params(acc)}
    {pred, temp} = eval(condition, state, cache)

    if Nx.to_number(pred) != 0 do
      {acc, _} = composite_eval(block, state, temp)
      while(acc, condition, block, state, cache)
    else
      acc
    end
  end

  defp composite_eval(composite, state, cache) do
    Composite.traverse(composite, cache, &eval(&1, state, &2))
  end

  defp composite_to_params(composite) do
    composite |> composite_to_params([]) |> Enum.reverse()
  end

  defp composite_to_params(tuple, acc) when is_tuple(tuple) do
    Enum.reduce(Tuple.to_list(tuple), acc, &composite_to_params/2)
  end

  defp composite_to_params(other, acc) do
    [other | acc]
  end

  defp cond_clause([{pred, clause} | clauses], last, state, cache) do
    {pred, cache} = eval(pred, state, cache)

    if Nx.to_number(pred) != 0,
      do: {clause, cache},
      else: cond_clause(clauses, last, state, cache)
  end

  defp cond_clause([], last, _state, cache) do
    {last, cache}
  end
end
