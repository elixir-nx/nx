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
  @indices_ops [:slice, :put_slice]

  @impl true
  def __stream__(_key, input, acc, vars, fun, [args], opts) do
    count = Nx.Defn.Composite.count(input) + Nx.Defn.Composite.count(acc)
    rest_params = Enum.drop(args, count)
    hooks = Keyword.get(opts, :hooks, %{})
    gc? = Keyword.get(opts, :garbage_collect, true)
    {expr, meta, cache} = precompile(fun, vars, hooks)

    [
      Nx.Defn.Stream.start_link(input, acc, fn input_params, acc ->
        acc_params = [acc] |> Nx.Defn.Composite.flatten_list() |> Enum.map(&fn -> &1 end)
        params = input_params ++ acc_params ++ rest_params

        expr
        |> composite_eval(%{params: params, gc: gc?, parent: nil, meta: meta}, cache)
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
    {expr, meta, cache} = precompile(fun, vars, hooks)

    fn [params] ->
      [expr |> composite_eval(%{params: params, gc: gc?, parent: nil, meta: meta}, cache) |> elem(0)]
    end
  end

  defp precompile(fun, vars, hooks) do
    expr = fun.(vars)
    state = %{hooks: hooks}
    {meta, cache} = composite_compute_cache(expr, state, {%{}, %{}})
    {expr, meta, cache}
  end

  defp composite_compute_cache(expr, state, acc) do
    Composite.reduce(expr, acc, &compute_cache(&1, state, &2))
  end

  defp compute_cache(%Nx.Tensor{data: %Expr{op: :metadata, args: [expr, _meta]}}, state, acc) do
    compute_cache(expr, state, acc)
  end

  defp compute_cache(%Nx.Tensor{data: %Expr{id: id, op: op}} = tensor, state, {meta, cache}) do
    case cache do
      %{^id => counter} -> {meta, %{cache | id => counter + 1}}
      %{} -> compute_cache(op, tensor, state, meta, Map.put(cache, id, 1))
    end
  end

  defp bump_counter(cache, key) do
    case cache do
      %{^key => counter} -> %{cache | key => counter + 1}
      %{} -> Map.put(cache, key, 1)
    end
  end

  defp compute_cache(:fun, %{data: %Expr{id: id, args: args}}, state, meta, cache) do
    [_args, expr, _mfa] = args
    {meta, fun_cache} = composite_compute_cache(expr, state, {meta, %{}})
    {Map.put(meta, id, fun_cache), cache}
  end

  defp compute_cache(:while, %{data: %Expr{args: args, id: id}}, state, meta, cache) do
    [initial, _arg, pred, block] = args
    {meta, cache} = composite_compute_cache(initial, state, {meta, cache})

    while_cache = %{}
    {meta, while_cache} = compute_cache(pred, state, {meta, while_cache})
    {meta, while_cache} = composite_compute_cache(block, state, {meta, while_cache})

    {Map.put(meta, id, while_cache), cache}
  end

  defp compute_cache(:optional, %{data: %Expr{args: args, id: id}}, state, meta, cache) do
    [expr, default_impl_expr] = args
    {meta, cache} = Enum.reduce(expr.data.args, {meta, cache}, &compute_cache(&1, state, &2))
    {meta, optional_cache} = composite_compute_cache(default_impl_expr, state, {meta, %{}})
    {Map.put(meta, id, optional_cache), cache}
  end

  defp compute_cache(:token, %{data: %Expr{args: [token], id: id}}, state, meta, cache) do
    hooks = state.hooks

    {hooks, {meta, cache}} =
      Enum.map_reduce(token.hooks, {meta, cache}, fn
        %{callback: callback, expr: expr, name: name}, acc ->
          hook_fun = hooks[name] || callback

          cond do
            hook_fun -> {hook_fun, composite_compute_cache(expr, state, acc)}
            Tree.has_hooks?(expr, hooks) -> {true, composite_compute_cache(expr, state, acc)}
            true -> {false, acc}
          end
      end)

    {Map.put(meta, id, hooks), cache}
  end

  defp compute_cache(:cond, %{data: %Expr{args: [clauses, last], id: id}}, state, meta, cache) do
    {clauses_cache, meta} = Enum.map_reduce(clauses, meta, fn clause, meta ->
      # TODO: We should be able to reuse the cache somehow here
      {meta, cache} = composite_compute_cache(clause, state, {meta, %{}})
      {cache, meta}
    end)

    {meta, last_cache} = composite_compute_cache(last, state, {meta, %{}})

    # Now compute all IDs used in the if (but remove cons pairs which are used for metadata).
    # They are incremented by one now and decremented by one at the end of every cond.
    all_ids =
      clauses_cache
      |> Enum.reduce(last_cache, &Map.merge/2)
      |> Map.keys()
      |> Enum.reject(&is_list/1)

    cache = Enum.reduce(all_ids, cache, &bump_counter(&2, &1))
    {Map.put(meta, id, {clauses_cache, last_cache, all_ids}), cache}
  end

  defp compute_cache(_op, tensor, state, meta, cache) do
    {_, acc} = Tree.apply_args(tensor, {meta, cache}, &{&1, compute_cache(&1, state, &2)})
    acc
  end

  ## Evaluation

  defp eval(%Nx.Tensor{data: %Expr{op: :tensor, args: [t]}}, _state, cache) do
    {t, cache}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :metadata, args: [expr, _meta]}}, state, cache) do
    eval(expr, state, cache)
  end

  defp eval(%Nx.Tensor{data: %Expr{op: op, id: id}} = ans, state, cache) do
    case cache do
      %{^id => 0} ->
        raise "trying to read evaluator cache that has expired during expression:\n\n#{inspect(ans)}\n\n" <>
                "Please report this bug with the relevant code that triggers it: https://github.com/elixir-nx/nx"

      %{^id => count} when is_integer(count) ->
        {res, cache} = eval_apply(op, ans, state, cache)
        state.gc && :erlang.garbage_collect(self())
        {res, decrement_cache(cache, id, count, res)}

      %{^id => {count, res}} ->
        {res, decrement_cache(cache, id, count, res)}
    end
  end

  defp eval(other, _state, cache) do
    {other, cache}
  end

  defp decrement_cache(cache, id, 1, _res), do: %{cache | id => 0}
  defp decrement_cache(cache, id, counter, res), do: %{cache | id => {counter - 1, res}}

  defp eval_apply(:parameter, %{data: %Expr{args: [i]}}, state, cache) do
    case Enum.fetch!(state.params, i).() do
      %Nx.Tensor{data: %Nx.Defn.Expr{}} = tensor ->
        raise ArgumentError,
              "cannot pass a tensor expression as argument to defn, got: #{inspect(tensor)}"

      %Nx.Tensor{} = tensor ->
        {tensor, cache}
    end
  end

  defp eval_apply(:elem, %Nx.Tensor{data: %Expr{args: [tuple, i]}}, state, cache) do
    {tuple, cache} = composite_eval(tuple, state, cache)
    {elem(tuple, i), cache}
  end

  defp eval_apply(:attach_token, %Nx.Tensor{data: %Expr{args: [token, expr]}}, state, cache) do
    {_, cache} = eval(token, state, cache)
    eval(expr, state, cache)
  end

  defp eval_apply(:fun, %{data: %Expr{args: [args, expr, _mfa], id: id}}, state, cache) do
    fun_cache = Map.fetch!(state.meta, id)

    fun =
      case length(args) do
        1 ->
          fn arg1 ->
            params = [fn -> Nx.to_tensor(arg1) end]
            {result, _cache} = composite_eval(expr, %{state | params: params}, fun_cache)
            result
          end

        2 ->
          fn arg1, arg2 ->
            params = [fn -> Nx.to_tensor(arg1) end, fn -> Nx.to_tensor(arg2) end]
            {result, _cache} = composite_eval(expr, %{state | params: params}, fun_cache)
            result
          end
      end

    {fun, cache}
  end

  defp eval_apply(:cond, %{data: %Expr{args: [clauses, last], id: id}}, state, cache) do
    {clauses_cache, last_cache, add_ids} = Map.fetch!(state.meta, id)
    state = %{state | parent: cache}
    {chosen, chosen_cache} = cond_clause(clauses, clauses_cache, last, last_cache, state)
    {res, res_cache} = composite_eval(chosen, state, chosen_cache)

    cache =
      Enum.reduce(add_ids, cache, fn id, cache ->
        case cache do
          # Value was cached, decrement it after cond
          %{^id => {count, value}} -> decrement_cache(cache, id, count, value)
          # Value was not cached but it was only used in cond
          %{^id => 1} -> %{cache | id => 0}
          # Value was not cached but it is meant to be used elsewhere...
          %{^id => count} when is_integer(count) ->
            case res_cache do
              # and we computed it in cond, so we cache it!
              # TODO: Do not expire cond caches if value is used upstream
              %{^id => {_, value}} -> decrement_cache(cache, id, count, value)
              # and we did not compute it in cond, so we only decrement it!
              %{} -> %{cache | id => count - 1}
            end
        end
      end)

    {res, cache}
  end

  defp eval_apply(:while, %{data: %Expr{args: args, id: id}}, state, cache) do
    [initial, _arg, condition, block] = args
    {initial, cache} = composite_eval(initial, state, cache)
    while_cache = Map.fetch!(state.meta, id)
    {while(initial, condition, block, state, while_cache), cache}
  end

  defp eval_apply(:token, %{data: %Expr{args: [token], id: id}}, state, cache) do
    hooks = Map.fetch!(state.meta, id)

    cache =
      token.hooks
      |> Enum.zip(hooks)
      |> List.foldr(cache, fn
        {%{expr: expr}, true}, cache ->
          {_expr, cache} = composite_eval(expr, state, cache)
          cache

        {%{}, false}, cache ->
          cache

        {%{expr: expr}, hook_fun}, cache ->
          {res, cache} = composite_eval(expr, state, cache)
          hook_fun.(res)
          cache
      end)

    {{}, cache}
  end

  defp eval_apply(:optional, %{data: %Expr{args: args, id: id}}, state, cache) do
    [expr, default_impl_expr] = args

    {args, cache} = Tree.apply_args(expr, cache, &eval(&1, state, &2))
    backend = Nx.Shared.list_impl!(args)

    if function_exported?(backend, expr.data.op, length(args) + 1) do
      {apply(backend, expr.data.op, [expr | args]), cache}
    else
      params = Enum.map(args, &fn -> &1 end)
      optional_cache = Map.fetch!(state.meta, id)
      {res, _cache} = eval(default_impl_expr, %{state | params: params}, optional_cache)
      {res, cache}
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
          {_backend, backend_options} = Nx.default_backend()
          {Nx.Shared.list_impl!(args), [ans | args] ++ [backend_options]}

        op in @list_ops ->
          {Nx.Shared.list_impl!(hd(args)), [ans | args]}

        op in @indices_ops ->
          [tensor, indices | _] = args
          {Nx.Shared.list_impl!([tensor | indices]), [ans | args]}

        match?({:tuple, _}, ans.type) ->
          {Nx.Shared.list_impl!(args), args}

        true ->
          {Nx.Shared.list_impl!(args), [ans | args]}
      end

    {apply(mod, op, args), cache}
  end

  ## Control flow helpers

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

  defp cond_clause([{pred, clause} | clauses], [cache | caches], last, last_cache, state) do
    {pred, cache} = eval(pred, state, cache)

    if Nx.to_number(pred) != 0,
      do: {clause, cache},
      else: cond_clause(clauses, caches, last, last_cache, state)
  end

  defp cond_clause([], [], last, last_cache, _state) do
    {last, last_cache}
  end

  ## Composite

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
    [fn -> other end | acc]
  end
end
