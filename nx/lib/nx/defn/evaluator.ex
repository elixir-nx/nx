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
        |> composite_eval(%{params: params, gc: gc?, meta: meta}, [cache])
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
      [
        expr
        |> composite_eval(%{params: params, gc: gc?, meta: meta}, [cache])
        |> elem(0)
      ]
    end
  end

  defp precompile(fun, vars, hooks) do
    expr = fun.(vars)
    state = %{hooks: hooks, parent_ids: nil, current_ids: nil}
    {meta, cache} = init_compute_cache(expr, state, %{}, %{})
    {expr, meta, cache}
  end

  defp init_compute_cache(expr, state, meta, parent_ids) do
    state = %{state | parent_ids: parent_ids, current_ids: collect_ids(expr, parent_ids)}
    composite_compute_cache(expr, state, {meta, %{}})
  end

  defp collect_ids(expr, ids) do
    Composite.reduce(expr, ids, &collect_ids_each/2)
  end

  defp collect_ids_each(%Nx.Tensor{data: %Expr{id: id}} = t, ids) do
    case ids do
      %{^id => _} ->
        ids

      %{} ->
        ids = Map.put(ids, id, 0)
        # Do not collect ids when a new lexical construct starts
        Tree.apply_args(t, :lexical, ids, &{&1, collect_ids_each(&1, &2)}) |> elem(1)
    end
  end

  defp composite_compute_cache(expr, state, acc) do
    Composite.reduce(expr, acc, &compute_cache(&1, state, &2))
  end

  defp compute_cache(%Nx.Tensor{data: %Expr{op: :metadata, args: [expr, _meta]}}, state, acc) do
    compute_cache(expr, state, acc)
  end

  defp compute_cache(%Nx.Tensor{data: %Expr{id: id, op: op}} = tensor, state, {meta, cache}) do
    case state.parent_ids do
      # If the id exists in the parent, the parent will compute it.
      %{^id => _} ->
        {meta, Map.put_new(cache, id, tensor)}

      %{} ->
        case cache do
          %{^id => counter} -> {meta, %{cache | id => counter + 1}}
          %{} -> compute_cache(op, tensor, state, meta, Map.put(cache, id, 1))
        end
    end
  end

  defp compute_cache(:fun, %{data: %Expr{id: id, args: args}}, state, meta, cache) do
    [_args, expr, _mfa] = args
    {meta, fun_cache} = init_compute_cache(expr, state, meta, %{})
    {Map.put(meta, id, fun_cache), cache}
  end

  defp compute_cache(:while, %{data: %Expr{args: args, id: id}}, state, meta, cache) do
    [initial, _arg, pred, block] = args
    {meta, cache} = composite_compute_cache(initial, state, {meta, cache})
    {meta, while_cache} = init_compute_cache({pred, block}, state, meta, %{})
    {Map.put(meta, id, while_cache), cache}
  end

  defp compute_cache(:optional, %{data: %Expr{args: args, id: id}}, state, meta, cache) do
    [expr, default_impl_expr] = args
    {meta, cache} = Enum.reduce(expr.data.args, {meta, cache}, &compute_cache(&1, state, &2))
    {meta, optional_cache} = init_compute_cache(default_impl_expr, state, meta, %{})
    {Map.put(meta, id, optional_cache), cache}
  end

  defp compute_cache(:cond, %{data: %Expr{args: [clauses, last], id: id}}, state, meta, cache) do
    current_ids = state.current_ids

    {clause_caches, meta} =
      Enum.map_reduce([last | clauses], meta, fn clause, meta ->
        state = %{state | parent_ids: current_ids, current_ids: collect_ids(clause, current_ids)}
        {meta, cache} = composite_compute_cache(clause, state, {meta, %{}})
        {cache, meta}
      end)

    # Now, for each cache, split the IDs from parents from the actual cond IDs
    {[last_cache | clauses_cache], {all_ids, {meta, cache}}} =
      Enum.map_reduce(clause_caches, {%{}, {meta, cache}}, fn clause_cache, seen_ids_acc ->
        {clause_cache, seen_ids_acc} =
          Enum.flat_map_reduce(clause_cache, seen_ids_acc, fn
            {id, %_{} = tensor}, {seen_ids, acc} ->
              case seen_ids do
                # We have already processed this id for the whole cond
                %{^id => _} ->
                  {[], {seen_ids, acc}}

                # Process the ID which exists outside of cond
                %{} ->
                  {[], {Map.put(seen_ids, id, true), composite_compute_cache(tensor, state, acc)}}
              end

            {id, counter}, seen_ids_acc ->
              {[{id, counter}], seen_ids_acc}
          end)

        {Map.new(clause_cache), seen_ids_acc}
      end)

    {Map.put(meta, id, {clauses_cache, last_cache, Map.keys(all_ids)}), cache}
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

  defp compute_cache(_op, tensor, state, meta, cache) do
    {_, acc} = Tree.apply_args(tensor, {meta, cache}, &{&1, compute_cache(&1, state, &2)})
    acc
  end

  ## Evaluation

  defp eval(%Nx.Tensor{data: %Expr{op: :tensor, args: [t]}}, _state, caches) do
    {t, caches}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :metadata, args: [expr, _meta]}}, state, caches) do
    eval(expr, state, caches)
  end

  defp eval(%Nx.Tensor{data: %Expr{op: op, id: id}} = ans, state, [cache | caches]) do
    case cache do
      %{^id => count} when is_integer(count) ->
        {res, [cache | caches]} = eval_apply(op, ans, state, [cache | caches])
        state.gc && :erlang.garbage_collect(self())
        {res, [decrement_cache(cache, id, count, res) | caches]}

      %{^id => {count, res}} ->
        {res, [decrement_cache(cache, id, count, res) | caches]}

      %{} ->
        # If we don't find the tensor in the current scope,
        # it may be in a parent scope, so look up. If we find
        # it in the parent, we don't decrement it, as that will
        # be done at the end of processing the current scope.
        eval_parent(caches, id, op, ans, state, [cache])
    end
  end

  defp eval(other, _state, [_ | _] = caches) do
    {other, caches}
  end

  defp decrement_cache(cache, id, 1, _res), do: Map.delete(cache, id)
  defp decrement_cache(cache, id, counter, res), do: %{cache | id => {counter - 1, res}}

  defp eval_parent([cache | caches], id, op, ans, state, acc) do
    case cache do
      %{^id => {_count, res}} ->
        {res, Enum.reverse(acc, [cache | caches])}

      %{^id => count} when is_integer(count) ->
        {res, [cache | caches]} = eval_apply(op, ans, state, [cache | caches])
        state.gc && :erlang.garbage_collect(self())
        {res, Enum.reverse(acc, [Map.put(cache, id, {count, res}) | caches])}

      %{} ->
        eval_parent(caches, id, op, ans, state, [cache | acc])
    end
  end

  defp eval_parent([], _id, _op, ans, _state, _acc) do
    raise "trying to read evaluator cache that has expired during expression:\n\n#{inspect(ans)}\n\n" <>
            "Please report this bug with the relevant code that triggers it: https://github.com/elixir-nx/nx"
  end

  defp decrement_parents([cache | caches], id) do
    case cache do
      %{^id => {count, value}} -> [decrement_cache(cache, id, count, value) | caches]
      %{^id => count} -> [%{cache | id => count - 1} | caches]
      %{} -> [cache | decrement_parents(caches, id)]
    end
  end

  defp eval_apply(:parameter, %{data: %Expr{args: [i]}}, state, caches) do
    case Enum.fetch!(state.params, i).() do
      %Nx.Tensor{data: %Nx.Defn.Expr{}} = tensor ->
        raise ArgumentError,
              "cannot pass a tensor expression as argument to defn, got: #{inspect(tensor)}"

      %Nx.Tensor{} = tensor ->
        {tensor, caches}
    end
  end

  defp eval_apply(:elem, %Nx.Tensor{data: %Expr{args: [tuple, i]}}, state, caches) do
    {tuple, caches} = composite_eval(tuple, state, caches)
    {elem(tuple, i), caches}
  end

  defp eval_apply(:attach_token, %Nx.Tensor{data: %Expr{args: [token, expr]}}, state, caches) do
    {_, caches} = eval(token, state, caches)
    eval(expr, state, caches)
  end

  defp eval_apply(:fun, %{data: %Expr{args: [args, expr, _mfa], id: id}}, state, caches) do
    fun_cache = Map.fetch!(state.meta, id)

    fun =
      case length(args) do
        1 ->
          fn arg1 ->
            params = [fn -> Nx.to_tensor(arg1) end]
            {result, _} = composite_eval(expr, %{state | params: params}, [fun_cache])
            result
          end

        2 ->
          fn arg1, arg2 ->
            params = [fn -> Nx.to_tensor(arg1) end, fn -> Nx.to_tensor(arg2) end]
            {result, _} = composite_eval(expr, %{state | params: params}, [fun_cache])
            result
          end
      end

    {fun, caches}
  end

  defp eval_apply(:cond, %{data: %Expr{args: [clauses, last], id: id}}, state, caches) do
    {clauses_cache, last_cache, parent_ids} = Map.fetch!(state.meta, id)

    {chosen, chosen_cache} =
      clauses
      |> Enum.zip(clauses_cache)
      |> cond_clause(last, last_cache, state, caches)

    {res, [_ | caches]} = composite_eval(chosen, state, chosen_cache)
    caches = Enum.reduce(parent_ids, caches, &decrement_parents(&2, &1))
    {res, caches}
  end

  defp eval_apply(:while, %{data: %Expr{args: args, id: id}}, state, caches) do
    [initial, _arg, condition, block] = args
    {initial, caches} = composite_eval(initial, state, caches)
    while_cache = Map.fetch!(state.meta, id)
    {while(initial, condition, block, state, [while_cache]), caches}
  end

  defp eval_apply(:token, %{data: %Expr{args: [token], id: id}}, state, caches) do
    hooks = Map.fetch!(state.meta, id)

    caches =
      token.hooks
      |> Enum.zip(hooks)
      |> List.foldr(caches, fn
        {%{expr: expr}, true}, caches ->
          {_expr, caches} = composite_eval(expr, state, caches)
          caches

        {%{}, false}, caches ->
          caches

        {%{expr: expr}, hook_fun}, caches ->
          {res, caches} = composite_eval(expr, state, caches)
          hook_fun.(res)
          caches
      end)

    {{}, caches}
  end

  defp eval_apply(:optional, %{data: %Expr{args: args, id: id}}, state, caches) do
    [expr, default_impl_expr] = args

    {args, caches} = Tree.apply_args(expr, caches, &eval(&1, state, &2))
    backend = Nx.Shared.list_impl!(args)

    if function_exported?(backend, expr.data.op, length(args) + 1) do
      {apply(backend, expr.data.op, [expr | args]), caches}
    else
      params = Enum.map(args, &fn -> &1 end)
      optional_cache = Map.fetch!(state.meta, id)
      {res, _} = eval(default_impl_expr, %{state | params: params}, [optional_cache])
      {res, caches}
    end
  end

  defp eval_apply(op, ans, state, caches) do
    {args, caches} = Tree.apply_args(ans, caches, &eval(&1, state, &2))

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

    {apply(mod, op, args), caches}
  end

  ## Control flow helpers

  defp while(acc, condition, block, state, caches) do
    state = %{state | params: composite_to_params(acc)}
    {pred, temp} = eval(condition, state, caches)

    if Nx.to_number(pred) != 0 do
      {acc, _} = composite_eval(block, state, temp)
      while(acc, condition, block, state, caches)
    else
      acc
    end
  end

  defp cond_clause([{{pred, body}, cache} | clauses], last, last_cache, state, caches) do
    {pred, pred_caches} = eval(pred, state, [cache | caches])

    if Nx.to_number(pred) != 0,
      do: {body, pred_caches},
      else: cond_clause(clauses, last, last_cache, state, caches)
  end

  defp cond_clause([], last, last_cache, _state, caches) do
    {last, [last_cache | caches]}
  end

  ## Composite

  defp composite_eval(composite, state, caches) do
    Composite.traverse(composite, caches, &eval(&1, state, &2))
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
