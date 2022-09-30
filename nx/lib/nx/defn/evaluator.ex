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
    {expr, state, cache} = precompile(fun, vars, hooks, gc?)

    [
      Nx.Defn.Stream.start_link(input, acc, fn input_params, acc ->
        acc_params = [acc] |> Nx.Defn.Composite.flatten_list() |> Enum.map(&fn -> &1 end)
        params = input_params ++ acc_params ++ rest_params

        expr
        |> composite_eval(%{state | params: params}, cache)
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
    {expr, state, cache} = precompile(fun, vars, hooks, gc?)

    fn [params] ->
      [expr |> composite_eval(%{state | params: params}, cache) |> elem(0)]
    end
  end

  defp precompile(fun, vars, hooks, gc?) do
    expr = fun.(vars)
    state = %{params: nil, hooks: hooks, gc: gc?, scope: nil}
    composite_compute_cache(expr, state, %{})
    cache = %{}
    {expr, state, cache}
  end

  defp composite_compute_cache(expr, state, cache) do
    Composite.reduce(expr, cache, &compute_cache(&1, state, &2))
  end

  defp compute_cache(%Nx.Tensor{data: %Expr{id: id, op: op}} = tensor, state, cache) do
    # Every time we see a reference, we bump its counter.
    #
    # Some expressions, such as while and fun, require their
    # own cache and those are precomputed too. However, given
    # those constructs do not act like closures, recomputing
    # them is straight-forward.
    #
    # The only complex construct are conds. Each reference
    # seen in a cond is bumped by one, regardless of how many
    # times it shows up in the cond, and then we keep a counter
    # specific for said cond. Once the cond counter reaches zero,
    # we decrease the actual reference by one. Furthermore,
    # different branches of a cond depend on different variables,
    # we need to keep track of all branches and immediatelly
    # decrease values that are seen in other branches but not the
    # one currently chosen.
    case cache do
      %{^id => counter} ->
        case state.scope do
          nil ->
            %{cache | id => counter + 1}

          scope ->
            cache = bump_counter(cache, [id | scope])

            case cache do
              # We have seen this element in this branch
              %{^scope => %{^id => _}} -> cache
              # We have not seen this element yet
              %{^scope => set} -> %{cache | scope => Map.put(set, id, [])}
            end
        end

      %{} ->
        cache = compute_cache(op, tensor, state, cache)

        case state.scope do
          nil ->
            Map.put(cache, id, 1)

          scope ->
            cache
            # We set it to zero because it will be incremented later by cond
            |> Map.put(id, 0)
            |> Map.put([id | scope], 1)
            |> Map.update!(scope, &Map.put(&1, id, []))
        end
    end
  end

  defp bump_counter(cache, key) do
    case cache do
      %{^key => counter} -> %{cache | key => counter + 1}
      %{} -> Map.put(cache, key, 1)
    end
  end

  defp compute_cache(:fun, %{data: %Expr{id: id, args: args}}, state, cache) do
    [_args, expr, _mfa] = args
    fun_cache = composite_compute_cache(expr, state, %{})
    Map.put(cache, [:fun | id], fun_cache)
  end

  defp compute_cache(:while, %{data: %Expr{args: args, id: id}}, state, cache) do
    [initial, _arg, pred, block] = args
    cache = composite_compute_cache(initial, state, cache)

    while_cache = %{}
    while_cache = compute_cache(pred, state, while_cache)
    while_cache = composite_compute_cache(block, state, while_cache)

    Map.put(cache, [:while | id], while_cache)
  end

  # defp compute_cache(:optional, %{data: %Expr{args: [expr, default_impl_expr]}}, state, cache) do
  #   # The arguments are shared between expr and default_impl_expr nodes,
  #   # so we don't do extra work regardless of the branch we choose.
  #   {args, cache} = Tree.apply_args(expr, cache, &compute_cache(&1, state, &2))
  #   backend = Nx.Shared.list_impl!(args.)

  #   if function_exported?(backend, expr.data.op, length(args) + 1) do
  #     {apply(backend, expr.data.op, [expr | args]), cache}
  #   else
  #     eval(default_impl_expr, state, cache)
  #   end

  # :cond :token

  defp compute_cache(_op, tensor, state, cache) do
    {_, cache} = Tree.apply_args(tensor, cache, &{&1, compute_cache(&1, state, &2)})
    cache
  end

  ## Evaluation

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

  defp eval_apply(:parameter, %{data: %Expr{args: [i]}}, state, cache) do
    case Enum.fetch!(state.params, i).() do
      %Nx.Tensor{data: %Nx.Defn.Expr{}} = tensor ->
        raise ArgumentError,
              "cannot pass a tensor expression as argument to defn, got: #{inspect(tensor)}"

      %Nx.Tensor{} = tensor ->
        {tensor, cache}
    end
  end

  defp eval_apply(:fun, %{data: %Expr{args: [args, expr, _mfa]}}, state, cache) do
    fun =
      case length(args) do
        1 ->
          fn arg1 ->
            params = [fn -> Nx.to_tensor(arg1) end]
            {result, _cache} = composite_eval(expr, %{state | params: params}, %{})
            result
          end

        2 ->
          fn arg1, arg2 ->
            params = [fn -> Nx.to_tensor(arg1) end, fn -> Nx.to_tensor(arg2) end]
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
      params = Enum.map(args, &fn -> &1 end)
      eval(default_impl_expr, %{state | params: params}, cache)
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

  defp cond_clause([{pred, clause} | clauses], last, state, cache) do
    {pred, cache} = eval(pred, state, cache)

    if Nx.to_number(pred) != 0,
      do: {clause, cache},
      else: cond_clause(clauses, last, state, cache)
  end

  defp cond_clause([], last, _state, cache) do
    {last, cache}
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
