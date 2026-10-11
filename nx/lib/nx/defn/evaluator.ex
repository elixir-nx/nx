defmodule Nx.Defn.Evaluator do
  @moduledoc """
  The default implementation of a `Nx.Defn.Compiler`
  that evaluates the expression tree against the
  tensor backend.

  ## Buffer donation

  The evaluator counts how many times each node is used. When a node
  is passed to an operation for the last time, the evaluator marks that
  argument with `Nx.donatable/1`, so backends that run operations eagerly
  may reuse its buffer. A user argument marked as donatable is only
  handed over at its last use. Nodes that may still be reachable through
  another node's result (such as values returned by a `cond` branch)
  and results of the function are never marked.

  ## Options

  The following options are specific to this compiler:

    * `:hooks` - a map of callbacks to override named `io_call` side effects.
      This allows overriding named `io_call`s at JIT time without recompiling the
      graph.

    * `:garbage_collect` - when true, garbage collects
      after evaluating each node

    * `:max_concurrency` - the number of partitions to
      start when running a `Nx.Serving` with this compiler

  """

  @behaviour Nx.Defn.Compiler
  alias Nx.Defn.{Composite, Expr, Tree}

  # Ops whose result is the value of some inner expression, so they keep the
  # `donatable?` mark that evaluating that expression produced.
  @keep_mark_ops [:parameter, :cond, :while, :elem]

  # Ops whose result may alias one of their arguments or comes from user code,
  # so they never hand their result over for donation.
  @never_donate_ops [:block, :optional, :runtime_call, :io_call, :fun]

  @creation_ops [:eye, :iota, :from_binary]
  @list_ops [:concatenate, :stack]
  @indices_ops [:slice, :put_slice]

  @impl true
  def __partitions_options__(opts) do
    List.duplicate(opts, Keyword.get(opts, :max_concurrency, 1))
  end

  @impl true
  def __to_backend__(_opts) do
    Nx.default_backend()
  end

  @impl true
  def __jit__(key, vars, fun, args_list, opts) do
    __compile__(key, vars, fun, opts).(args_list)
  end

  @impl true
  def __compile__(_key, vars, fun, opts) do
    hooks = Keyword.get(opts, :hooks, %{})

    gc? = Keyword.get(opts, :garbage_collect, false)
    {expr, output, cache, aliased_ids} = precompile(fun, vars, hooks)

    fn [params] ->
      state = %{
        params: params,
        gc: gc?,
        hooks: hooks,
        aliased_ids: aliased_ids,
        held: %{}
      }

      [expr |> composite_eval(state, [cache]) |> apply_output(output)]
    end
  end

  @impl true
  def __shard_jit__(_key, _mesh, _vars, _fun, _args_list, _opts) do
    raise "sharding is not supported by Nx.Defn.Evaluator"
  end

  defp apply_output({result, _cache}, output) do
    {result, []} =
      Composite.traverse(result, output, fn result, [out | acc] ->
        {%{out | data: result.data, donatable?: false}, acc}
      end)

    result
  end

  defp precompile(fun, vars, hooks) do
    {expr, output} =
      vars
      |> fun.()
      |> Composite.traverse([], &{Nx.devectorize(&1), [Nx.to_template(&1) | &2]})

    state = %{hooks: hooks, parent_ids: nil, current_ids: nil}
    aliased_ids = aliased_ids(expr)
    {expr, cache} = init_compute_cache(expr, state)
    {expr, Enum.reverse(output), cache, aliased_ids}
  end

  # Collects the ids whose buffer may still be reachable through another
  # node's result. Branches of a cond, the initial value of a while, and the
  # inputs of io_call, runtime_call and block can return or keep their input
  # under a different id, so the use count of the original id undercounts the
  # references to its buffer.
  defp aliased_ids(expr), do: expr |> collect_aliased({%{}, %{}}) |> elem(1)

  defp collect_aliased(composite, acc) do
    Composite.reduce(composite, acc, &collect_aliased_node/2)
  end

  defp collect_aliased_node(%Nx.Tensor{data: %Expr{op: op}}, acc)
       when op in [:constant, :tensor],
       do: acc

  defp collect_aliased_node(%Nx.Tensor{data: %Expr{op: :metadata, args: [expr, _]}}, acc) do
    collect_aliased(expr, acc)
  end

  defp collect_aliased_node(%Nx.Tensor{data: %Expr{id: id, op: op, args: args}} = tensor, acc) do
    {seen, aliased} = acc

    if is_map_key(seen, id) do
      acc
    else
      acc = {Map.put(seen, id, true), add_aliased(op, args, aliased)}

      {_, acc} =
        Tree.apply_args(tensor, :all, acc, fn arg, acc ->
          {arg, collect_aliased_node(arg, acc)}
        end)

      acc
    end
  end

  defp collect_aliased_node(_other, acc), do: acc

  defp add_aliased(:cond, [clauses, last], aliased) do
    Enum.reduce([last | Enum.map(clauses, &elem(&1, 1))], aliased, &add_leaf_ids/2)
  end

  defp add_aliased(:while, [initial | _], aliased), do: add_leaf_ids(initial, aliased)

  defp add_aliased(op, [tensor_expr | _], aliased) when op in [:io_call, :runtime_call],
    do: add_leaf_ids(tensor_expr, aliased)

  defp add_aliased(:block, [_struct, in_args | _], aliased),
    do: Enum.reduce(in_args, aliased, &add_leaf_ids/2)

  defp add_aliased(_op, _args, aliased), do: aliased

  defp add_leaf_ids(composite, aliased) do
    Composite.reduce(composite, aliased, fn
      %Nx.Tensor{data: %Expr{op: :metadata, args: [expr, _]}}, aliased ->
        add_leaf_ids(expr, aliased)

      %Nx.Tensor{data: %Expr{id: id}}, aliased ->
        Map.put(aliased, id, true)

      _other, aliased ->
        aliased
    end)
  end

  defp init_compute_cache(expr, state) do
    state = %{state | parent_ids: %{}, current_ids: Tree.scope_ids(expr, %{})}
    composite_compute_cache(expr, state, %{})
  end

  defp composite_compute_cache(expr, state, cache) do
    Composite.traverse(expr, cache, &compute_cache(&1, state, &2))
  end

  defp compute_cache(%Nx.Tensor{data: %Expr{op: op}} = tensor, _state, cache)
       when op in [:constant, :tensor] do
    {tensor, cache}
  end

  defp compute_cache(%Nx.Tensor{data: %Expr{op: :metadata, args: [expr, _meta]}}, state, cache) do
    composite_compute_cache(expr, state, cache)
  end

  defp compute_cache(%Nx.Tensor{data: %Expr{id: id, op: op}} = tensor, state, cache) do
    cache =
      case state.parent_ids do
        # If the id exists in the parent, the parent will compute it.
        %{^id => _} ->
          Map.put_new(cache, id, tensor)

        %{} ->
          case cache do
            %{^id => {:args, counter, args_or_placeholder}} ->
              %{cache | id => {:args, counter + 1, args_or_placeholder}}

            %{} ->
              cache = Map.put(cache, id, {:args, 1, nil})
              {args, cache} = compute_cache(op, tensor, state, cache)
              Map.update!(cache, id, fn {:args, counter, _} -> {:args, counter, args} end)
          end
      end

    {put_in(tensor.data.args, nil), cache}
  end

  defp compute_cache(:fun, %{data: %Expr{args: args}}, state, cache) do
    [args, expr, _mfa] = args
    {expr, expr_cache} = init_compute_cache(expr, state)
    {[length(args), expr, expr_cache], cache}
  end

  defp compute_cache(:while, %{data: %Expr{args: args}}, state, cache) do
    [initial, _arg, pred, block] = args
    {initial, cache} = composite_compute_cache(initial, state, cache)
    {{pred, block}, while_cache} = init_compute_cache({pred, block}, state)
    {[initial, pred, block, while_cache], cache}
  end

  defp compute_cache(:block, %{data: %Expr{args: args}}, state, cache) do
    [struct, in_args, expr, callback] = args

    {call_prefix, call_suffix} = Enum.split_while(in_args, &(not is_list(&1)))
    {call_prefix, cache} = Enum.map_reduce(call_prefix, cache, &compute_cache(&1, state, &2))
    in_args = call_prefix ++ call_suffix

    {[struct, in_args, expr, callback], cache}
  end

  defp compute_cache(:cond, %{data: %Expr{args: [clauses, last]}}, state, cache) do
    %{parent_ids: parent_ids, current_ids: current_ids} = state

    clause_caches =
      Enum.map([last | clauses], fn clause ->
        state = %{
          state
          | parent_ids: current_ids,
            current_ids: Tree.scope_ids(clause, current_ids)
        }

        composite_compute_cache(clause, state, %{})
      end)

    # Now, for each cache, split the IDs from parents from the actual cond IDs
    {[last_cache | clauses_cache], {all_ids, cache}} =
      Enum.map_reduce(clause_caches, {%{}, cache}, fn {clause, clause_cache}, seen_ids_cache ->
        {clause_cache, seen_ids_cache} =
          Enum.flat_map_reduce(clause_cache, seen_ids_cache, fn
            {id, %_{} = tensor}, {seen_ids, cache} ->
              case seen_ids do
                # We have already processed this id for the whole cond
                %{^id => _} ->
                  {[], {seen_ids, cache}}

                # The ID belongs to our own parents
                %{} when is_map_key(parent_ids, id) ->
                  {[], {seen_ids, Map.put_new(cache, id, tensor)}}

                # The ID belongs to us
                %{} ->
                  {_, cache} = composite_compute_cache(tensor, state, cache)
                  {[], {Map.put(seen_ids, id, true), cache}}
              end

            {id, counter}, seen_ids_cache ->
              {[{id, counter}], seen_ids_cache}
          end)

        {{clause, Map.new(clause_cache)}, seen_ids_cache}
      end)

    {[clauses_cache, last_cache, Map.keys(all_ids)], cache}
  end

  defp compute_cache(:io_call, %{data: %Expr{args: args}}, state, cache) do
    [tensor_expr, callback_spec, template, ref] = args
    {tensor_expr, cache} = composite_compute_cache(tensor_expr, state, cache)
    {[tensor_expr, callback_spec, template, ref], cache}
  end

  defp compute_cache(_op, tensor, state, cache) do
    Tree.apply_args(tensor, cache, &compute_cache(&1, state, &2))
  end

  ## Evaluation

  # Tensors embedded in the expression outlive this evaluation.
  defp eval(%Nx.Tensor{data: %Expr{op: :tensor, args: [t]}}, _state, caches) do
    {%{t | donatable?: false}, caches}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :constant, args: [constant]}} = ans, _state, caches) do
    {backend, backend_options} = Nx.default_backend()
    {backend.constant(ans, constant, backend_options), caches}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: op, id: id}} = ans, state, [cache | caches]) do
    case cache do
      %{^id => {:args, count, args}} ->
        {res, [cache | caches]} = eval_apply(op, args, ans, state, [cache | caches])
        state.gc && :erlang.garbage_collect(self())
        res = store_mark(res, op)
        {mark(res, op, id, count, state), [decrement_cache(cache, id, count, res) | caches]}

      %{^id => {:result, count, res}} ->
        {mark(res, op, id, count, state), [decrement_cache(cache, id, count, res) | caches]}

      %{} ->
        # If we don't find the tensor in the current scope,
        # it may be in a parent scope, so look up. If we find
        # it in the parent, we don't decrement it, as that will
        # be done at the end of processing the current scope.
        eval_parent(caches, id, op, ans, state, [cache])
    end
  end

  # The mark stored in the cache is the one the result came with. Backends may
  # copy the `donatable?` flag of their input into the result, so only ops that
  # return an inner value keep it.
  defp store_mark(%Nx.Tensor{} = res, op) when op in @keep_mark_ops, do: res
  defp store_mark(%Nx.Tensor{} = res, _op), do: %{res | donatable?: false}
  defp store_mark(res, _op), do: res

  defp not_donatable(%Nx.Tensor{} = res), do: %{res | donatable?: false}
  defp not_donatable(res), do: res

  # The value handed to a consumer is donatable only at its last use. A node
  # held by an operation that is still collecting its arguments, or one that
  # may be reachable through another result, is never donatable.
  defp mark(%Nx.Tensor{} = res, op, id, count, state) do
    donatable? =
      cond do
        count > 1 -> false
        is_map_key(state.held, id) -> false
        is_map_key(state.aliased_ids, id) -> false
        op in @keep_mark_ops -> res.donatable?
        op in @never_donate_ops -> false
        true -> true
      end

    %{res | donatable?: donatable?}
  end

  defp mark(res, _op, _id, _count, _state), do: res

  defp decrement_cache(cache, id, 1, _res), do: Map.delete(cache, id)
  defp decrement_cache(cache, id, counter, res), do: %{cache | id => {:result, counter - 1, res}}

  defp eval_parent([cache | caches], id, op, ans, state, acc) do
    case cache do
      %{^id => {:result, _count, res}} ->
        {not_donatable(res), Enum.reverse(acc, [cache | caches])}

      %{^id => {:args, count, args}} ->
        {res, [cache | caches]} = eval_apply(op, args, ans, state, [cache | caches])
        state.gc && :erlang.garbage_collect(self())
        res = store_mark(res, op)

        {not_donatable(res),
         Enum.reverse(acc, [Map.put(cache, id, {:result, count, res}) | caches])}

      %{} ->
        eval_parent(caches, id, op, ans, state, [cache | acc])
    end
  end

  defp eval_parent([], id, op, _ans, _state, _acc) do
    raise "trying to read evaluator cache that has expired for OP=#{op} ID=#{inspect(id)}\n\n" <>
            "Please report this bug with the relevant code that triggers it: https://github.com/elixir-nx/nx"
  end

  defp decrement_parents([cache | caches], id) do
    case cache do
      %{^id => {:result, count, value}} -> [decrement_cache(cache, id, count, value) | caches]
      %{^id => {:args, count, args}} -> [%{cache | id => {:args, count - 1, args}} | caches]
      %{} -> [cache | decrement_parents(caches, id)]
    end
  end

  defp eval_apply(:parameter, [i], _ans, state, caches) do
    case Enum.fetch!(state.params, i).() do
      %Nx.Tensor{data: %Nx.Defn.Expr{}} = tensor ->
        raise ArgumentError,
              "cannot pass a tensor expression as argument to defn, got: #{inspect(tensor)}"

      %Nx.Tensor{} = tensor ->
        {Nx.devectorize(tensor), caches}
    end
  end

  defp eval_apply(:elem, [tuple, i], _ans, state, caches) do
    {value, caches} = composite_eval(tuple, state, caches)
    {tuple_element(elem(value, i), tuple), caches}
  end

  defp eval_apply(:io_call, [tensor_expr, callback_spec, out_template, _ref], _ans, state, caches) do
    {tensor_value, caches} = composite_eval(tensor_expr, state, caches)
    caches = run_io_call_side_effect(callback_spec, tensor_value, state, caches)

    result =
      case out_template do
        %Nx.Tensor{} ->
          tensor_value

        _ ->
          tensor_value
          |> List.wrap()
          |> Composite.flatten_list()
          |> List.to_tuple()
      end

    {result, caches}
  end

  defp eval_apply(:fun, [length, expr, expr_cache], _ans, state, caches) do
    fun =
      case length do
        1 ->
          fn arg1 ->
            params = [fn -> not_donatable(Nx.to_tensor(arg1)) end]
            {result, _} = composite_eval(expr, %{state | params: params}, [expr_cache])
            result
          end

        2 ->
          fn arg1, arg2 ->
            params = [
              fn -> not_donatable(Nx.to_tensor(arg1)) end,
              fn -> not_donatable(Nx.to_tensor(arg2)) end
            ]
            {result, _} = composite_eval(expr, %{state | params: params}, [expr_cache])
            result
          end
      end

    {fun, caches}
  end

  defp eval_apply(:cond, [clauses_cache, last_cache, parent_ids], _ans, state, caches) do
    {chosen, chosen_cache} = cond_clause(clauses_cache, last_cache, state, caches)
    {res, [_ | caches]} = composite_eval(chosen, state, chosen_cache)
    caches = Enum.reduce(parent_ids, caches, &decrement_parents(&2, &1))
    {res, caches}
  end

  defp eval_apply(:while, [initial, pred, block, while_cache], _ans, state, caches) do
    {initial, caches} = composite_eval(initial, state, caches)
    {while(initial, pred, block, state, [while_cache]), caches}
  end

  defp eval_apply(:block, [struct, in_args, expr, callback], ans, state, caches) do
    {in_args, caches} = Enum.map_reduce(in_args, caches, &eval(&1, state, &2))
    {param_prefix, _} = Enum.split_while(in_args, &(not is_list(&1)))
    backend = Nx.Shared.list_impl!(param_prefix)

    out =
      case ans do
        %{type: {:tuple, _}} -> expr
        _ -> ans
      end

    {backend.block(struct, out, in_args, callback), caches}
  end

  defp eval_apply(:runtime_call, [expr, fun, out_template, opts], _ans, state, caches) do
    {tensor_value, caches} = composite_eval(expr, state, caches)
    result = fun.(tensor_value, opts)

    if not Nx.compatible?(out_template, result) do
      raise "expected the runtime_call function to match the given output template"
    end

    case out_template do
      %Nx.Tensor{} -> {result, caches}
      _ -> {[result] |> Composite.flatten_list() |> List.to_tuple(), caches}
    end
  end

  defp eval_apply(op, args, ans, state, caches) do
    ans = put_in(ans.data.args, args)
    # Arguments evaluated so far are held until the op runs, so no argument
    # evaluated later may consume them.
    {args, {caches, _held}} =
      Tree.apply_args(ans, {caches, state.held}, fn arg, {caches, held} ->
        {value, caches} = eval(arg, %{state | held: held}, caches)
        {value, {caches, hold(held, arg)}}
      end)

    {mod, args} =
      cond do
        op in @creation_ops ->
          {backend, backend_options} = Nx.default_backend()
          {backend, [ans | args] ++ [backend_options]}

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

  # Elements of a tuple returned by an operation are not donated, as several
  # elem nodes may read the same element.
  defp tuple_element(%Nx.Tensor{} = element, %Nx.Tensor{data: %Expr{op: op}})
       when op in [:cond, :while],
       do: element

  defp tuple_element(element, _tuple), do: not_donatable(element)

  defp hold(held, %Nx.Tensor{data: %Expr{id: id}}), do: Map.put(held, id, true)
  defp hold(held, _other), do: held

  defp run_io_call_side_effect(callback_spec, tensor_value, state, caches) do
    case resolve_io_call(callback_spec, state.hooks) do
      nil -> :ok
      fun -> fun.(tensor_value)
    end

    caches
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

  defp cond_clause([{{pred, body}, cache} | clauses], last_cache, state, caches) do
    {pred, pred_caches} = eval(pred, state, [cache | caches])

    if Nx.to_number(pred) != 0,
      do: {body, pred_caches},
      else: cond_clause(clauses, last_cache, state, caches)
  end

  defp cond_clause([], {last, last_cache}, _state, caches) do
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

  # The loop state is read by the condition and the body, so it is never donated.
  defp composite_to_params(other, acc) do
    [fn -> not_donatable(other) end | acc]
  end

  defp resolve_io_call({:fn, fun}, _hooks), do: fun
  defp resolve_io_call({:named, name, callback}, hooks), do: hooks[name] || callback
end
