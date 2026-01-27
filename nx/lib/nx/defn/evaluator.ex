defmodule Nx.Defn.Evaluator do
  @moduledoc """
  The default implementation of a `Nx.Defn.Compiler`
  that evaluates the expression tree against the
  tensor backend.

  ## Options

  The following options are specific to this compiler:

    * `:garbage_collect` - when true, garbage collects
      after evaluating each node

    * `:max_concurrency` - the number of partitions to
      start when running a `Nx.Serving` with this compiler

    * `:debug_options` - a keyword list of options for
      debugging the evaluation of the expression tree.
      If not given, no debugging information will be printed or saved.
      The following options are supported:

      * `:save_path` - the base path for output files. If not given,
        the output will be printed to the standard output.
      * `:inspect_limit` - limit that will be passed to `inspect/2`
        for the result and arguments of each node.

  """

  @behaviour Nx.Defn.Compiler
  alias Nx.Defn.{Composite, Expr, Tree}

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
    debug_options = Keyword.get(opts, :debug_options)

    if debug_options do
      if save_path = Keyword.get(debug_options, :save_path) do
        File.mkdir_p!(save_path)
      end
    end

    {expr, output, cache, templates} = precompile(fun, vars, hooks)

    fn [params] ->
      printed_nodes =
        if debug_options do
          :ets.new(:printed_nodes, [:set, :protected])
        end

      state = %{
        params: params,
        gc: gc?,
        hooks: hooks,
        debug_options: debug_options,
        printed_nodes: printed_nodes,
        templates: templates
      }

      result =
        [
          expr
          |> composite_eval(state, [cache])
          |> apply_output(output)
        ]

      if printed_nodes do
        :ets.delete(printed_nodes)
      end

      result
    end
  end

  defp apply_output({result, _cache}, output) do
    {result, []} =
      Composite.traverse(result, output, fn
        result, [out | acc] ->
          {%{out | data: result.data}, acc}
      end)

    result
  end

  defp precompile(fun, vars, hooks) do
    expr = fun.(vars)
    state = %{hooks: hooks, parent_ids: nil, current_ids: nil}
    {cache, templates} = init_compute_cache(expr, state)

    # Devectorize the expression for evaluation, but keep original tensors for output templates
    {devectorized_expr, output_templates} =
      Nx.Defn.Composite.traverse(expr, [], fn tensor, acc ->
        devectorized = Nx.devectorize(tensor)
        {devectorized, [tensor | acc]}
      end)

    {devectorized_expr, Enum.reverse(output_templates), cache, templates}
  end

  defp init_compute_cache(expr, state) do
    state = %{state | parent_ids: %{}, current_ids: Tree.scope_ids(expr, %{})}
    composite_compute_cache(expr, state, {%{}, %{}})
  end

  defp composite_compute_cache(expr, state, {cache, templates}) do
    Composite.reduce(expr, {cache, templates}, fn tensor, {cache, templates} ->
      compute_cache(tensor, state, {cache, templates})
    end)
  end

  # Create flattened node info for cache
  # Store metadata separately in a tensor_templates map that we can look up by ID
  defp make_node_info(:parameter, %{data: %Expr{args: [i]}} = _tensor),
    do: {:param, i}

  defp make_node_info(:constant, %{data: %Expr{args: [constant]}} = _tensor),
    do: {:constant, constant}

  defp make_node_info(:tensor, %{data: %Expr{args: [t]}} = _tensor),
    do: {:tensor, t}

  defp make_node_info(:metadata, %{data: %Expr{args: [expr, _meta]}} = _tensor) do
    # Store the wrapped expression ID so we can defer evaluation to it
    {:metadata, extract_id(expr)}
  end

  defp make_node_info(:fun, %{data: %Expr{id: id}} = _tensor),
    do: {:fun, id}

  defp make_node_info(:while, %{data: %Expr{id: id}} = _tensor),
    do: {:while, id}

  defp make_node_info(:cond, %{data: %Expr{id: id}} = _tensor),
    do: {:cond, id}

  defp make_node_info(:optional, %{data: %Expr{id: id}} = _tensor),
    do: {:optional, id}

  defp make_node_info(:token, %{data: %Expr{id: id}} = _tensor),
    do: {:token, id}

  defp make_node_info(:elem, %{data: %Expr{args: [tuple, i]}} = _tensor),
    do: {:elem, extract_id(tuple), i}

  defp make_node_info(:attach_token, %{data: %Expr{args: [token, expr]}} = _tensor),
    do: {:attach_token, extract_id(token), extract_id(expr)}

  defp make_node_info(:runtime_call, %{data: %Expr{args: [tensor_expr | _]}} = _tensor),
    do: {:runtime_call, extract_id(tensor_expr)}

  defp make_node_info(op, %{data: %Expr{args: args}} = _tensor) do
    # For each arg, store either {:reference, id} for expressions or the literal value
    arg_refs = Enum.map(args, &extract_arg_ref/1)
    {:generic, op, arg_refs}
  end

  # Extract argument reference - either an ID for expressions or the literal value
  defp extract_arg_ref(%Nx.Tensor{data: %Expr{id: id}}), do: {:reference, id}
  defp extract_arg_ref(other), do: other

  defp extract_id(%Nx.Tensor{data: %Expr{id: id}}), do: id
  defp extract_id(_), do: nil

  defp compute_cache(%Nx.Tensor{data: %Expr{op: :constant}}, _state, acc) do
    acc
  end

  defp compute_cache(%Nx.Tensor{data: %Expr{op: :metadata, args: [expr, _meta]}}, state, acc) do
    composite_compute_cache(expr, state, acc)
  end

  defp compute_cache(%Nx.Tensor{data: %Expr{id: id, op: op}} = tensor, state, {cache, templates}) do
    case state.parent_ids do
      # If the id exists in the parent, the parent will compute it.
      %{^id => _} ->
        {Map.put_new(cache, id, tensor), templates}

      %{} ->
        case cache do
          %{^id => {node_info, counter}} ->
            {%{cache | id => {node_info, counter + 1}}, templates}
          %{} ->
            node_info = make_node_info(op, tensor)
            templates = Map.put(templates, id, tensor)
            compute_cache(op, tensor, state, {Map.put(cache, id, {node_info, 1}), templates})
        end
    end
  end

  defp compute_cache(:fun, %{data: %Expr{id: id, args: args}}, state, {cache, templates}) do
    [_args, expr, _mfa] = args
    {fun_cache, fun_templates} = init_compute_cache(expr, state)
    # Merge fun templates into global templates
    templates = Map.merge(templates, fun_templates)
    {Map.put(cache, [:fun | id], fun_cache), templates}
  end

  defp compute_cache(:while, %{data: %Expr{args: args, id: id}}, state, {cache, templates}) do
    [initial, _arg, pred, block] = args
    {cache, templates} = composite_compute_cache(initial, state, {cache, templates})
    {while_cache, while_templates} = init_compute_cache({pred, block}, state)
    # Merge while templates into global templates
    templates = Map.merge(templates, while_templates)
    {Map.put(cache, [:while | id], while_cache), templates}
  end

  defp compute_cache(:optional, %{data: %Expr{args: args, id: id}}, state, {cache, templates}) do
    [call, expr, _callback] = args
    %{data: %{args: call_args_in, op: call_name}} = call

    {call_args, opts} = Enum.split_while(call_args_in, &(not is_list(&1)))

    {cache, templates} = Enum.reduce(call_args, {cache, templates}, fn arg, acc ->
      compute_cache(arg, state, acc)
    end)
    key = computation_key(call_name, call_args ++ opts)

    {optional_expr_cache, cache, templates} =
      case cache do
        %{^key => optional_expr_cache} ->
          {optional_expr_cache, cache, templates}

        %{} ->
          {opt_cache, opt_templates} = init_compute_cache(expr, state)
          # Merge optional templates into global templates
          templates = Map.merge(templates, opt_templates)
          optional_expr_cache = {expr, opt_cache}
          {optional_expr_cache, Map.put(cache, key, optional_expr_cache), templates}
      end

    {Map.put(cache, [:optional | id], optional_expr_cache), templates}
  end

  defp compute_cache(
         :runtime_call,
         %{data: %Expr{args: [tensor_expr, _opts, _fun, _out]}},
         state,
         acc
       ) do
    composite_compute_cache(tensor_expr, state, acc)
  end

  defp compute_cache(:cond, %{data: %Expr{args: [clauses, last], id: id}}, state, {cache, templates}) do
    %{parent_ids: parent_ids, current_ids: current_ids} = state

    clause_caches =
      Enum.map([last | clauses], fn clause ->
        state = %{
          state
          | parent_ids: current_ids,
            current_ids: Tree.scope_ids(clause, current_ids)
        }

        composite_compute_cache(clause, state, {%{}, %{}})
      end)

    # Now, for each cache, split the IDs from parents from the actual cond IDs
    {[last_cache | clauses_cache], {all_ids, cache, templates}} =
      Enum.map_reduce(clause_caches, {%{}, cache, templates}, fn {clause_cache, clause_templates}, {seen_ids, cache, templates} ->
        {clause_cache, {seen_ids, cache, templates}} =
          Enum.flat_map_reduce(clause_cache, {seen_ids, cache, templates}, fn
            {id, %_{} = tensor}, {seen_ids, cache, templates} ->
              case seen_ids do
                # We have already processed this id for the whole cond
                %{^id => _} ->
                  {[], {seen_ids, cache, templates}}

                # The ID belongs to our own parents
                %{} when is_map_key(parent_ids, id) ->
                  {[], {seen_ids, Map.put_new(cache, id, tensor), templates}}

                # The ID belongs to us
                %{} ->
                  {cache, templates} = composite_compute_cache(tensor, state, {cache, templates})
                  {[], {Map.put(seen_ids, id, true), cache, templates}}
              end

            {id, {node_info, counter}}, {seen_ids, cache, templates} ->
              {[{id, {node_info, counter}}], {seen_ids, cache, templates}}

            # Handle special control flow keys like [:cond | id], [:while | id], etc.
            {key, value}, {seen_ids, cache, templates} when is_list(key) ->
              {[{key, value}], {seen_ids, cache, templates}}
          end)

        # Merge clause templates into main templates
        templates = Map.merge(templates, clause_templates)
        {Map.new(clause_cache), {seen_ids, cache, templates}}
      end)

    {Map.put(cache, [:cond | id], {clauses_cache, last_cache, Map.keys(all_ids)}), templates}
  end

  defp compute_cache(:token, %{data: %Expr{args: [token], id: id}}, state, {cache, templates}) do
    hooks = state.hooks

    {hooks, cache, templates} =
      Enum.reduce(token.hooks, {[], cache, templates}, fn
        %{callback: callback, expr: expr, name: name}, {hooks_acc, cache, templates} ->
          hook_fun = hooks[name] || callback

          cond do
            hook_fun -> 
              {cache, templates} = composite_compute_cache(expr, state, {cache, templates})
              {[hook_fun | hooks_acc], cache, templates}
            Tree.has_hooks?(expr, hooks) -> 
              {cache, templates} = composite_compute_cache(expr, state, {cache, templates})
              {[true | hooks_acc], cache, templates}
            true -> 
              {[false | hooks_acc], cache, templates}
          end
      end)

    {Map.put(cache, [:token | id], Enum.reverse(hooks)), templates}
  end

  defp compute_cache(_op, tensor, state, acc) do
    {_, acc} = Tree.apply_args(tensor, acc, fn arg, acc ->
      {arg, compute_cache(arg, state, acc)}
    end)
    acc
  end

  defp computation_key(op, args) do
    keys =
      Enum.map(args, fn
        %Nx.Tensor{shape: shape, names: names, type: type} -> {type, shape, names}
        opts -> opts
      end)

    {op, keys}
  end

  ## Evaluation

  # New ID-based evaluation - dispatch to node_info or legacy path
  defp eval(id, state, [cache | caches]) when is_reference(id) do
    case cache do
      %{^id => {node_info, count}} when is_integer(count) ->
        {res, [cache | caches]} = eval_node_info(node_info, id, state, [cache | caches])
        state.gc && :erlang.garbage_collect(self())
        # debug_node needs the tensor, skip for now or reconstruct
        {res, [decrement_cache(cache, id, count, res) | caches]}

      %{^id => {_node_info, count, res}} ->
        # debug_node needs the tensor, skip for now
        {res, [decrement_cache(cache, id, count, res) | caches]}

      %{} ->
        # If we don't find the id in the current scope, look up parent scopes
        eval_parent_by_id(caches, id, state, [cache])
    end
  end

  # Legacy path for tensors (still needed for some cases)
  defp eval(%Nx.Tensor{data: %Expr{op: :tensor, args: [t]}}, _state, caches) do
    {t, caches}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :constant, args: [constant]}} = ans, _state, caches) do
    {backend, backend_options} = Nx.default_backend()
    {backend.constant(ans, constant, backend_options), caches}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :metadata, args: [expr, _meta]}}, state, caches) do
    composite_eval(expr, state, caches)
  end

  defp eval(%Nx.Tensor{data: %Expr{op: op, id: id}} = ans, state, [cache | caches]) do
    case cache do
      %{^id => {_node_info, count}} when is_integer(count) ->
        {res, [cache | caches]} = eval_apply(op, ans, state, [cache | caches])
        state.gc && :erlang.garbage_collect(self())
        debug_node(ans, res, state)
        {res, [decrement_cache(cache, id, count, res) | caches]}

      %{^id => {_node_info, count, res}} ->
        debug_node(ans, res, state)
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

  defp decrement_cache(cache, id, counter, res) do
    case cache do
      %{^id => {node_info, ^counter}} ->
        %{cache | id => {node_info, counter - 1, res}}
      %{^id => {node_info, _old_counter, _old_res}} ->
        %{cache | id => {node_info, counter - 1, res}}
    end
  end

  defp eval_parent([cache | caches], id, op, ans, state, acc) do
    case cache do
      %{^id => {_node_info, _count, res}} ->
        {res, Enum.reverse(acc, [cache | caches])}

      %{^id => {node_info, count}} when is_integer(count) ->
        {res, [cache | caches]} = eval_apply(op, ans, state, [cache | caches])
        state.gc && :erlang.garbage_collect(self())
        {res, Enum.reverse(acc, [Map.put(cache, id, {node_info, count, res}) | caches])}

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
      %{^id => {_node_info, count, value}} ->
        [decrement_cache(cache, id, count, value) | caches]
      %{^id => {node_info, count}} ->
        [%{cache | id => {node_info, count - 1}} | caches]
      %{} ->
        [cache | decrement_parents(caches, id)]
    end
  end

  # Evaluate by ID looking up parent caches
  defp eval_parent_by_id([cache | caches], id, state, acc) do
    case cache do
      %{^id => {_node_info, _count, res}} ->
        {res, Enum.reverse(acc, [cache | caches])}

      %{^id => {node_info, count}} when is_integer(count) ->
        {res, [cache | caches]} = eval_node_info(node_info, id, state, [cache | caches])
        state.gc && :erlang.garbage_collect(self())
        {res, Enum.reverse(acc, [Map.put(cache, id, {node_info, count, res}) | caches])}

      %{} ->
        eval_parent_by_id(caches, id, state, [cache | acc])
    end
  end

  defp eval_parent_by_id([], id, _state, _acc) do
    raise "trying to read evaluator cache that has expired for ID: #{inspect(id)}\n\n" <>
            "Please report this bug with the relevant code that triggers it: https://github.com/elixir-nx/nx"
  end

  # Evaluate based on node_info from flattened cache
  defp eval_node_info({:param, i}, _id, state, caches) do
    case Enum.fetch!(state.params, i).() do
      %Nx.Tensor{data: %Nx.Defn.Expr{}} = tensor ->
        raise ArgumentError,
              "cannot pass a tensor expression as argument to defn, got: #{inspect(tensor)}"

      %Nx.Tensor{} = tensor ->
        {Nx.devectorize(tensor), caches}
    end
  end

  defp eval_node_info({:constant, constant}, id, state, caches) do
    # Look up tensor template from state
    ans = Map.fetch!(state.templates, id)
    {backend, backend_options} = Nx.default_backend()
    {backend.constant(ans, constant, backend_options), caches}
  end

  defp eval_node_info({:tensor, t}, _id, _state, caches) do
    {t, caches}
  end

  defp eval_node_info({:metadata, wrapped_expr_id}, _id, state, caches) do
    # Defer to the wrapped expression
    eval(wrapped_expr_id, state, caches)
  end

  defp eval_node_info({:elem, tuple_id, i}, _id, state, caches) do
    {tuple, caches} = eval(tuple_id, state, caches)
    {elem(tuple, i), caches}
  end

  defp eval_node_info({:attach_token, token_id, expr_id}, _id, state, caches) do
    {_, caches} = eval(token_id, state, caches)
    eval(expr_id, state, caches)
  end

  defp eval_node_info({:fun, _fun_id}, id, state, caches) do
    # Look up tensor template from state
    tensor = Map.fetch!(state.templates, id)
    eval_apply(:fun, tensor, state, caches)
  end

  defp eval_node_info({:cond, _cond_id}, id, state, caches) do
    tensor = Map.fetch!(state.templates, id)
    eval_apply(:cond, tensor, state, caches)
  end

  defp eval_node_info({:while, _while_id}, id, state, caches) do
    tensor = Map.fetch!(state.templates, id)
    eval_apply(:while, tensor, state, caches)
  end

  defp eval_node_info({:optional, _optional_id}, id, state, caches) do
    tensor = Map.fetch!(state.templates, id)
    eval_apply(:optional, tensor, state, caches)
  end

  defp eval_node_info({:token, _token_id}, id, state, caches) do
    tensor = Map.fetch!(state.templates, id)
    eval_apply(:token, tensor, state, caches)
  end

  defp eval_node_info({:runtime_call, _tensor_expr_id}, id, state, caches) do
    tensor = Map.fetch!(state.templates, id)
    eval_apply(:runtime_call, tensor, state, caches)
  end

  defp eval_node_info({:generic, op, _arg_refs}, id, state, caches) do
    # Look up tensor template from state to get full operation info
    tensor = Map.fetch!(state.templates, id)
    eval_apply(op, tensor, state, caches)
  end


  defp eval_apply(:parameter, %{data: %Expr{args: [i]}}, state, caches) do
    case Enum.fetch!(state.params, i).() do
      %Nx.Tensor{data: %Nx.Defn.Expr{}} = tensor ->
        raise ArgumentError,
              "cannot pass a tensor expression as argument to defn, got: #{inspect(tensor)}"

      %Nx.Tensor{} = tensor ->
        {Nx.devectorize(tensor), caches}
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
    {fun_cache, caches} = pop_cache!(caches, [:fun | id])

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
    {{clauses_cache, last_cache, parent_ids}, caches} = pop_cache!(caches, [:cond | id])

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
    {while_cache, caches} = pop_cache!(caches, [:while | id])
    {while(initial, condition, block, state, [while_cache]), caches}
  end

  defp eval_apply(:token, %{data: %Expr{args: [token], id: id}}, state, caches) do
    {hooks, caches} = pop_cache!(caches, [:token | id])

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

  defp eval_apply(:optional, %{data: %Expr{args: [call, out, _callback], id: id}}, state, caches) do
    {args, caches} = Tree.apply_args(call, caches, &eval(&1, state, &2))
    backend = Nx.Shared.list_impl!(args)

    if function_exported?(backend, call.data.op, length(args) + 1) do
      out =
        case call do
          %{type: {:tuple, _}} -> out
          _ -> call
        end

      {apply(backend, call.data.op, [out | args]), caches}
    else
      params = Enum.map(args, &fn -> &1 end)
      {{expr, optional_cache}, caches} = pop_cache!(caches, [:optional | id])
      {res, _} = composite_eval(expr, %{state | params: params}, [optional_cache])
      {res, caches}
    end
  end

  defp eval_apply(
         :runtime_call,
         %{data: %Expr{args: [tensor_expr, static_argument, fun, out_template]}},
         state,
         caches
       ) do
    {tensor_value, caches} = composite_eval(tensor_expr, state, caches)
    result = fun.(tensor_value, static_argument)
    {reshape_runtime_call_result(result, out_template), caches}
  end

  defp eval_apply(op, %{vectorized_axes: [_ | _]} = ans, _state, _caches) do
    raise "unexpected vectorized axes in evaluator for operation #{inspect(op)}: #{inspect(ans)}"
  end

  defp eval_apply(op, ans, state, caches) do
    {args, caches} = Tree.apply_args(ans, caches, &eval(&1, state, &2))

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

  defp pop_cache!([cache | caches], key) do
    {value, cache} = Map.pop!(cache, key)
    {value, [cache | caches]}
  end

  defp reshape_runtime_call_result(result, %Nx.Tensor{} = template) do
    # Single-tensor output: just ensure compatibility with the template.
    if not Nx.compatible?(template, result) do
      raise "expected the runtime_call function to match the given output template"
    end

    result
  end

  defp reshape_runtime_call_result(result, template_container) do
    # Container (tuple/map/etc) output: we expect the callback to return
    # a container with the same flattened tensor leaves as the template.
    if not Nx.compatible?(result, template_container) do
      raise "expected the runtime_call function to match the given output template"
    end

    result_leaves = Composite.flatten_list([result])

    List.to_tuple(result_leaves)
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

  # Handle ID-based evaluation for the new flattened representation
  defp composite_eval(id, state, caches) when is_reference(id) do
    eval(id, state, caches)
  end

  # Handle structs (including Container) - check if they contain IDs
  defp composite_eval(%_{} = struct, state, caches) do
    if is_id_based?(struct) do
      # Struct contains IDs - traverse and evaluate them
      Composite.traverse(struct, caches, fn element, caches ->
        cond do
          is_reference(element) -> eval(element, state, caches)
          is_id_based?(element) -> composite_eval(element, state, caches)
          true -> eval(element, state, caches)
        end
      end)
    else
      # Regular struct without IDs
      Composite.traverse(struct, caches, &eval(&1, state, &2))
    end
  end

  # Handle tuples - recursively evaluate each element
  defp composite_eval(tuple, state, caches) when is_tuple(tuple) do
    if tuple_is_composite?(tuple) do
      # This is a tuple that may contain IDs or nested structures
      {result_list, caches} =
        tuple
        |> Tuple.to_list()
        |> Enum.map_reduce(caches, fn element, caches ->
          composite_eval(element, state, caches)
        end)
      
      {List.to_tuple(result_list), caches}
    else
      # Use Composite.traverse for tensor tuples (no IDs involved)
      Composite.traverse(tuple, caches, &eval(&1, state, &2))
    end
  end

  # Handle maps - recursively evaluate each value
  defp composite_eval(map, state, caches) when is_map(map) and not is_struct(map) do
    if map_is_composite?(map) do
      # This is a map that may contain IDs or nested structures
      {result_map, caches} =
        Enum.map_reduce(map, caches, fn {key, value}, caches ->
          {evaluated_value, caches} = composite_eval(value, state, caches)
          {{key, evaluated_value}, caches}
        end)
      
      {Map.new(result_map), caches}
    else
      # Use Composite.traverse for maps containing tensors
      Composite.traverse(map, caches, &eval(&1, state, &2))
    end
  end

  # Fallback for any composite structure - handles both tensors and ID-based structures
  defp composite_eval(composite, state, caches) do
    # Use Composite.traverse with a callback that can handle both IDs and nested structures
    Composite.traverse(composite, caches, fn element, caches ->
      cond do
        # Direct ID - evaluate it
        is_reference(element) ->
          eval(element, state, caches)
        
        # Nested composite structure with IDs - recursively evaluate
        is_id_based?(element) ->
          composite_eval(element, state, caches)
        
        # Regular element (tensor, number, etc.) - evaluate normally
        true ->
          eval(element, state, caches)
      end
    end)
  end

  # Check if a tuple contains IDs or nested ID-based structures
  defp tuple_is_composite?(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.any?(&is_id_based?/1)
  end

  # Check if a map contains IDs or nested ID-based structures
  defp map_is_composite?(map) do
    Enum.any?(map, fn {_k, v} -> is_id_based?(v) end)
  end

  # Check if a value is ID-based (either a reference or a structure containing references)
  defp is_id_based?(value) when is_reference(value), do: true
  
  defp is_id_based?(tuple) when is_tuple(tuple) do
    tuple |> Tuple.to_list() |> Enum.any?(&is_id_based?/1)
  end
  
  defp is_id_based?(map) when is_map(map) and not is_struct(map) do
    Enum.any?(map, fn {_k, v} -> is_id_based?(v) end)
  end
  
  # Check structs by directly checking field values (non-recursive for structs to avoid infinite loops)
  defp is_id_based?(%_{} = struct) do
    # Convert struct to map and check all values
    # Only check for direct references, tuples, or maps - don't recurse into nested structs
    struct
    |> Map.from_struct()
    |> Map.values()
    |> Enum.any?(fn
      value when is_reference(value) -> true
      value when is_tuple(value) -> value |> Tuple.to_list() |> Enum.any?(&is_reference/1)
      value when is_map(value) and not is_struct(value) -> 
        Enum.any?(value, fn {_k, v} -> is_reference(v) end)
      _ -> false
    end)
  end
  
  defp is_id_based?(_), do: false

  defp composite_to_params(composite) do
    composite |> composite_to_params([]) |> Enum.reverse()
  end

  defp composite_to_params(tuple, acc) when is_tuple(tuple) do
    Enum.reduce(Tuple.to_list(tuple), acc, &composite_to_params/2)
  end

  defp composite_to_params(other, acc) do
    [fn -> other end | acc]
  end

  # Debug hook: print/save node info if enabled and not already printed
  defp debug_node(_ans, _res, %{debug_options: nil}), do: :ok

  defp debug_node(%Nx.Tensor{data: expr}, res, state) do
    %Expr{id: id} = expr
    %{debug_options: opts} = state
    opts = Keyword.validate!(opts, [:inspect_limit, :save_path])

    if [] == :ets.lookup(state.printed_nodes, id) do
      inspect_limit = opts[:inspect_limit]
      save_path = opts[:save_path]
      node_info = format_node_info(expr, res, inspect_limit)

      :ok =
        if save_path do
          save_node_info_to_file(save_path, id, node_info)
        else
          IO.puts(node_info)
        end

      :ets.insert(state.printed_nodes, {id})
    end
  end

  defp format_node_info(%Expr{id: id, op: op, args: args}, res, inspect_limit) do
    id_str = ref_to_string(id)

    inspect_opts =
      case inspect_limit do
        nil -> [custom_options: [print_id: true]]
        limit -> [custom_options: [print_id: true], limit: limit]
      end

    args_code =
      args
      |> Enum.map(fn arg ->
        inspected =
          arg
          |> inspect(inspect_opts)
          |> String.trim()

        "  #{inspect(inspected)}"
      end)
      |> Enum.join(",\n")

    # Format result as serialized tensor
    result_code = "result = #{serialize_tensor(res)}"

    """
    node_id = "#{id_str}"
    operation = #{inspect(op)}

    args = [
    #{args_code}
    ]

    # Result:
    #{result_code}
    """
  end

  defp serialize_tensor(%Nx.Tensor{data: %Expr{id: id}} = _tensor) when is_reference(id) do
    # This is an unevaluated expression, not a concrete tensor
    # Show the Node ID so users can find which file contains this tensor
    id_str = :erlang.ref_to_list(id) |> List.to_string() |> String.replace(["#Ref<", ">"], "")
    "# See Node ID: #{id_str}"
  end

  defp serialize_tensor(%Nx.Tensor{data: %Expr{}} = _tensor) do
    # Expression without a valid reference ID
    "# <unevaluated expression>"
  end

  defp serialize_tensor(%Nx.Tensor{} = tensor) do
    # Get the backend information from the tensor's data
    {backend, backend_opts} =
      case tensor.data do
        %backend_mod{} -> {backend_mod, []}
        _ -> Nx.default_backend()
      end

    # Convert tensor to binary and get metadata
    binary = Nx.to_binary(tensor)
    type = tensor.type
    shape = tensor.shape
    names = tensor.names

    # Format the binary as a binary literal
    binary_str = inspect(binary, limit: :infinity)

    # Build the executable Nx code
    backend_str = "{#{inspect(backend)}, #{inspect(backend_opts)}}"

    code = "Nx.from_binary(#{binary_str}, #{inspect(type)}, backend: #{backend_str})"

    # Add reshape if needed (non-scalar)
    code =
      if shape != {} do
        shape_str = inspect(shape)
        code <> " |> Nx.reshape(#{shape_str})"
      else
        code
      end

    # Add rename if there are non-nil names
    code =
      if Enum.any?(names, fn name -> not is_nil(name) end) do
        names_list = inspect(names)
        code <> " |> Nx.rename(#{names_list})"
      else
        code
      end

    code
  end

  defp serialize_tensor(other) do
    # For non-tensor values (numbers, tuples, etc.)
    inspect(other)
  end

  defp save_node_info_to_file(save_path, id, node_info) do
    sanitized_id = id |> ref_to_string() |> String.replace(".", "_")
    file = Path.join(save_path, "node_#{sanitized_id}.exs")
    File.write!(file, node_info)
  end

  defp ref_to_string(id) when is_reference(id) do
    id
    |> :erlang.ref_to_list()
    |> List.to_string()
    |> String.replace(["#Ref<", ">"], "")
  end
end
