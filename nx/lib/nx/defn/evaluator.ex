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
        result_tensor, [template | acc] ->
          # Reconstruct tensor from shallow template and evaluated result
          reconstructed = %Nx.Tensor{
            shape: template.shape,
            type: template.type,
            names: template.names,
            vectorized_axes: template.vectorized_axes,
            data: result_tensor.data
          }
          {reconstructed, acc}
      end)

    result
  end

  defp precompile(fun, vars, hooks) do
    expr = fun.(vars)

    # Devectorize and create shallow templates
    {devectorized_expr, output_templates} =
      Nx.Defn.Composite.traverse(expr, [], fn tensor, acc ->
        devectorized = Nx.devectorize(tensor)
        shallow_template = make_shallow_template(tensor)
        {devectorized, [shallow_template | acc]}
      end)

    # Compute cache on devectorized expression
    state = %{hooks: hooks, parent_ids: nil, current_ids: nil}
    {cache, templates} = init_compute_cache(devectorized_expr, state)

    # Make devectorized_expr shallow to reduce closure size
    shallow_expr = make_shallow_expr(devectorized_expr)

    # Make templates shallow to reduce closure size
    # Templates are indexed by ID and used in eval_node_info, so we need to preserve structure
    # but make the args shallow where possible
    shallow_templates = make_templates_shallow(templates)

    {shallow_expr, Enum.reverse(output_templates), cache, shallow_templates}
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
    # If the wrapped expression has no ID (e.g., it's a literal), we store the expression itself
    case extract_id(expr) do
      nil -> {:metadata_literal, expr}
      id -> {:metadata, id}
    end
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

  # Create a shallow template that only contains metadata needed for output reconstruction
  defp make_shallow_template(%Nx.Tensor{shape: shape, type: type, names: names, vectorized_axes: vectorized_axes}) do
    %{shape: shape, type: type, names: names, vectorized_axes: vectorized_axes}
  end

  # Replace Expr data with shallow references in composite structure
  # This makes the devectorized_expr small enough to not hang :erts_debug.size()
  defp make_shallow_expr(composite) do
    Composite.traverse(composite, fn
      %Nx.Tensor{data: %Expr{id: id}} = tensor when is_reference(id) ->
        %{tensor | data: {:ref, id}}
      other ->
        other
    end)
  end

  # Make templates shallow to reduce closure size
  # Templates must preserve Expr structure for evaluation, but we can make args shallow
  defp make_templates_shallow(templates) do
    Map.new(templates, fn {id, tensor} ->
      {id, make_template_tensor_shallow(tensor)}
    end)
  end

  # Make a template tensor shallow based on its operation type
  # Control flow ops need to preserve nested expression trees, but can make data args shallow
  defp make_template_tensor_shallow(%Nx.Tensor{data: %Expr{op: op, args: args} = expr_data} = tensor) do
    shallow_args = case op do
      # For :while, args are [initial, arg, condition, block]
      # Keep condition/block as full Expr trees (evaluated in nested scope)
      # Make initial/arg shallow (just references to other nodes)
      :while ->
        [initial, arg, condition, block] = args
        [make_arg_shallow(initial), make_arg_shallow(arg), condition, block]
      
      # For :fun, args are [args_template, expr, mfa]
      # Keep expr as full tree (the function body)
      # Make args_template and mfa shallow
      :fun ->
        [args_template, expr, mfa] = args
        [make_arg_shallow(args_template), expr, make_arg_shallow(mfa)]
      
      # For :cond, args are [clauses, last] where clauses = [{pred, body}, ...]
      # Keep preds and bodies as full trees (conditional branches)
      :cond ->
        args  # Keep all as full trees
      
      # For :optional, args are [call, expr, callback]
      # Keep expr as full tree (the fallback computation)
      # The call tensor also needs special handling to preserve its Expr structure
      :optional ->
        [call, expr, callback] = args
        [make_optional_call_shallow(call), expr, make_arg_shallow(callback)]
      
      # For all other ops, make all args shallow
      _ ->
        Enum.map(args, &make_arg_shallow/1)
    end
    
    # Use Nx.to_template for clean base tensor, then restore Expr with shallow args
    clean_base = Nx.to_template(tensor)
    %{clean_base | data: %{expr_data | args: shallow_args}}
  end

  # For non-Expr tensors (shouldn't happen in templates, but handle gracefully)
  defp make_template_tensor_shallow(tensor), do: tensor

  # Make an arg shallow - replace Expr tensors with minimal ref tensors
  # We keep them as tensors (not bare {:ref, id} tuples) so Tree.apply_args can traverse them
  defp make_arg_shallow(%Nx.Tensor{data: %Expr{id: id}} = tensor) when is_reference(id) do
    # Use minimal tensor with {:ref, id} - just like make_shallow_expr does
    %{tensor | data: {:ref, id}}
  end

  defp make_arg_shallow(tuple) when is_tuple(tuple) do
    tuple |> Tuple.to_list() |> Enum.map(&make_arg_shallow/1) |> List.to_tuple()
  end

  defp make_arg_shallow(list) when is_list(list) do
    Enum.map(list, &make_arg_shallow/1)
  end

  defp make_arg_shallow(other), do: other

  # Special handling for :optional call arg
  # The call tensor has its own args that need to remain as Expr structure for Tree.apply_args
  defp make_optional_call_shallow(%Nx.Tensor{data: %Expr{args: call_args} = expr_data} = call_tensor) do
    # Make the call's args shallow, but keep the Expr structure
    shallow_call_args = Enum.map(call_args, &make_arg_shallow/1)
    clean_base = Nx.to_template(call_tensor)
    %{clean_base | data: %{expr_data | args: shallow_call_args}}
  end

  defp make_optional_call_shallow(other), do: make_arg_shallow(other)

  defp compute_cache(%Nx.Tensor{data: %Expr{op: :constant}}, _state, acc) do
    acc
  end

  defp compute_cache(%Nx.Tensor{data: %Expr{id: id, op: :metadata, args: [expr, _meta]}} = tensor, state, {cache, templates}) do
    # First compute cache for the wrapped expression
    {cache, templates} = composite_compute_cache(expr, state, {cache, templates})

    # Then add the metadata node itself to the cache
    case state.parent_ids do
      %{^id => _} ->
        {Map.put_new(cache, id, tensor), templates}

      %{} ->
        case cache do
          %{^id => {node_info, counter}} ->
            {%{cache | id => {node_info, counter + 1}}, templates}
          %{} ->
            node_info = make_node_info(:metadata, tensor)
            templates = Map.put(templates, id, tensor)
            {Map.put(cache, id, {node_info, 1}), templates}
        end
    end
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
    # expr is already devectorized since we devectorized the parent expression
    {fun_cache, fun_templates} = init_compute_cache(expr, state)
    # Merge fun templates into global templates
    templates = Map.merge(templates, fun_templates)
    {Map.put(cache, [:fun | id], fun_cache), templates}
  end

  defp compute_cache(:while, %{data: %Expr{args: args, id: id}}, state, {cache, templates}) do
    [initial, _arg, pred, block] = args
    {cache, templates} = composite_compute_cache(initial, state, {cache, templates})
    # pred and block are already devectorized since we devectorized the parent expression
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
          # expr is already devectorized since we devectorized the parent expression
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

  # Shallow reference evaluation - unwrap and evaluate by ID
  defp eval(%Nx.Tensor{data: {:ref, id}}, state, caches) when is_reference(id) do
    eval(id, state, caches)
  end

  # ID-based evaluation - dispatch to node_info
  defp eval(id, state, [cache | caches]) when is_reference(id) do
    case cache do
      %{^id => {node_info, count}} when is_integer(count) ->
        {res, [cache | caches]} = eval_node_info(node_info, id, state, [cache | caches])
        state.gc && :erlang.garbage_collect(self())
        # Call debug_node with tensor template if debugging is enabled
        if state.debug_options do
          tensor = Map.fetch!(state.templates, id)
          debug_node(tensor, res, state)
        end
        {res, [decrement_cache(cache, id, count, res) | caches]}

      %{^id => {_node_info, count, res}} ->
        # Call debug_node with tensor template if debugging is enabled
        if state.debug_options do
          tensor = Map.fetch!(state.templates, id)
          debug_node(tensor, res, state)
        end
        {res, [decrement_cache(cache, id, count, res) | caches]}

      %{} ->
        # If we don't find the id in the current scope, look up parent scopes
        eval_parent_by_id(caches, id, state, [cache])
    end
  end

  # Tensor expressions - regular evaluation path
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

  defp eval_parent([], id, op, ans, _state, _acc) do
    raise "trying to read evaluator cache that has expired for ID #{inspect(id)} (op: #{inspect(op)}) during expression:\n\n#{inspect(ans)}\n\n" <>
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
        # Call debug_node with tensor template if debugging is enabled
        if state.debug_options do
          tensor = Map.fetch!(state.templates, id)
          debug_node(tensor, res, state)
        end
        {res, Enum.reverse(acc, [Map.put(cache, id, {node_info, count, res}) | caches])}

      %{} ->
        eval_parent_by_id(caches, id, state, [cache | acc])
    end
  end

  defp eval_parent_by_id([], id, _state, _acc) do
    raise "trying to read evaluator cache that has expired for ID #{inspect(id)}\n\n" <>
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

  defp eval_node_info({:metadata_literal, expr}, _id, state, caches) do
    # The wrapped expression is a literal (no ID), evaluate it directly
    # Use composite_eval to handle tuples and other composite structures
    composite_eval(expr, state, caches)
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
    # Only pass while_cache - nested expressions should be self-contained
    result = while(initial, condition, block, state, [while_cache])
    {result, caches}
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
      # Create clean output template without Expr data
      # Backends use this template only for shape/type metadata
      out_template =
        case call do
          %{type: {:tuple, _}} ->
            # For tuple outputs, strip Expr data from all tuple elements
            Composite.traverse(out, fn
              %Nx.Tensor{shape: shape, type: type, names: names} = tensor ->
                case tensor.data do
                  %Expr{} -> Nx.template(shape, type, names: names)
                  {:ref, _} -> Nx.template(shape, type, names: names)
                  _ -> tensor
                end
              other -> other
            end)

          _ ->
            # For single outputs, use call metadata to create clean template
            case call.data do
              %Expr{} -> Nx.template(call.shape, call.type, names: call.names)
              {:ref, _} -> Nx.template(call.shape, call.type, names: call.names)
              _ -> call
            end
        end

      {apply(backend, call.data.op, [out_template | args]), caches}
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

  # Simplified composite_eval - now that all references are wrapped in tensors
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

  # Format an arg for debug output - handle both full Expr tensors and shallow refs
  defp format_debug_arg(%Nx.Tensor{data: {:ref, id}}, _inspect_opts) do
    # For shallow refs, just show a simplified representation
    "#Nx.Tensor<ref: #{ref_to_string(id)}>"
  end

  defp format_debug_arg(arg, inspect_opts) do
    inspect(arg, inspect_opts)
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
          |> format_debug_arg(inspect_opts)
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
