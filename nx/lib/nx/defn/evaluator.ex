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

    {expr, output, cache} = precompile(fun, vars, hooks)

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
        printed_nodes: printed_nodes
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
    cache = init_compute_cache(expr, state)
    {expr, output} = Nx.Defn.Composite.traverse(expr, [], &{Nx.devectorize(&1), [&1 | &2]})
    {expr, Enum.reverse(output), cache}
  end

  defp init_compute_cache(expr, state) do
    state = %{state | parent_ids: %{}, current_ids: Tree.scope_ids(expr, %{})}
    composite_compute_cache(expr, state, %{})
  end

  defp composite_compute_cache(expr, state, cache) do
    Composite.reduce(expr, cache, &compute_cache(&1, state, &2))
  end

  defp compute_cache(%Nx.Tensor{data: %Expr{op: :constant}}, _state, cache) do
    cache
  end

  defp compute_cache(%Nx.Tensor{data: %Expr{op: :metadata, args: [expr, _meta]}}, state, cache) do
    composite_compute_cache(expr, state, cache)
  end

  defp compute_cache(%Nx.Tensor{data: %Expr{id: id, op: op}} = tensor, state, cache) do
    case state.parent_ids do
      # If the id exists in the parent, the parent will compute it.
      %{^id => _} ->
        Map.put_new(cache, id, tensor)

      %{} ->
        case cache do
          %{^id => counter} -> %{cache | id => counter + 1}
          %{} -> compute_cache(op, tensor, state, Map.put(cache, id, 1))
        end
    end
  end

  defp compute_cache(:fun, %{data: %Expr{id: id, args: args}}, state, cache) do
    [_args, expr, _mfa] = args
    fun_cache = init_compute_cache(expr, state)
    Map.put(cache, [:fun | id], fun_cache)
  end

  defp compute_cache(:while, %{data: %Expr{args: args, id: id}}, state, cache) do
    [initial, _arg, pred, block] = args
    cache = composite_compute_cache(initial, state, cache)
    while_cache = init_compute_cache({pred, block}, state)
    Map.put(cache, [:while | id], while_cache)
  end

  defp compute_cache(:optional, %{data: %Expr{args: args, id: id}}, state, cache) do
    [call, expr, _callback] = args
    %{data: %{args: call_args_in, op: call_name}} = call

    {call_args, opts} = Enum.split_while(call_args_in, &(not is_list(&1)))

    cache = Enum.reduce(call_args, cache, &compute_cache(&1, state, &2))
    key = computation_key(call_name, call_args ++ opts)

    {optional_expr_cache, cache} =
      case cache do
        %{^key => optional_expr_cache} ->
          {optional_expr_cache, cache}

        %{} ->
          optional_expr_cache = {expr, init_compute_cache(expr, state)}
          {optional_expr_cache, Map.put(cache, key, optional_expr_cache)}
      end

    Map.put(cache, [:optional | id], optional_expr_cache)
  end

  defp compute_cache(:cond, %{data: %Expr{args: [clauses, last], id: id}}, state, cache) do
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
      Enum.map_reduce(clause_caches, {%{}, cache}, fn clause_cache, seen_ids_cache ->
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
                  cache = composite_compute_cache(tensor, state, cache)
                  {[], {Map.put(seen_ids, id, true), cache}}
              end

            {id, counter}, seen_ids_cache ->
              {[{id, counter}], seen_ids_cache}
          end)

        {Map.new(clause_cache), seen_ids_cache}
      end)

    Map.put(cache, [:cond | id], {clauses_cache, last_cache, Map.keys(all_ids)})
  end

  defp compute_cache(:token, %{data: %Expr{args: [token], id: id}}, state, cache) do
    hooks = state.hooks

    {hooks, cache} =
      Enum.map_reduce(token.hooks, cache, fn
        %{callback: callback, expr: expr, name: name}, cache ->
          hook_fun = hooks[name] || callback

          cond do
            hook_fun -> {hook_fun, composite_compute_cache(expr, state, cache)}
            Tree.has_hooks?(expr, hooks) -> {true, composite_compute_cache(expr, state, cache)}
            true -> {false, cache}
          end
      end)

    Map.put(cache, [:token | id], hooks)
  end

  defp compute_cache(_op, tensor, state, cache) do
    {_, acc} = Tree.apply_args(tensor, cache, &{&1, compute_cache(&1, state, &2)})
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
      %{^id => count} when is_integer(count) ->
        {res, [cache | caches]} = eval_apply(op, ans, state, [cache | caches])
        state.gc && :erlang.garbage_collect(self())
        debug_node(ans, res, state)
        {res, [decrement_cache(cache, id, count, res) | caches]}

      %{^id => {count, res}} ->
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
