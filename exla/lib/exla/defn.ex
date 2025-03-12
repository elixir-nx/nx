defmodule EXLA.Defn do
  @moduledoc false

  require Logger
  require EXLA.Defn.Outfeed, as: Outfeed
  alias Nx.Defn.{Composite, Expr, Tree}
  alias Nx.Tensor, as: T

  alias EXLA.Typespec
  alias EXLA.MLIR.Value
  alias EXLA.MLIR.Function

  @doc false
  def __partitions_options__(options) do
    client_name = Keyword.get_lazy(options, :client, &EXLA.Client.default_name/0)
    device_count = EXLA.Client.fetch!(client_name).device_count

    Enum.map(1..device_count//1, &Keyword.put(options, :device_id, &1 - 1))
  end

  @doc false
  def __to_backend__(options) do
    client_name = Keyword.get_lazy(options, :client, &EXLA.Client.default_name/0)

    device_id =
      Keyword.get_lazy(options, :device_id, fn ->
        EXLA.Client.fetch!(client_name).default_device_id
      end)

    {EXLA.Backend, [client: client_name, device_id: device_id]}
  end

  @doc false
  def __jit__(key, vars, fun, args_list, options) do
    __compile__(key, vars, fun, options).(args_list)
  end

  @doc false
  def __compile__(key, vars, fun, options) do
    {run_options, compile_options} = Keyword.pop(options, :run_options, [])
    debug? = Keyword.get(compile_options, :debug, false)
    callback = &to_computation(&1, &2, &3, &4, &5, compile_options)

    {executable, {used_inputs, outputs, outfeed, _input_typespecs?}} =
      compile(key, vars, fun, compile_options, 0, [], callback)

    if compile_options[:module_compilation] == :to_mlir do
      throw({:mlir_module, executable.ref, MapSet.new(Map.keys(used_inputs)), outputs})
    end

    fn [args] ->
      {time, lock} =
        :timer.tc(fn ->
          EXLA.Defn.Lock.lock(run_key(executable))
        end)

      debug? && Logger.debug("EXLA device #{executable.device_id} lock in #{us_to_ms(time)}ms")

      {time, res} =
        :timer.tc(fn ->
          maybe_outfeed(lock, executable, args, used_inputs, outputs, outfeed, run_options)
        end)

      debug? &&
        Logger.debug("EXLA execution on device #{executable.device_id} in #{us_to_ms(time)}ms")

      res
    end
  end

  defp to_computation(%Function{} = function, expr, used_typespecs, outfeed, client, options) do
    params =
      Enum.zip_with(used_typespecs, Function.get_arguments(function), fn {pos, _typespec}, arg ->
        {pos, arg}
      end)

    unless client do
      raise ArgumentError, "missing client"
    end

    state = %{
      client: client,
      precision: Keyword.get(options, :precision, :default),
      builder: function,
      params: Map.new(params ++ outfeed.infeeds),
      scope_ids: Tree.scope_ids(expr)
    }

    {res, cache} = recur_flatten(expr, state, new_cache(outfeed))
    outfeed = cache |> get_outfeed() |> Outfeed.close(function)
    Value.func_return(function, res)
    outfeed
  end

  defp maybe_outfeed(lock, executable, args, used_inputs, outputs, outfeed, run_options)
       when Outfeed.will_outfeed(outfeed) do
    {buffers, infeeds} =
      EXLA.Defn.Buffers.split_by_value(args, used_inputs, fn
        arg, _i, nil -> EXLA.Defn.Buffers.from_nx!(arg, executable, true)
        arg, i, _depth -> {i, EXLA.Defn.Buffers.from_nx!(arg, executable, false)}
      end)

    {:ok, runner} =
      EXLA.Defn.Runner.start_link(lock, fn ->
        EXLA.Executable.run(executable, [Enum.reverse(buffers)], run_options)
      end)

    {:ok, outfeed_pid} =
      Outfeed.start_child(executable, outfeed, Process.group_leader(), Map.new(infeeds))

    _ = EXLA.Defn.Lock.transfer(lock, fn -> send(runner, lock) end, outfeed_pid)
    ref = Process.monitor(outfeed_pid)

    receive do
      {:DOWN, ^ref, _, _, _} ->
        [result] = EXLA.Defn.Runner.read(runner)
        [EXLA.Defn.Buffers.to_nx!(result, outputs)]
    end
  end

  defp maybe_outfeed(lock, executable, args, used_inputs, outputs, _outfeed, run_options) do
    try do
      buffers =
        EXLA.Defn.Buffers.filter_by_indexes(args, used_inputs, fn arg, _i ->
          EXLA.Defn.Buffers.from_nx!(arg, executable)
        end)

      EXLA.Executable.run(executable, [buffers], run_options)
    else
      [result] ->
        [EXLA.Defn.Buffers.to_nx!(result, outputs)]
    after
      EXLA.Defn.Lock.unlock(lock)
    end
  end

  defp run_key(%{client: %{ref: ref}, device_id: device_id}), do: [ref | device_id]

  ## Compile

  defp compile(key, vars, fun, options, used_buffers, used_inputs, to_computation) do
    {cache, options} = Keyword.pop(options, :cache, true)
    {hooks, options} = Keyword.pop(options, :hooks, %{})
    {debug?, options} = Keyword.pop(options, :debug, false)
    {lazy_transfers, options} = Keyword.pop(options, :lazy_transfers, :opt_in)

    {client_name, options} = Keyword.pop_lazy(options, :client, &EXLA.Client.default_name/0)
    client = EXLA.Client.fetch!(client_name)

    {args_key, reverse_args_identifiers} =
      Enum.map_reduce(vars, [], fn var, acc ->
        Nx.Defn.Composite.traverse(var, acc, fn
          %T{vectorized_axes: vectorized_axes} = t, acc ->
            %T{type: type, shape: shape, names: names} = Nx.devectorize(t)
            identifier = {type, shape, names}
            cache_key = {type, shape, names, vectorized_axes}
            {cache_key, [identifier | acc]}
        end)
      end)

    disk_key = %{
      client: client.name,
      args: args_key,
      lazy_transfers: lazy_transfers,
      hooks: Map.keys(hooks),
      options: options
    }

    EXLA.Defn.Disk.cache(cache, client, disk_key, debug?, fn ->
      {{expr_cache_fun, comp_cache_fun}, options} =
        if cache do
          Keyword.pop(options, EXLA, {&EXLA.Defn.LockedCache.run/2, &EXLA.Defn.LockedCache.run/2})
        else
          cache_fun = fn _key, fun -> fun.() end
          {{cache_fun, cache_fun}, Keyword.delete(options, EXLA)}
        end

      {eval_time, {expr, {ref, outputs, {used_inputs, defined_hooks}}}} =
        :timer.tc(fn ->
          expr_cache_fun.({key, args_key, lazy_transfers}, fn ->
            expr = fun.(vars)
            inputs_and_hooks = Outfeed.used_inputs_and_hooks(expr, used_inputs, lazy_transfers)
            {expr, {make_ref(), Nx.to_template(expr), inputs_and_hooks}}
          end)
        end)

      if debug? do
        hit_or_miss = if expr, do: "miss", else: "hit"

        Logger.debug(
          "EXLA defn evaluation #{inspect(key)} cache #{hit_or_miss} in #{us_to_ms(eval_time)}ms"
        )
      end

      outfeed = Outfeed.new(hooks, defined_hooks)
      comp_key = {ref, client.name, outfeed.used_hooks, lazy_transfers, options}

      {comp_time, {evaled, {xla_time, executable, inputs_and_typespecs, outfeed}}} =
        :timer.tc(fn ->
          comp_cache_fun.(comp_key, fn ->
            {reverse_inputs_and_typespecs, reverse_infeeds} =
              reverse_args_identifiers
              |> Enum.reverse()
              |> EXLA.Defn.Buffers.split_by_value(used_inputs, fn
                {type, shape, _names}, i, nil -> {i, Typespec.tensor(type, shape)}
                {type, shape, _names}, i, depth -> {i, depth, Typespec.tensor(type, shape)}
              end)

            inputs_and_typespecs = Enum.reverse(reverse_inputs_and_typespecs)

            comp_typespecs =
              for {i, typespec} <- inputs_and_typespecs, i >= used_buffers, do: typespec

            out_typespecs =
              [outputs]
              |> Nx.Defn.Composite.flatten_list()
              |> Enum.map(fn t ->
                t
                |> Nx.devectorize()
                |> then(&Typespec.tensor(&1.type, &1.shape))
              end)

            EXLA.MLIR.Module.new(comp_typespecs, out_typespecs, fn builder ->
              # Only create the token when we know it will actually be
              # used, that is: streaming, lazy transfers or hooks
              outfeed =
                if reverse_infeeds != [] or hooks != %{} or defined_hooks != %{} do
                  outfeed
                  |> Outfeed.with_token(Value.create_token(builder))
                  |> Outfeed.add_infeeds(builder, reverse_infeeds)
                else
                  outfeed
                end

              expr = Nx.Defn.Composite.traverse(expr || fun.(vars), &Nx.devectorize/1)
              outfeed = to_computation.(builder, expr, inputs_and_typespecs, outfeed, client)

              {xla_time, executable} =
                :timer.tc(fn ->
                  EXLA.MLIR.Module.compile(
                    builder.module,
                    client,
                    comp_typespecs,
                    builder.return_typespecs,
                    options
                  )
                end)

              {:ok, {xla_time, executable, inputs_and_typespecs, %{outfeed | infeeds: []}}}
            end)
          end)
        end)

      cond do
        not debug? ->
          :ok

        evaled ->
          Logger.debug(
            "EXLA compilation #{inspect(key)} cache miss in #{us_to_ms(comp_time)}ms (#{us_to_ms(xla_time)}ms in XLA)"
          )

        true ->
          Logger.debug("EXLA compilation #{inspect(key)} cache hit in #{us_to_ms(comp_time)}ms")
      end

      if expr || evaled do
        measurements = %{
          eval_time: eval_time,
          compile_time: comp_time,
          total_time: eval_time + comp_time
        }

        :telemetry.execute([:exla, :compilation], measurements, %{key: key})
      end

      outfeed = Outfeed.with_user_hooks(outfeed, hooks)
      {executable, {used_inputs, outputs, outfeed, inputs_and_typespecs}}
    end)
  end

  defp us_to_ms(time), do: Float.round(time / 1000, 1)

  ## Operator handling

  defp recur_flatten(composite, state, cache) do
    {acc, cache} =
      Composite.reduce(composite, {[], cache}, fn %T{} = expr, {acc, cache} ->
        {expr, cache} = recur_operator(expr, state, cache)
        {[acc, expr], cache}
      end)

    {List.flatten(acc), cache}
  end

  defp recur_operator(%T{data: %Expr{id: id, op: op}} = expr, state, cache) do
    case cache do
      %{^id => res} ->
        {res, cache}

      %{} ->
        {res, cache} = cached_recur_operator(op, Nx.devectorize(expr), state, cache)
        {res, Map.put(cache, id, res)}
    end
  end

  defp cached_recur_operator(
         :while,
         %T{data: %Expr{args: args}},
         %{builder: %Function{} = function} = state,
         cache
       ) do
    [initial_arg, _arg, pred, body] = args

    initial =
      if token = get_token(cache) do
        {token, initial_arg}
      else
        initial_arg
      end

    {initial, cache} = recur_composite(initial, state, cache)

    {pred_computation, cache} = mlir_while_computation(pred, initial, {:pred, 8}, state, cache)
    {body_computation, cache} = mlir_while_computation(body, initial, :with_token, state, cache)

    results =
      Value.while(function, pred_computation, body_computation, List.flatten(initial))

    if get_token(cache) do
      [token | results] = results
      result = wrap_tuple_result(results, initial_arg)
      {result, update_token(cache, token)}
    else
      result = wrap_tuple_result(results, initial_arg)
      {result, cache}
    end
  end

  defp cached_recur_operator(:cond, %T{data: %Expr{args: args}} = t, state, cache) do
    [clauses, last] = args

    case clauses do
      [{pred, on_true}] ->
        to_if(pred, on_true, last, state, cache)

      _ ->
        # We convert cond into a nested tree of conds in order to compile it to ifs
        %T{data: %Expr{args: [[{pred, on_true}], on_false]}} =
          clauses
          |> Enum.reverse()
          |> Enum.reduce(last, fn {pred, on_true}, on_false ->
            update_in(t.data, fn data ->
              %{data | args: [[{pred, on_true}], on_false], id: make_ref()}
            end)
          end)

        to_if(pred, on_true, on_false, state, cache)
    end
  end

  defp cached_recur_operator(:parameter, %T{data: %Expr{args: [i]}}, state, cache) do
    {Map.fetch!(state.params, i), cache}
  end

  defp cached_recur_operator(:fun, %T{data: %Expr{args: args}, type: type}, state, cache) do
    [args, expr, {_, _, _}] = args
    {fun_computation(args, expr, type, state), cache}
  end

  defp cached_recur_operator(
         :optional,
         %T{
           data: %Expr{
             args: [
               %{data: %{op: :qr, args: [tensor, _opts]}},
               {%{type: {type_kind, _}} = q_expr, r_expr},
               _callback
             ]
           }
         },
         %{client: %EXLA.Client{platform: :host}, builder: %Function{}} = state,
         cache
       )
       when type_kind != :c do
    # We match only on platform: :host for MLIR, as we want to support
    # QR-on-cpu as a custom call only in this case
    {tensor, cache} = recur_operator(tensor, state, cache) |> unwrap_single_tensor!()

    tensor =
      if op_type(tensor) != q_expr.type do
        to_type(tensor, q_expr.type)
      else
        tensor
      end

    {q, r} = Value.qr(tensor, expr_to_typespec(q_expr), expr_to_typespec(r_expr))
    {[q, r], cache}
  end

  defp cached_recur_operator(
         :optional,
         %T{
           data: %Expr{
             args: [
               %{data: %{op: :eigh, args: [tensor, _opts]}},
               {%{type: {evec_type_kind, _}} = eigenvecs_expr,
                %{type: {eval_type_kind, _}} = eigenvals_expr},
               _callback
             ]
           }
         },
         %{client: %EXLA.Client{platform: :host}, builder: %Function{}} = state,
         cache
       )
       when evec_type_kind != :c and eval_type_kind != :c do
    # We match only on platform: :host for MLIR, as we want to support
    # eigh-on-cpu as a custom call only in this case
    {tensor, cache} = recur_operator(tensor, state, cache) |> unwrap_single_tensor!()

    # convert to float and ensure that we're either using f32 or f64, because Eigen
    # only supports f32 and f64 easily.
    out_type = Nx.Type.merge(Nx.Type.to_floating(eigenvecs_expr.type), {:f, 32})

    tensor =
      if op_type(tensor) != out_type do
        to_type(tensor, out_type)
      else
        tensor
      end

    {eigenvecs, eigenvals} =
      Value.eigh(
        tensor,
        expr_to_typespec(%{eigenvecs_expr | type: out_type}),
        expr_to_typespec(%{eigenvals_expr | type: out_type})
      )

    {[to_type(eigenvecs, eigenvecs_expr.type), to_type(eigenvals, eigenvals_expr.type)], cache}
  end

  defp cached_recur_operator(
         :optional,
         %T{
           data: %Expr{
             args: [%{data: %{op: :take, args: [tensor, indices, opts]}}, expr, _callback]
           }
         },
         state,
         cache
       ) do
    axis = opts[:axis]
    {tensor, cache} = recur_operator(tensor, state, cache) |> unwrap_single_tensor!()
    {indices, cache} = recur_operator(indices, state, cache) |> unwrap_single_tensor!()

    tensor_rank = tensor |> op_shape() |> tuple_size()
    indices_rank = indices |> op_shape() |> tuple_size()
    result_rank = tensor_rank - 1 + indices_rank

    index_vector_dim = indices_rank
    slice_sizes = tensor |> op_shape() |> put_elem(axis, 1) |> Tuple.to_list()

    {left, right} = result_rank |> axes_for_rank() |> Enum.split(axis)
    offset_dims = left ++ Enum.drop(right, indices_rank)

    collapsed_slice_dims = [axis]
    start_index_map = [axis]

    result =
      Value.gather(
        tensor,
        indices,
        index_vector_dim,
        slice_sizes,
        offset_dims,
        collapsed_slice_dims,
        start_index_map,
        expr_to_typespec(expr)
      )

    {result, cache}
  end

  defp cached_recur_operator(
         :optional,
         %T{data: %Expr{args: [%{data: %{op: :top_k, args: [tensor, opts]}}, expr, _callback]}},
         state,
         cache
       ) do
    {tensor, cache} = recur_operator(tensor, state, cache) |> unwrap_single_tensor!()
    {values, idx} = expr
    typespecs = [expr_to_typespec(values), expr_to_typespec(idx)]
    results = Value.top_k(tensor, opts[:k], typespecs)
    {results, cache}
  end

  defp cached_recur_operator(
         :optional,
         %T{data: %Expr{args: [%{data: %{op: :fft2, args: [tensor, opts]}}, expr, _callback]}},
         state,
         cache
       ) do
    {tensor, cache} = recur_operator(tensor, state, cache) |> unwrap_single_tensor!()

    {fft2(&Value.fft(&1, :fft, &2, &3), [tensor, opts], expr, state), cache}
  end

  defp cached_recur_operator(
         :optional,
         %T{data: %Expr{args: [%{data: %{op: :ifft2, args: [tensor, opts]}}, expr, _callback]}},
         state,
         cache
       ) do
    {tensor, cache} = recur_operator(tensor, state, cache) |> unwrap_single_tensor!()

    {fft2(&Value.fft(&1, :ifft, &2, &3), [tensor, opts], expr, state), cache}
  end

  defp cached_recur_operator(:optional, %T{data: %Expr{args: args}}, state, cache) do
    [call, expr, _callback] = args
    %{data: %{args: in_args, op: op}} = call

    {args, opts} = Enum.split_while(in_args, &(not is_list(&1)))

    {call_args, cache} = Enum.map_reduce(args, cache, &recur_operator(&1, state, &2))
    key = computation_key(op, call_args ++ opts)

    {call_body, cache} =
      case cache do
        %{^key => computation} ->
          {computation, cache}

        %{} ->
          {computation, cache} = optional_computation("optional", call_args, expr, state, cache)
          {computation, Map.put(cache, key, computation)}
      end

    if token = get_token(cache) do
      typespecs = [Typespec.token() | container_to_typespecs(expr)]
      [token | result] = Value.call(state.builder, [token | call_args], call_body, typespecs)
      {wrap_tuple_result(result, expr), update_token(cache, token)}
    else
      typespecs = container_to_typespecs(expr)
      result = Value.call(state.builder, call_args, call_body, typespecs)
      {wrap_tuple_result(result, expr), cache}
    end
  end

  defp cached_recur_operator(
         :lu,
         %T{
           data: %Expr{args: [{p_expr, l_expr, u_expr}, %{type: {type_kind, _}} = tensor, _opts]}
         },
         %{client: %{platform: :host}} = state,
         cache
       )
       when type_kind != :c do
    # We only want to accelerate the LU operation for real inputs on the host device.
    # Otherwise, we use the default implementation in Nx.
    {tensor, cache} = recur_operator(tensor, state, cache) |> unwrap_single_tensor!()

    tensor =
      if op_type(tensor) != u_expr.type do
        to_type(tensor, u_expr.type)
      else
        tensor
      end

    {p, l, u} =
      Value.lu(
        tensor,
        expr_to_typespec(p_expr),
        expr_to_typespec(l_expr),
        expr_to_typespec(u_expr)
      )

    {[p, l, u], cache}
  end

  defp cached_recur_operator(:attach_token, %T{data: %Expr{args: [token, expr]}}, state, cache) do
    {op, cache} = recur_operator(expr, state, cache)
    {_, cache} = recur_operator(token, state, cache)
    {op, cache}
  end

  defp cached_recur_operator(:token, %T{data: %Expr{args: [token]}}, state, cache) do
    builder = state.builder

    cache =
      List.foldr(token.hooks, cache, fn %{name: name, expr: expr}, cache ->
        # First traverse the child because if it has hooks,
        # we need to handle them first
        {tuple, cache} = recur_flatten(expr, state, cache)

        cache
        |> get_outfeed()
        |> Outfeed.maybe_add_function_hook(builder, tuple, name, expr)
        |> then(&put_outfeed(cache, &1))
      end)

    {[], cache}
  end

  defp cached_recur_operator(op, expr, state, cache) do
    {args, cache} = Tree.apply_args(expr, cache, &recur_operator(&1, state, &2))
    {to_operator(op, args, expr, state), cache}
  end

  ## to_operator creation

  defp to_operator(:constant, [constant], ans, state) do
    op = to_constant(state.builder, constant, ans.type)

    if ans.shape == {} do
      op
    else
      Value.broadcast_in_dim(op, [], expr_to_typespec(ans))
    end
  end

  defp to_operator(:tensor, [tensor], _ans, state) do
    tensor = Nx.devectorize(tensor)

    case tensor.shape do
      {} ->
        to_constant(state.builder, Nx.to_number(tensor), tensor.type)

      shape ->
        Value.constant(
          state.builder,
          Nx.to_flat_list(tensor),
          Typespec.tensor(tensor.type, shape)
        )
    end
  end

  defp to_operator(:iota, [axis], ans, state) do
    EXLA.Lib.iota(state.builder, axis, expr_to_typespec(ans))
  end

  defp to_operator(:eye, [], %{type: type, shape: shape}, state) do
    iota_type = Nx.Type.merge_number({:u, 8}, Tuple.product(shape))
    iota_typespec = Typespec.tensor(iota_type, shape)
    rank = tuple_size(shape)

    i0 = Value.iota(state.builder, rank - 2, iota_typespec)
    i1 = Value.iota(state.builder, rank - 1, iota_typespec)

    typespec = Typespec.tensor({:pred, 8}, shape)
    Value.equal(i0, i1, typespec) |> to_type(type)
  end

  ## to_operator shape

  defp to_operator(:reshape, [%Value{} = op], ans, _state) do
    Value.reshape(op, expr_to_typespec(ans))
  end

  defp to_operator(:pad, [%Value{} = op, %Value{} = value, padding_config], ans, _state) do
    Value.pad(
      to_type(op, ans.type),
      to_type(value, ans.type),
      padding_config,
      expr_to_typespec(ans)
    )
  end

  defp to_operator(:broadcast, [%Value{} = op, _shape, axes], ans, _state) do
    Value.broadcast_in_dim(to_type(op, ans.type), axes, expr_to_typespec(ans))
  end

  defp to_operator(:transpose, [%Value{} = op, axes], ans, _state) do
    Value.transpose(op, axes, expr_to_typespec(ans))
  end

  defp to_operator(:squeeze, [%Value{} = op, _axes], ans, _state) do
    Value.reshape(op, expr_to_typespec(ans))
  end

  ## to_operator others

  defp to_operator(:metadata, [op, _metadata], _ans, _state) do
    case op do
      %Value{} ->
        op

      op when is_tuple(op) ->
        Tuple.to_list(op)
    end
  end

  defp to_operator(:elem, [op, index], _ans, _state) when is_list(op) do
    Enum.fetch!(op, index)
  end

  defp to_operator(
         :dot,
         [
           %Value{} = left,
           contract_axes1,
           batch_axes1,
           %Value{} = right,
           contract_axes2,
           batch_axes2
         ],
         ans,
         state
       ) do
    precision = state.precision

    Value.dot_general(
      left,
      right,
      {contract_axes1, batch_axes1, contract_axes2, batch_axes2},
      precision,
      expr_to_typespec(ans)
    )
  end

  defp to_operator(
         :conv,
         [operand, kernel, opts],
         ans,
         state
       ) do
    padding = opts[:padding]
    strides = opts[:strides]
    input_dilation = opts[:input_dilation]
    kernel_dilation = opts[:kernel_dilation]
    feature_group_count = opts[:feature_group_size]
    batch_group_count = opts[:batch_group_size]

    %{type: output_type} = ans

    dimension_numbers =
      {opts[:input_permutation], opts[:kernel_permutation], opts[:output_permutation]}

    # Ensure both types are floating
    operand = to_type(operand, output_type)
    kernel = to_type(kernel, output_type)

    Value.convolution(
      operand,
      kernel,
      strides,
      padding,
      input_dilation,
      kernel_dilation,
      dimension_numbers,
      feature_group_count,
      batch_group_count,
      state.precision,
      expr_to_typespec(ans)
    )
  end

  defp to_operator(
         :select,
         [%Value{} = pred, %Value{} = on_true, %Value{} = on_false],
         %{type: type, shape: shape} = ans,
         _state
       ) do
    pred = to_type(pred, {:pred, 8})

    typespec = expr_to_typespec(ans)

    on_true =
      on_true
      |> to_type(type)
      |> Value.broadcast_in_dim(broadcast_axes(op_shape(on_true), shape), typespec)

    on_false =
      on_false
      |> to_type(type)
      |> Value.broadcast_in_dim(broadcast_axes(op_shape(on_false), shape), typespec)

    Value.select(pred, on_true, on_false, typespec)
  end

  defp to_operator(:triangular_solve, [%Value{} = a, b, opts], %{type: type} = ans, _state) do
    left_side = Keyword.fetch!(opts, :left_side)
    lower = Keyword.fetch!(opts, :lower)
    transform = Keyword.fetch!(opts, :transform_a)

    case Value.get_typespec(b).shape do
      {dim} ->
        b_shape = {dim, 1}

        b =
          b
          |> to_type(type)
          |> Value.reshape(Typespec.tensor(type, b_shape))

        typespec = Typespec.tensor(type, b_shape)

        to_type(a, type)
        |> Value.triangular_solve(b, left_side, lower, transform, typespec)
        |> Value.reshape(Typespec.tensor(type, ans.shape))

      _ ->
        typespec = Typespec.tensor(type, ans.shape)

        to_type(a, type)
        |> Value.triangular_solve(to_type(b, type), left_side, lower, transform, typespec)
    end
  end

  ## to_operator element-wise

  defp to_operator(:negate, [%Value{} = op], ans, _state),
    do: Value.negate(op, expr_to_typespec(ans))

  defp to_operator(:abs, [%Value{} = op], ans, _state), do: Value.abs(op, expr_to_typespec(ans))

  defp to_operator(:sign, [%Value{} = op], ans, state) do
    typespec = expr_to_typespec(ans)

    case typespec.type do
      {:u, _} ->
        one = Value.constant(state.builder, [1], Typespec.to_shape(typespec, {}))

        one
        |> Value.broadcast_in_dim([], typespec)
        |> Value.min(op, typespec)

      _ ->
        Value.sign(op, typespec)
    end
  end

  defp to_operator(:right_shift, [%Value{} = left, %Value{} = right], out, _state) do
    op =
      if match?({:u, _}, out.type),
        do: :right_shift_logical,
        else: :right_shift_arithmetic

    apply_mlir_broadcasted_bin_op(op, out, left, right)
  end

  @bin_op [:add, :subtract, :multiply, :min, :max, :remainder, :pow, :divide, :atan2] ++
            [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift]

  defp to_operator(op, [%Value{} = left, %Value{} = right], out, _state)
       when op in @bin_op do
    apply_mlir_broadcasted_bin_op(op, out, left, right)
  end

  defp to_operator(:quotient, [left, right], ans, state) do
    to_operator(:divide, [to_type(left, ans.type), to_type(right, ans.type)], ans, state)
  end

  @bin_comp_op [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal]

  defp to_operator(op, [%Value{} = left, %Value{} = right], ans, _state)
       when op in @bin_comp_op do
    apply_mlir_broadcasted_bin_op(op, ans, left, right)
  end

  @bin_pred_op [logical_and: :bitwise_and, logical_or: :bitwise_or, logical_xor: :bitwise_xor]

  for {logical, bitwise} <- @bin_pred_op do
    defp to_operator(unquote(logical), [%Value{} = left, %Value{} = right], ans, _state) do
      apply_mlir_broadcasted_bin_op(
        unquote(bitwise),
        ans,
        to_mlir_logical(left),
        to_mlir_logical(right)
      )
    end
  end

  @unary_op [:exp, :expm1, :log, :log1p, :sigmoid, :cos, :sin, :tan, :tanh, :sqrt, :rsqrt, :cbrt] ++
              [:bitwise_not, :count_leading_zeros, :population_count, :cosh, :sinh, :acos] ++
              [:asin, :atan, :floor, :ceil, :round, :acosh, :asinh, :atanh, :erf] ++
              [:erfc, :erf_inv, :conjugate]

  defp to_operator(op, [%Value{} = arg], %{type: type} = ans, _state)
       when op in @unary_op do
    apply(Value, op, [to_type(arg, type), expr_to_typespec(ans)])
  end

  defp to_operator(:fft, [%Value{} | _] = args, out, state),
    do: fft(&Value.fft(&1, :fft, &2, &3), args, out, state)

  defp to_operator(:ifft, [%Value{} | _] = args, out, state),
    do: fft(&Value.fft(&1, :ifft, &2, &3), args, out, state)

  defp to_operator(:is_nan, [%Value{} = arg], out, _state),
    do: Value.is_nan(arg, expr_to_typespec(out))

  defp to_operator(:is_infinity, [%Value{} = arg], out, _state),
    do: Value.is_infinity(arg, expr_to_typespec(out))

  # These operations do the type conversion implicitly, and so
  # we cannot mess with the output type (e.g. the to_type conversion)
  # because it will throw an error
  @complex_op [:real, :imag]

  defp to_operator(op, [%Value{} = arg], ans, _state)
       when op in @complex_op do
    maybe_cast_arg =
      if Nx.Type.integer?(op_type(arg)) do
        to_type(arg, ans.type)
      else
        arg
      end

    apply(Value, op, [maybe_cast_arg, expr_to_typespec(ans)])
  end

  defp to_operator(:as_type, [arg], %{type: type}, _state) do
    to_type(arg, type)
  end

  defp to_operator(:bitcast, [%Value{} = arg], ans, _state) do
    if op_type(arg) == ans.type do
      arg
    else
      Value.bitcast_convert(arg, expr_to_typespec(ans))
    end
  end

  ## to_operator reduction

  defp to_operator(:all, [arg, opts], %{shape: shape}, %{builder: %Function{}} = state) do
    to_aggregate(:bitwise_and, {:u, 8}, shape, to_mlir_logical(arg), 1, opts, state)
  end

  defp to_operator(:any, [arg, opts], %{shape: shape}, %{builder: %Function{}} = state) do
    to_aggregate(:bitwise_or, {:u, 8}, shape, to_mlir_logical(arg), 0, opts, state)
  end

  defp to_operator(:all, [arg, opts], %{shape: shape}, state) do
    to_aggregate(:bitwise_and, {:pred, 8}, shape, arg, 1, opts, state)
  end

  defp to_operator(:any, [arg, opts], %{shape: shape}, state) do
    to_aggregate(:bitwise_or, {:pred, 8}, shape, arg, 0, opts, state)
  end

  defp to_operator(:sum, [arg, opts], %{type: type, shape: shape}, state) do
    to_aggregate(:add, type, shape, arg, 0, opts, state)
  end

  defp to_operator(:product, [arg, opts], %{type: type, shape: shape}, state) do
    to_aggregate(:multiply, type, shape, arg, 1, opts, state)
  end

  defp to_operator(:reduce_max, [arg, opts], %{type: type, shape: shape}, state) do
    min_number = EXLA.Lib.min_number(state.builder, type)
    to_aggregate(:max, type, shape, arg, min_number, opts, state)
  end

  defp to_operator(:reduce_min, [arg, opts], %{type: type, shape: shape}, state) do
    max_number = EXLA.Lib.max_number(state.builder, type)
    to_aggregate(:min, type, shape, arg, max_number, opts, state)
  end

  defp to_operator(
         :reduce,
         [%Value{} = arg, %Value{} = acc, opts, fun],
         %{type: type, shape: shape},
         _state
       ) do
    arg = to_type(arg, type)
    keep_axes = opts[:keep_axes]
    reduce_axes = reduce_axes(arg, opts[:axes])

    typespec = Typespec.tensor(type, remove_axes(op_shape(arg), reduce_axes))
    [result] = Value.reduce(fun, [to_type(acc, type)], [arg], reduce_axes, [typespec])

    if keep_axes do
      Value.reshape(result, Typespec.tensor(type, shape))
    else
      result
    end
  end

  defp to_operator(:window_sum, [arg, window_dims, opts], ans, state) do
    to_window_aggregate(:add, ans, arg, 0, window_dims, opts, state)
  end

  defp to_operator(:window_max, [arg, window_dims, opts], %{type: type} = ans, state) do
    min_number = EXLA.Lib.min_number(state.builder, type)
    to_window_aggregate(:max, ans, arg, min_number, window_dims, opts, state)
  end

  defp to_operator(:window_min, [arg, window_dims, opts], %{type: type} = ans, state) do
    max_number = EXLA.Lib.max_number(state.builder, type)
    to_window_aggregate(:min, ans, arg, max_number, window_dims, opts, state)
  end

  defp to_operator(:window_product, [arg, window_dims, opts], ans, state) do
    to_window_aggregate(:multiply, ans, arg, 1, window_dims, opts, state)
  end

  defp to_operator(
         :window_reduce,
         [arg, acc, window_dimensions, opts, fun],
         %{type: type} = ans,
         %{builder: %Function{}}
       ) do
    padding_config = opts[:padding]
    strides = opts[:strides]
    window_dilations = opts[:window_dilations]
    arg = to_type(arg, type)
    acc = to_type(acc, type)

    [result] =
      Value.window_reduce(
        fun,
        [acc],
        [arg],
        Tuple.to_list(window_dimensions),
        strides,
        List.duplicate(1, tuple_size(op_shape(arg))),
        window_dilations,
        padding_config,
        [expr_to_typespec(ans)]
      )

    result
  end

  defp to_operator(
         :window_scatter_max,
         [%Value{} = arg, %Value{} = source, %Value{} = init_value, window_dimensions, opts],
         %{type: type} = ans,
         _state
       ) do
    padding_config = opts[:padding]
    strides = opts[:strides]

    arg = to_type(arg, type)
    source = to_type(source, type)
    init_value = to_type(init_value, type)

    Value.select_and_scatter(
      arg,
      source,
      init_value,
      :gt,
      Tuple.to_list(window_dimensions),
      strides,
      padding_config,
      expr_to_typespec(ans)
    )
  end

  defp to_operator(
         :window_scatter_min,
         [%Value{} = arg, %Value{} = source, %Value{} = init_value, window_dimensions, opts],
         %{type: type} = ans,
         _state
       ) do
    padding_config = opts[:padding]
    strides = opts[:strides]

    arg = to_type(arg, type)
    source = to_type(source, type)
    init_value = to_type(init_value, type)

    Value.select_and_scatter(
      arg,
      source,
      init_value,
      :lt,
      Tuple.to_list(window_dimensions),
      strides,
      padding_config,
      expr_to_typespec(ans)
    )
  end

  defp to_operator(:indexed_add, [%Value{} | _] = tensors, out, _state) do
    mlir_scatter(tensors, out, :add)
  end

  defp to_operator(:indexed_put, [%Value{} | _] = tensors, out, _state) do
    mlir_scatter(tensors, out, :put)
  end

  defp to_operator(op, [arg, opts], ans, state) when op in [:argmax, :argmin] do
    apply(EXLA.Lib, op, [state.builder, arg, ans.type, opts])
  end

  defp to_operator(:clip, [%Value{} = operand, %Value{} = min, %Value{} = max], ans, _state) do
    min = to_type(min, ans.type)
    max = to_type(max, ans.type)
    operand = to_type(operand, ans.type)

    Value.clamp(operand, min, max, expr_to_typespec(ans))
  end

  defp to_operator(:slice, [%Value{} = tensor, start_indices, lengths, strides], ans, _state) do
    all_static? = Enum.all?(start_indices, &is_integer/1)

    if all_static? do
      limit_indices = Enum.zip_with(start_indices, lengths, fn i, len -> i + len end)
      Value.slice(tensor, start_indices, limit_indices, strides, expr_to_typespec(ans))
    else
      sample = Enum.find(start_indices, &(not is_integer(&1)))

      type =
        Enum.reduce(start_indices, op_type(sample), fn
          index, acc when is_integer(index) -> acc
          value, acc -> merge_type(op_type(value), acc)
        end)

      start_indices = Enum.map(start_indices, &to_type(&1, type))
      zeros = List.duplicate(0, tuple_size(ans.shape))

      typespec = Typespec.tensor(ans.type, List.to_tuple(lengths))
      slice = Value.dynamic_slice(tensor, start_indices, lengths, typespec)

      if Enum.all?(strides, &(&1 == 1)) do
        slice
      else
        Value.slice(slice, zeros, lengths, strides, expr_to_typespec(ans))
      end
    end
  end

  defp to_operator(:put_slice, [%Value{} = tensor, start_indices, slice], ans, _state) do
    tensor = to_type(tensor, ans.type)
    slice = to_type(slice, ans.type)
    Value.dynamic_update_slice(tensor, slice, start_indices, expr_to_typespec(ans))
  end

  defp to_operator(:gather, [%Value{} = tensor, indices, opts], ans, _state) do
    axes = Keyword.fetch!(opts, :axes)
    tensor_shape = op_shape(tensor)
    tensor_rank = tuple_size(tensor_shape)
    tensor_axes = axes_for_rank(tensor_rank)
    indices_rank = tuple_size(op_shape(indices))
    index_vector_dim = indices_rank - 1

    slice_sizes =
      for i <- tensor_axes do
        if i in axes, do: 1, else: elem(tensor_shape, i)
      end

    batch_size = tensor_rank - length(axes)
    offset_size = indices_rank - length(axes)
    offset_dims = count_up(batch_size, offset_size)

    Value.gather(
      tensor,
      indices,
      index_vector_dim,
      slice_sizes,
      offset_dims,
      axes,
      axes,
      expr_to_typespec(ans)
    )
  end

  defp to_operator(:reverse, [%Value{} = tensor, axes], ans, _state) do
    Value.reverse(tensor, axes, expr_to_typespec(ans))
  end

  defp to_operator(:concatenate, [[%Value{} | _rest] = tensors, axis], ans, _state) do
    tensors = Enum.map(tensors, &to_type(&1, ans.type))
    Value.concatenate(tensors, axis, expr_to_typespec(ans))
  end

  defp to_operator(:stack, [[%Value{} | _rest] = tensors, axis], ans, _state) do
    reshape_typespec = Typespec.tensor(ans.type, put_elem(ans.shape, axis, 1))
    tensors = Enum.map(tensors, &(&1 |> to_type(ans.type) |> Value.reshape(reshape_typespec)))
    Value.concatenate(tensors, axis, expr_to_typespec(ans))
  end

  defp to_operator(:sort, [%Value{} = tensor, opts], ans, state) do
    dimension = opts[:axis]

    op =
      case opts[:direction] do
        :asc -> :less
        :desc -> :greater
      end

    arg_typespec = Typespec.tensor(ans.type, {})
    arg_typespecs = [arg_typespec, arg_typespec]

    comp = sort_computation(op, ans.type, arg_typespecs, state)

    Value.sort([tensor], comp, dimension, opts[:stable] == true, [expr_to_typespec(ans)]) |> hd()
  end

  defp to_operator(:argsort, [tensor, opts], ans, state) do
    dimension = opts[:axis]
    stable = opts[:stable] == true

    op =
      case opts[:direction] do
        :asc -> :less
        :desc -> :greater
      end

    type = op_type(tensor)

    value_typespec = Typespec.tensor(type, {})
    idx_typespec = Typespec.tensor(ans.type, {})
    arg_typespecs = [value_typespec, value_typespec, idx_typespec, idx_typespec]

    comp = sort_computation(op, type, arg_typespecs, state)

    EXLA.Lib.argsort(state.builder, tensor, dimension, stable, comp, ans.type)
  end

  defp fft(exla_op, [%Value{} = tensor, opts], %{type: type} = ans, state) do
    n = opts[:length]
    axis = opts[:axis]
    output_type = Nx.Type.to_complex(type)
    tensor = to_type(tensor, output_type)

    shape = op_shape(tensor)
    m = elem(shape, axis)

    tensor = fft_pad_or_slice(tensor, m, n, axis, shape, output_type, state)

    last_axis = tuple_size(shape) - 1

    if axis != last_axis do
      permutation =
        Enum.map(0..last_axis, fn
          ^axis -> last_axis
          ^last_axis -> axis
          ax -> ax
        end)

      {transposed_shape, _} = Nx.Shape.transpose(ans.shape, permutation, ans.names)
      transposed_typespec = Typespec.tensor(ans.type, transposed_shape)

      tensor
      |> Value.transpose(permutation, transposed_typespec)
      |> exla_op.([n], transposed_typespec)
      |> Value.transpose(permutation, expr_to_typespec(ans))
    else
      exla_op.(tensor, [n], expr_to_typespec(ans))
    end
  end

  defp fft2(exla_op, [%Value{} = tensor, opts], %{type: type} = ans, state) do
    [l1, l2] = lengths = opts[:lengths]
    [ax1, ax2] = axes = opts[:axes]
    output_type = Nx.Type.to_complex(type)
    tensor = to_type(tensor, output_type)

    shape = op_shape(tensor)
    m1 = elem(shape, ax1)
    m2 = elem(shape, ax2)

    tensor = fft_pad_or_slice(tensor, m1, l1, ax1, shape, output_type, state)
    tensor = fft_pad_or_slice(tensor, m2, l2, ax2, op_shape(tensor), output_type, state)

    last_axis = tuple_size(shape) - 1
    penultimate_axis = last_axis - 1
    last_axes = [penultimate_axis, last_axis]

    if axes != last_axes do
      permutation =
        Enum.map(0..last_axis, fn
          ^ax1 -> penultimate_axis
          ^penultimate_axis -> ax1
          ^ax2 -> last_axis
          ^last_axis -> ax2
          ax -> ax
        end)

      {transposed_shape, _} = Nx.Shape.transpose(ans.shape, permutation, ans.names)
      transposed_typespec = Typespec.tensor(ans.type, transposed_shape)

      tensor
      |> Value.transpose(permutation, transposed_typespec)
      |> exla_op.(lengths, transposed_typespec)
      |> Value.transpose(permutation, expr_to_typespec(ans))
    else
      exla_op.(tensor, lengths, expr_to_typespec(ans))
    end
  end

  defp fft_pad_or_slice(tensor, m, n, axis, shape, output_type, state) do
    cond do
      m == n ->
        tensor

      m > n ->
        lengths =
          shape
          |> Tuple.insert_at(axis + 1, n)
          |> Tuple.delete_at(axis)
          |> Tuple.to_list()

        starts = List.duplicate(0, tuple_size(shape))
        strides = List.duplicate(1, tuple_size(shape))

        limit_indices = Enum.zip_with(starts, lengths, fn i, len -> i + len end)

        {_, shape} = Nx.Shape.slice(shape, starts, limit_indices, strides)
        typespec = Typespec.tensor(output_type, shape)
        Value.slice(tensor, starts, limit_indices, strides, typespec)

      m < n ->
        zero =
          Value.constant(state.builder, [Complex.new(0)], Typespec.tensor(output_type, {}))

        padding_config =
          {0, 0, 0}
          |> List.duplicate(tuple_size(shape))
          |> List.replace_at(axis, {0, n - m, 0})

        shape = Nx.Shape.pad(shape, padding_config)
        typespec = Typespec.tensor(output_type, shape)
        Value.pad(tensor, zero, padding_config, typespec)
    end
  end

  defp mlir_scatter([target, indices, updates, opts], %{type: type} = ans, kind)
       when kind in [:add, :put] do
    target = to_type(target, type)
    updates = to_type(updates, type)
    update_rank = updates |> op_shape() |> tuple_size()
    update_axes = tl(axes_for_rank(update_rank))
    index_axes = Keyword.fetch!(opts, :axes)

    Value.scatter(
      target,
      indices,
      updates,
      kind,
      1,
      update_axes,
      index_axes,
      index_axes,
      expr_to_typespec(ans)
    )
  end

  ## Cache and hook helpers helpers

  defp no_token_cache(),
    do: %{__MODULE__ => Outfeed.empty()}

  defp new_cache(outfeed),
    do: %{__MODULE__ => outfeed}

  defp merge_outfeed(%{__MODULE__ => outfeed} = cache, %{__MODULE__ => new_outfeed}),
    do: %{cache | __MODULE__ => Outfeed.with_token(new_outfeed, outfeed.token)}

  defp reset_token(%{__MODULE__ => outfeed}, token),
    do: %{__MODULE__ => Outfeed.with_token(outfeed, token)}

  defp update_token(%{__MODULE__ => outfeed} = cache, token),
    do: %{cache | __MODULE__ => Outfeed.with_token(outfeed, token)}

  defp get_token(%{__MODULE__ => outfeed}), do: outfeed.token

  defp get_outfeed(%{__MODULE__ => value}), do: value

  defp put_outfeed(cache, outfeed), do: %{cache | __MODULE__ => outfeed}

  ## Computation helpers

  defp sort_computation(operator, type, arg_typespecs, %{
         builder: %EXLA.MLIR.Function{} = function
       }) do
    {region, [lhs, rhs | _]} = Function.push_region(function, arg_typespecs)

    typespec = Typespec.tensor({:pred, 8}, {})

    {lhs, rhs} =
      if Nx.Type.integer?(type) do
        {lhs, rhs}
      else
        {sort_computation_canonicalize_float(lhs), sort_computation_canonicalize_float(rhs)}
      end

    op = apply(Value, operator, [lhs, rhs, typespec, [total_order: true]])

    Value.return(function, [op])
    Function.pop_region(function)
    region
  end

  defp sort_computation_canonicalize_float(%Value{function: func} = op) do
    # Standardize the representation of NaNs (-NaN, NaN) and zeros (-0, 0).
    # See https://github.com/google/jax/blob/e81c82605f0e1813080cfe1037d043b27b38291d/jax/_src/lax/lax.py#L4248-L4253

    op_typespec = Value.get_typespec(op)

    zero = Value.constant(func, [0], Typespec.to_shape(op_typespec, {}))
    zeros = Value.constant(func, [0], op_typespec)
    nans = Value.constant(func, [:nan], op_typespec)

    pred_typespec = Typespec.tensor({:pred, 8}, {})
    op = Value.select(Value.equal(op, zero, pred_typespec), zeros, op, op_typespec)
    Value.select(Value.is_nan(op, pred_typespec), nans, op, op_typespec)
  end

  defp op_computation(
         op,
         arg_typespecs,
         %{builder: %EXLA.MLIR.Function{} = builder},
         prepare_args
       ) do
    {region, args} = Function.push_region(builder, arg_typespecs)
    op = apply(Value, op, prepare_args.(args) ++ [hd(arg_typespecs)])
    Value.return(builder, [op])
    Function.pop_region(builder)
    region
  end

  defp fun_computation(args, expr, type, %{builder: %Function{} = function} = state) do
    arg_typespecs =
      Enum.map(args, fn %{type: type, shape: shape} -> Typespec.tensor(type, shape) end)

    {region, mlir_args} = Function.push_region(function, arg_typespecs)

    arg_params = Enum.zip(args, mlir_args)

    params = Enum.flat_map(arg_params, &computation_arg_param/1)

    state = %{
      state
      | builder: function,
        params: Map.new(params),
        scope_ids: Tree.scope_ids(expr)
    }

    {res, _} = recur_composite(expr, state, no_token_cache())
    Value.return(function, Enum.map(res, &to_type(&1, type)))
    Function.pop_region(function)
    region
  end

  defp mlir_while_computation(expr, initial, type, state, cache) do
    arg_typespecs = Enum.map(List.flatten(initial), &Value.get_typespec/1)

    {region, args} = Function.push_region(state.builder, arg_typespecs)

    outer_token = get_token(cache)

    {inner_token, arg_params} =
      if outer_token do
        [arg_token | arg_params] = args
        {arg_token, arg_params}
      else
        {nil, args}
      end

    params = Enum.with_index(arg_params, &{&2, &1})

    state = %{
      state
      | params: Map.new(params),
        scope_ids: Tree.scope_ids(expr)
    }

    expr =
      if type == {:pred, 8} and expr.type == {:u, 8} do
        %{expr | type: {:pred, 8}}
      else
        expr
      end

    {res, comp_cache} = recur_composite(expr, & &1, state, reset_token(cache, inner_token))

    res =
      if type == :with_token do
        if outer_token do
          [get_token(comp_cache) | List.flatten(res)]
        else
          List.flatten(res)
        end
      else
        Enum.map(res, &to_type(&1, type))
      end

    Value.return(state.builder, res)
    Function.pop_region(state.builder)

    {region, merge_outfeed(cache, comp_cache)}
  end

  defp optional_computation(name, args, expr, %{builder: %Function{}} = state, cache) do
    %Function{module: module, name: name} = subbuilder(state.builder, name)

    arg_typespecs = Enum.map(args, &Value.get_typespec/1)
    out_typespecs = container_to_typespecs(expr)

    outer_token = get_token(cache)
    token_typespec = Typespec.token()

    {arg_typespecs, out_typespecs} =
      if outer_token do
        {[token_typespec | arg_typespecs], [token_typespec | out_typespecs]}
      else
        {arg_typespecs, out_typespecs}
      end

    function = EXLA.MLIR.Module.add_function(module, name, arg_typespecs, out_typespecs)
    args = EXLA.MLIR.Function.get_arguments(function)

    {inner_token, args} =
      if outer_token do
        [arg_token | args] = args
        {arg_token, args}
      else
        {nil, args}
      end

    params = Enum.with_index(args, fn param, i -> {i, param} end)

    state = %{
      state
      | builder: function,
        params: Map.new(params),
        scope_ids: Tree.scope_ids(expr)
    }

    {res, comp_cache} = recur_composite(expr, state, reset_token(cache, inner_token))

    if outer_token do
      Value.func_return(function, [get_token(comp_cache) | List.flatten(res)])
    else
      Value.func_return(function, List.flatten(res))
    end

    {function, merge_outfeed(cache, comp_cache)}
  end

  # The cache is built on top of call args because we need to handle pred/u8.
  defp computation_key(op, args) do
    keys =
      Enum.map(args, fn
        %Value{} = op ->
          %Typespec{type: type, shape: shape} = Value.get_typespec(op)
          {shape, type}

        opts ->
          opts
      end)

    {op, keys}
  end

  defp computation_arg_param({tuple, params}) when is_tuple(tuple) and is_list(params) do
    tuple
    |> Tuple.to_list()
    |> Enum.zip(params)
    |> Enum.flat_map(&computation_arg_param/1)
  end

  defp computation_arg_param({%T{data: %Expr{op: :parameter, args: [pos]}}, [param]}) do
    [{pos, param}]
  end

  defp computation_arg_param({%T{data: %Expr{op: :parameter, args: [pos]}}, param}) do
    [{pos, param}]
  end

  defp recur_composite(composite, state, cache) do
    recur_composite(composite, & &1, state, cache)
  end

  defp recur_composite(tuple, transform, state, cache) when is_tuple(tuple) do
    list = Tuple.to_list(tuple)

    if expr = full_tuple(list) do
      recur_composite(expr, transform, state, cache)
    else
      Enum.map_reduce(list, cache, &recur_composite(&1, transform, state, &2))
    end
  end

  defp recur_composite(%Value{} = op, transform, _state, cache) do
    {[transform.(op)], cache}
  end

  defp recur_composite(expr, transform, state, cache) do
    {op, cache} = recur_operator(expr, state, cache)

    result =
      if is_list(op) do
        Enum.map(op, transform)
      else
        [transform.(op)]
      end

    {result, cache}
  end

  # If each element of the tuple is just a reference to the parent expression,
  # discard the tuple elements and return the parent expression.
  defp full_tuple(list) do
    with [%T{data: %Expr{op: :elem, args: [%T{data: %Expr{id: id} = expr}, 0]}} | _] <- list,
         true <- full_tuple?(list, 0, id) do
      expr
    else
      _ -> nil
    end
  end

  defp full_tuple?([arg | args], index, id) do
    case arg do
      %T{data: %Expr{op: :elem, args: [%T{data: %Expr{id: ^id}, shape: shape}, ^index]}} ->
        if shape == {:tuple, index} do
          true
        else
          full_tuple?(args, index + 1, id)
        end

      _ ->
        false
    end
  end

  # We got until the end without traversing the whole tuple
  defp full_tuple?([], _index, _id), do: false

  ## Aggregation

  defp to_aggregate(op, type, shape, %Value{} = arg, initial, opts, state) do
    arg = to_type(arg, type)

    acc =
      case initial do
        %Value{} = initial ->
          initial

        initial when is_number(initial) ->
          Value.constant(state.builder, [initial], Typespec.tensor(type, {}))
      end

    args = [Typespec.tensor(type, {}), Typespec.tensor(type, {})]
    comp = op_computation(op, args, state, &Enum.reverse/1)

    keep_axes = opts[:keep_axes]
    reduce_axes = reduce_axes(arg, opts[:axes])

    typespec = Typespec.tensor(type, remove_axes(op_shape(arg), reduce_axes))
    [result] = Value.reduce(comp, [acc], [arg], reduce_axes, [typespec])

    if keep_axes do
      Value.reshape(result, Typespec.tensor(type, shape))
    else
      result
    end
  end

  defp to_window_aggregate(op, %{type: type} = ans, arg, initial, window_dimensions, opts, state) do
    arg = to_type(arg, type)

    acc =
      case initial do
        %Value{} = initial ->
          initial

        initial when is_number(initial) ->
          Value.constant(state.builder, [initial], Typespec.tensor(type, {}))
      end

    args = [Typespec.tensor(type, {}), Typespec.tensor(type, {})]
    # We reverse the argument order because :nan + :infinity
    # returns :nan but :infinity + :nan returns :infinity.
    # So we want to keep the current value as first argument
    # to preserve such properties.
    comp = op_computation(op, args, state, &Enum.reverse/1)

    strides = opts[:strides]
    padding = opts[:padding]
    window_dilations = opts[:window_dilations]

    [result] =
      Value.window_reduce(
        comp,
        [acc],
        [arg],
        Tuple.to_list(window_dimensions),
        strides,
        List.duplicate(1, tuple_size(op_shape(arg))),
        window_dilations,
        padding,
        [expr_to_typespec(ans)]
      )

    result
  end

  ## Cond

  defp to_if(pred, on_true, on_false, %{builder: %Function{}} = state, cache) do
    {pred_op, cache} = recur_operator(pred, state, cache) |> unwrap_single_tensor!()

    true_ids = Tree.scope_ids(on_true)
    false_ids = Tree.scope_ids(on_false)

    cache = recur_shared_ids(on_true, false_ids, state, cache)
    cache = recur_shared_ids(on_false, true_ids, state, cache)

    out_typespecs = container_to_typespecs(on_true)

    outer_token = get_token(cache)

    result_typespecs =
      if outer_token do
        [Typespec.token() | out_typespecs]
      else
        out_typespecs
      end

    {true_computation, cache} = to_mlir_if_branch(on_true, true_ids, state, cache)
    {false_computation, cache} = to_mlir_if_branch(on_false, false_ids, state, cache)
    if_results = Value.if_op(pred_op, true_computation, false_computation, result_typespecs)

    if outer_token do
      [token | results] = if_results
      {wrap_tuple_result(results, on_true), update_token(cache, token)}
    else
      {wrap_tuple_result(if_results, on_true), cache}
    end
  end

  defp recur_shared_ids(
         expr,
         other_ids,
         %{scope_ids: ids} = state,
         cache
       ) do
    {_, cache} =
      Composite.reduce(expr, {%{}, cache}, fn node, acc ->
        do_recur_shared_ids(node, state, acc, {ids, other_ids})
      end)

    cache
  end

  defp shared?(_id, :parameter, _args, _shared_ids),
    do: true

  # We never pass reference to tuples around, only through their elements,
  # so if a tuple is in a predicate, then it all must be in a predicate.
  defp shared?(_id, :elem, [%T{data: %Expr{id: tuple_id}}, _pos], {parent_ids, sibling_ids})
       when is_map_key(parent_ids, tuple_id) or is_map_key(sibling_ids, tuple_id),
       do: true

  defp shared?(id, _op, _args, {parent_ids, sibling_ids}),
    do: is_map_key(parent_ids, id) or is_map_key(sibling_ids, id)

  defp do_recur_shared_ids(
         %T{data: %Expr{id: id, op: op, args: args}} = expr,
         state,
         {visited, cache},
         shared_ids
       ) do
    cond do
      Map.has_key?(visited, id) ->
        {visited, cache}

      op == :constant or shared?(id, op, args, shared_ids) ->
        {_, cache} = recur_operator(expr, state, cache)
        {Map.put(visited, id, true), cache}

      true ->
        {_, {visited, cache}} =
          Tree.apply_args(
            expr,
            :scope,
            {visited, cache},
            &{&1, do_recur_shared_ids(&1, state, &2, shared_ids)}
          )

        {Map.put(visited, id, true), cache}
    end
  end

  defp to_mlir_if_branch(expr, current_ids, state, cache) do
    {region, []} = Function.push_region(state.builder, [])

    comp_state = %{state | scope_ids: current_ids}

    {res, res_cache} = recur_composite(expr, & &1, comp_state, cache)

    if token = get_token(cache) do
      Value.return(state.builder, [token | List.flatten(res)])
    else
      Value.return(state.builder, List.flatten(res))
    end

    Function.pop_region(state.builder)

    {region, merge_outfeed(cache, res_cache)}
  end

  ## Axes helpers

  defp broadcast_axes(left, right) do
    {min, max} = if left <= right, do: {left, right}, else: {right, left}
    min_size = tuple_size(min)
    max_size = tuple_size(max)

    # To reproduce Nx broadcast, we simply match the lower dimensions to the highest ones.
    count_up(min_size, max_size - min_size)
  end

  defp reduce_axes(op, axes) do
    if axes do
      Enum.sort(axes)
    else
      Nx.axes(op_shape(op))
    end
  end

  defp count_up(0, _n), do: []
  defp count_up(i, n), do: [n | count_up(i - 1, n + 1)]

  defp axes_for_rank(0), do: []

  defp axes_for_rank(rank) do
    Enum.to_list(0..(rank - 1))
  end

  ## Op Helpers

  defp op_type(%Value{} = op), do: Value.get_typespec(op).type

  defp op_shape(%Value{} = op), do: Value.get_typespec(op).shape

  defp to_type(%Value{} = op, type) do
    typespec = Value.get_typespec(op)

    if typespec.type == type do
      op
    else
      Value.convert(op, Typespec.to_type(typespec, type))
    end
  end

  defp merge_type({:pred, 8}, {:pred, 8}), do: {:pred, 8}
  defp merge_type(left, right), do: Nx.Type.merge(to_nx_type(left), to_nx_type(right))

  defp to_nx_type({:pred, 8}), do: {:u, 8}
  defp to_nx_type(type), do: type

  defp to_constant(%EXLA.MLIR.Function{} = function, constant, type) do
    Value.constant(function, [constant], Typespec.tensor(type, {}))
  end

  defp subbuilder(%EXLA.MLIR.Function{name: name} = function, description) do
    suffix = System.unique_integer([:positive])
    %{function | name: name <> "_" <> description <> "_" <> Integer.to_string(suffix)}
  end

  # Helpers

  defp apply_mlir_broadcasted_bin_op(op, out, left, right) do
    left_typespec = Value.get_typespec(left)
    right_typespec = Value.get_typespec(right)
    left_dims = broadcast_axes(left_typespec.shape, out.shape)
    right_dims = broadcast_axes(right_typespec.shape, out.shape)

    type = merge_type(left_typespec.type, right_typespec.type)
    type = merge_type(type, out.type)

    left = to_type(left, type)

    left =
      if left_typespec.shape == out.shape do
        left
      else
        Value.broadcast_in_dim(left, left_dims, Typespec.tensor(type, out.shape))
      end

    right = to_type(right, type)

    right =
      if right_typespec.shape == out.shape do
        right
      else
        Value.broadcast_in_dim(right, right_dims, Typespec.tensor(type, out.shape))
      end

    Value
    |> apply(op, [left, right, Typespec.tensor(type, out.shape)])
    |> to_type(out.type)
  end

  defp to_mlir_logical(%Value{} = value) do
    to_type(value, {:pred, 8})
  end

  defp container_to_typespecs(container) do
    [container]
    |> Nx.Defn.Composite.flatten_list()
    |> Enum.flat_map(fn
      %Nx.Tensor{type: {:tuple, _}, data: %{args: values}} ->
        Enum.flat_map(values, &container_to_typespecs/1)

      t ->
        [Typespec.tensor(t.type, t.shape)]
    end)
  end

  defp wrap_tuple_result(list, template) when is_tuple(template) do
    list
  end

  defp wrap_tuple_result(list, %Nx.Tensor{type: {:tuple, _}}) do
    list
  end

  defp wrap_tuple_result([value], _), do: value

  defp unwrap_single_tensor!({[%Value{} = op], cache}), do: {op, cache}
  defp unwrap_single_tensor!({%Value{} = op, cache}), do: {op, cache}

  defp remove_axes(shape, axes) do
    axes
    |> Enum.reverse()
    |> Enum.reduce(shape, &Tuple.delete_at(&2, &1))
  end

  defp expr_to_typespec(expr) do
    Typespec.tensor(expr.type, expr.shape)
  end
end
