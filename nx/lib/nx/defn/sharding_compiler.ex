defmodule Nx.Defn.ShardingCompiler do
  alias Nx.Defn.Composite
  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  alias Nx.Defn.ShardingCompiler.Shard

  @behaviour Nx.Defn.Compiler

  defstruct [:id, :shards, :input_tensor_shardings, :parameter_ids_to_index]

  @impl true
  def inspect(
        %T{data: %__MODULE__{shards: shards}},
        inspect_opts
      ) do
    import Inspect.Algebra

    shards
    |> Enum.sort_by(fn {axis, _} -> axis end)
    |> Enum.map(fn {axis, shards} ->
      axis_name = inspect_opts.custom_options[:axis_names][axis] || axis

      shard_doc =
        shards
        |> Enum.flat_map(fn %Shard{} = shard ->
          opts = put_in(inspect_opts.custom_options[:single_line], true)
          opts = put_in(opts.custom_options[:print_axis], false)

          [
            line(),
            Shard.inspect(shard, opts)
          ]
        end)
        |> concat()
        |> nest(2)

      concat([
        "#{axis_name}: ",
        shard_doc
      ])
    end)
    |> Enum.intersperse(line())
    |> concat()
  end

  def init(opts), do: opts

  def put_shards(tensor, shards, opts \\ []) do
    shards =
      if input_id = opts[:input_id] do
        Map.new(shards, fn {axis, shards} ->
          {axis, Enum.map(shards, &%Shard{&1 | input_id: input_id})}
        end)
      else
        shards
      end

    data = %__MODULE__{id: make_ref(), shards: shards}
    %{tensor | data: data}
  end

  def shard_from_config(tensor, config, opts \\ []) do
    shards = Shard.from_config(tensor, config, opts)
    put_shards(tensor, shards, opts)
  end

  @impl true
  def __jit__(key, vars, fun, args, opts) do
    opts =
      Keyword.validate!(opts, [
        :sharding_config,
        sharding_compiler: Nx.Defn.Evaluator,
        sharding_compiler_options: []
      ])

    [args] = args

    [
      %T{
        shape: shape,
        type: type,
        data: %__MODULE__{
          shards: output_shards,
          parameter_ids_to_index: parameter_ids_to_index
        }
      }
    ] =
      __compile__(key, vars, fun, sharding_config: opts[:sharding_config]).([args])

    data_sections =
      output_shards |> Enum.sort_by(fn {axis, _} -> axis end) |> cartesian_product()

    # Find the parents for each data section
    # Group by inputs
    # For each input, sort the shards by axis
    # For each axis, find the minimum start and the maximum end (we need to test for slicing inside the code as well)
    # it might be the case where an axis is not present in the mapping. This means we need the full axis.

    result =
      for section <- data_sections do
        shards_by_input_id =
          section
          |> Enum.flat_map(fn {_axis, shard} ->
            get_root_parents(shard)
          end)
          |> Enum.group_by(fn shard -> shard.input_id end)

        inputs_by_index =
          parameter_ids_to_index
          |> Enum.sort_by(fn {_id, idx} -> idx end)
          |> Enum.map(fn {id, idx} -> {id, Enum.fetch!(args, idx)} end)

        sliced_inputs =
          for {input_id, input_fn} <- inputs_by_index do
            input = input_fn.()
            shards = shards_by_input_id[input_id]
            shards_by_axis = Enum.group_by(shards, & &1.axis)

            {_, _, starts_reverse, lengths_reverse} =
              Enum.reduce(Tuple.to_list(input.shape), {shards_by_axis, 0, [], []}, fn axis_size,
                                                                                      {shards_by_axis,
                                                                                       axis,
                                                                                       starts,
                                                                                       lengths} ->
                {shards, shards_by_axis} = Map.pop(shards_by_axis, axis)

                {starts, lengths} =
                  if shards do
                    min_start = Enum.min(Enum.map(shards, & &1.start))
                    max_end = Enum.max(Enum.map(shards, &(&1.start + &1.length - 1)))

                    starts = [min_start | starts]
                    lengths = [max_end - min_start + 1 | lengths]
                    {starts, lengths}
                  else
                    starts = [0 | starts]
                    lengths = [axis_size | lengths]
                    {starts, lengths}
                  end

                {shards_by_axis, axis + 1, starts, lengths}
              end)

            starts = Enum.reverse(starts_reverse)
            lengths = Enum.reverse(lengths_reverse)

            Nx.slice(input, starts, lengths)
          end

        {out_starts, []} =
          Enum.map_reduce(0..(tuple_size(shape) - 1)//1, section, fn
            axis, [{axis, shard} | shards] ->
              {shard.start, shards}

            _axis, shards ->
              {0, shards}
          end)

        caster_fn = fn result, acc ->
          Nx.put_slice(acc, out_starts, result)
        end

        sharding_compiler = opts[:sharding_compiler]
        sharding_compiler_options = opts[:sharding_compiler_options]

        vars =
          Enum.with_index(sliced_inputs, fn arg, idx ->
            arg
            |> Expr.parameter(:root, idx)
          end)

        compiled_fun =
          sharding_compiler.__compile__({key, section}, vars, fun, sharding_compiler_options)

        shard_fn = fn [args] ->
          [res] =
            compiled_fun.([
              Enum.map(Tuple.to_list(args), fn arg ->
                fn -> arg end
              end)
            ])

          res
        end

        {[List.to_tuple(sliced_inputs)], shard_fn, caster_fn}
      end

    output_holder = Nx.iota(shape, type: type)
    [{output_holder, result}]
  end

  defp cartesian_product([{axis, first} | rest]) do
    for x <- first, y <- cartesian_product(rest), do: [{axis, x} | y]
  end

  defp cartesian_product([]), do: [[]]

  @impl true
  def __compile__(_key, vars, fun, opts) do
    opts = Keyword.validate!(opts, [:sharding_config])
    expr = fun.(vars)

    state_shard_configs =
      opts
      |> Keyword.get(:sharding_config)
      |> Enum.zip_with(vars, fn config, var ->
        Shard.from_config(var, config)
      end)
      |> Enum.with_index(fn x, idx -> {idx, x} end)
      |> Map.new()

    fn [params] ->
      state_params = params |> Enum.with_index(fn x, idx -> {idx, x} end) |> Map.new()

      {container, {_cache, state}} =
        composite_eval(
          expr,
          %{
            gc?: opts[:garbage_collect] || false,
            params: state_params,
            tensor_shardings: state_shard_configs,
            parameter_ids_to_index: %{},
            contracted_tensor_shardings: %{}
          },
          %{}
        )

      container = put_in(container.data.input_tensor_shardings, state_shard_configs)
      container = put_in(container.data.parameter_ids_to_index, state.parameter_ids_to_index)

      [container]
    end
  end

  defp composite_eval(expr, state, cache) do
    Composite.traverse(expr, {cache, state}, &eval/2)
  end

  defp eval(%T{data: %Expr{op: :tensor, args: [t]}}, {cache, state}) do
    {shard_from_config(t, %{}), {cache, state}}
  end

  defp eval(%T{data: %Expr{op: :constant, args: [_constant]}} = ans, {cache, state}) do
    {shard_from_config(ans, %{}), {cache, state}}
  end

  defp eval(%T{data: %Expr{op: :metadata, args: [expr, _meta]}}, {cache, state}) do
    composite_eval(expr, state, cache)
  end

  defp eval(%T{data: %Expr{id: id, op: op}} = ans, {cache, state}) do
    case cache do
      %{^id => res} ->
        {res, {cache, state}}

      _ ->
        eval_apply(op, ans, {cache, state})
    end
  end

  defp eval(other, {cache, state}) do
    {other, {cache, state}}
  end

  defp eval_apply(:parameter, %T{data: %Expr{id: id, args: [i]}}, {cache, state}) do
    %T{} = tensor = Map.fetch!(state.params, i).()
    shards = Map.fetch!(state.tensor_shardings, i)
    res = tensor |> Nx.devectorize() |> put_shards(shards, input_id: id)

    state = put_in(state.parameter_ids_to_index[id], i)
    {res, {Map.put(cache, id, res), state}}
  end

  defp eval_apply(:elem, %T{data: %Expr{id: id, args: [tuple, i]}}, {cache, state}) do
    {tuple, cache} = composite_eval(tuple, state, cache)
    res = elem(tuple, i)
    {res, {Map.put(cache, id, res), state}}
  end

  defp eval_apply(op, %T{data: %Expr{id: id}} = ans, {cache, state}) do
    {args, {cache, state}} = Nx.Defn.Tree.apply_args(ans, {cache, state}, &eval/2)

    {res, state} = apply_op(op, ans, args, state)
    {res, {Map.put(cache, id, res), state}}
  end

  @unary_ops [:exp, :expm1, :log, :log1p, :sigmoid, :cos, :sin, :tan, :cosh, :sinh, :tanh] ++
               [:acosh, :asinh, :atanh, :sqrt, :rsqrt, :cbrt, :negate, :sign, :abs, :bitwise_not] ++
               [:is_nan, :is_infinity] ++
               [:conjugate, :population_count, :count_leading_zeros, :floor, :ceil, :round] ++
               [:erf, :erfc, :erf_inv, :acos, :asin, :atan, :bitcast, :real, :imag]

  defp apply_op(op, ans, [arg], state) when op in @unary_ops do
    {put_shards(ans, arg.data.shards), state}
  end

  @binary_ops [
                :add,
                :subtract,
                :multiply,
                :pow,
                :remainder,
                :divide,
                :atan2,
                :min,
                :max,
                :quotient
              ] ++
                [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift] ++
                [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal] ++
                [:logical_and, :logical_or, :logical_xor]

  defp apply_op(op, ans, [arg0, arg1], state) when op in @binary_ops do
    {shards, state} = bin_op_tensor_sharding(arg0, arg1, ans, state)
    {put_shards(ans, shards), state}
  end

  # defp apply_op(:reshape, ans, [arg], state) do
  # TODO: support reshape: we need to find a way to map the shards to the new shape
  # This can be achieved in a few cases, but not all.
  # Examples:
  #   [2, 3] -> [3, 2] works iff both axes are fully sharded
  #   [2, 6] -> [2, 1, 3, 2] works if we can fit the shards of axis 1 into [3, 2]
  # end

  defp apply_op(:squeeze, ans, [arg, squeeze_axes], state) do
    shards = Enum.sort_by(arg.data.shards, fn {axis, _} -> axis end)

    {[], _, out_shards} =
      Enum.reduce(shards, {Enum.sort(squeeze_axes, :asc), 0, %{}}, fn
        {axis, _shards}, {[axis | squeeze_axes], num_squeezed_axes, out_shards} ->
          {squeeze_axes, num_squeezed_axes + 1, out_shards}

        {axis, shards}, {squeeze_axes, num_squeezed_axes, out_shards} ->
          {squeeze_axes, num_squeezed_axes, Map.put(out_shards, axis - num_squeezed_axes, shards)}
      end)

    {put_shards(ans, out_shards), state}
  end

  defp apply_op(:transpose, ans, [arg, axes], state) do
    shards = arg.data.shards

    out_shards =
      axes
      |> Enum.with_index(fn out_axis, in_axis ->
        out_shards = make_child_shards(Map.fetch!(shards, in_axis), out_axis)
        {out_axis, out_shards}
      end)
      |> Map.new()

    {put_shards(ans, out_shards), state}
  end

  defp apply_op(:dot, ans, [t0, c0, b0, t1, c1, b1], state) do
    # TO-DO: Make it so that sharding over a contracting axis becomes
    # an all-reduce op instead of simply ignoring the sharding
    left_sharding =
      Enum.reduce(c0, t0.data.shards, fn axis, acc ->
        Map.put(acc, axis, [
          %Shard{
            id: make_ref(),
            axis: axis,
            start: 0,
            length: elem(t0.shape, axis),
            parents: []
          }
        ])
      end)

    right_sharding =
      Enum.reduce(c1, t1.data.shards, fn axis, acc ->
        Map.put(acc, axis, [
          %Shard{
            id: make_ref(),
            axis: axis,
            start: 0,
            length: elem(t1.shape, axis),
            parents: []
          }
        ])
      end)

    offset = length(b0)

    batch_shards =
      Enum.zip_with([b0, b1, 0..(offset - 1)], fn left_axis, right_axis, axis ->
        left_shards = Map.fetch!(left_sharding, left_axis)
        right_shards = Map.fetch!(right_sharding, right_axis)
        resolve_sharding_broadcast(axis, left_shards, false, right_shards, false)
      end)

    out_shards_left =
      Enum.with_index(Nx.axes(t0) -- c0, fn axis, idx ->
        {idx + offset,
         left_sharding
         |> Map.fetch!(axis)
         |> make_child_shards(idx + offset)}
      end)

    offset = offset + length(out_shards_left)

    out_shards_right =
      Enum.with_index(Nx.axes(t1) -- c1, fn axis, idx ->
        {idx + offset,
         right_sharding
         |> Map.fetch!(axis)
         |> make_child_shards(idx + offset)}
      end)

    out_shards = Map.new(batch_shards ++ out_shards_left ++ out_shards_right)
    {put_shards(ans, out_shards), state}
  end

  defp apply_op(op, _ans, _args, _state) do
    raise "Unsupported op: #{op}"
  end

  defp bin_op_tensor_sharding(
         %T{
           shape: left_shape,
           data: %__MODULE__{
             shards: left_config
           }
         },
         %T{
           shape: right_shape,
           data: %__MODULE__{
             shards: right_config
           }
         },
         %T{shape: out_shape},
         state
       ) do
    left_broadcast_axes = Nx.Shape.broadcast_axes(left_shape, out_shape)
    right_broadcast_axes = Nx.Shape.broadcast_axes(right_shape, out_shape)

    left_shards =
      Enum.map(Nx.axes(left_shape), fn axis ->
        {axis, left_config[axis]}
      end)

    left_shards =
      Enum.zip_with(left_broadcast_axes, left_shards, fn new_axis, {_id, shard} ->
        {new_axis, shard}
      end)
      |> Map.new()

    left_axis_sizes =
      Enum.with_index(left_broadcast_axes, fn out_axis, in_axis ->
        {out_axis, elem(left_shape, in_axis)}
      end)
      |> Map.new()

    right_shards =
      Enum.map(Nx.axes(right_shape), fn axis ->
        {axis, right_config[axis]}
      end)

    right_axis_sizes =
      Enum.with_index(right_broadcast_axes, fn out_axis, in_axis ->
        {out_axis, elem(right_shape, in_axis)}
      end)
      |> Map.new()

    right_shards =
      Enum.zip_with(right_broadcast_axes, right_shards, fn new_axis, {_id, shard} ->
        {new_axis, shard}
      end)
      |> Map.new()

    out_axes = Nx.axes(out_shape)

    left_shards_list =
      Enum.map(out_axes, fn axis ->
        left_shards[axis] || []
      end)

    right_shards_list =
      Enum.map(out_axes, fn axis ->
        right_shards[axis] || []
      end)

    result =
      Enum.reduce(out_axes, {left_shards_list, right_shards_list, []}, fn
        axis, {[left_shards | left_acc], [right_shards | right_acc], out_acc} ->
          out_shards =
            case {left_shards, right_shards} do
              {[], []} ->
                []

              {[], shards} ->
                make_child_shards(shards, axis)

              {shards, []} ->
                make_child_shards(shards, axis)

              {left, right} ->
                # If we are dealing with a broadcast axis on either tensor, we can
                # map the single shard to all shards on the other tensor.

                left_size = left_axis_sizes[axis]
                right_size = right_axis_sizes[axis]

                left_is_broadcasting = left_size != right_size and left_size == 1
                right_is_broadcasting = left_size != right_size and right_size == 1

                resolve_sharding_broadcast(
                  axis,
                  left,
                  left_is_broadcasting,
                  right,
                  right_is_broadcasting
                )
            end

          {
            left_acc,
            right_acc,
            [out_shards | out_acc]
          }
      end)

    {[], [], out_reverse_shards} = result

    out_shards =
      out_reverse_shards
      |> Enum.reverse()
      |> Enum.with_index()
      |> Map.new(fn {shards, idx} -> {idx, shards} end)

    {out_shards, state}
  end

  defp get_root_parents(shard, acc \\ [])

  defp get_root_parents(%Shard{parents: []} = shard, acc), do: List.flatten([shard | acc])

  defp get_root_parents(%Shard{parents: parents}, acc) do
    Enum.reduce(parents, acc, &get_root_parents/2)
    |> List.flatten()
  end

  defp resolve_sharding_broadcast(axis, [left_shard], true, right_shards, false) do
    # We have a single shard on the left that we'll map onto the right shards.
    make_child_shards(right_shards, axis, [left_shard])
  end

  defp resolve_sharding_broadcast(axis, left_shards, false, [right_shard], true) do
    # We have a single shard on the right that we'll map onto the left shards.
    make_child_shards(left_shards, axis, [right_shard])
  end

  defp resolve_sharding_broadcast(axis, left_shards, false, right_shards, false) do
    # We have a shard on both sides. We need to determine the intersection of the two.
    # This is fine only if all shards are equal
    {reverse_out_shards, all_shards_match} =
      Enum.zip_reduce(left_shards, right_shards, {[], true}, fn left,
                                                                right,
                                                                {out_acc, match_acc} ->
        match_acc = match_acc and left.start == right.start and left.length == right.length

        out_acc = make_child_shards([left], axis, [right]) ++ out_acc

        {out_acc, match_acc}
      end)

    if not all_shards_match do
      raise "incompatible sharding"
    end

    Enum.reverse(reverse_out_shards)
  end

  defp make_child_shards(shards, axis, extra_parents \\ []) do
    Enum.map(shards, fn shard ->
      %Shard{
        id: make_ref(),
        axis: axis,
        start: shard.start,
        length: shard.length,
        input_id: nil,
        parents: [shard | extra_parents]
      }
    end)
  end
end
