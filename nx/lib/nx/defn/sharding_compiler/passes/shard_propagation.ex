defmodule Nx.Defn.ShardingCompiler.Passes.ShardPropagation do
  alias Nx.Defn.Composite
  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  alias Nx.Defn.ShardingCompiler.Shard

  defstruct [:id, :shards, :input_tensor_shardings, :parameter_ids_to_index, :expr]

  def traverse(expr, tensor_shardings) do
    {container, {cache, state}} =
      composite_traverse(
        expr,
        %{
          tensor_shardings: tensor_shardings,
          parameter_ids_to_index: %{},
          expr_shards: %{}
        },
        %{}
      )

    container = put_in(container.data.input_tensor_shardings, tensor_shardings)
    container = put_in(container.data.parameter_ids_to_index, state.parameter_ids_to_index)

    {container, cache, state}
  end

  defp put_shards(tensor, shards, opts \\ []) do
    shards =
      if input_id = opts[:input_id] do
        Map.new(shards, fn {axis, shards} ->
          {axis, Enum.map(shards, &%Shard{&1 | input_id: input_id})}
        end)
      else
        shards
      end

    data = %__MODULE__{id: make_ref(), shards: shards, expr: tensor.data}
    %{tensor | data: data}
  end

  def shard_from_config(tensor, config, opts \\ []) do
    shards = Shard.from_config(tensor, config, opts)
    put_shards(tensor, shards, opts)
  end

  defp composite_traverse(expr, state, cache) do
    Composite.traverse(expr, {cache, state}, &eval/2)
  end

  defp eval(%T{data: %Expr{op: :tensor, args: [t]}}, {cache, state}) do
    config =
      t
      |> Nx.axes()
      |> Map.new(fn axis ->
        {axis, elem(t.shape, axis)}
      end)

    expr = shard_from_config(t, config)
    state = put_in(state.expr_shards[expr.data.id], expr.data)
    {expr, {cache, state}}
  end

  defp eval(%T{data: %Expr{op: :constant, args: [_constant]}} = ans, {cache, state}) do
    expr = shard_from_config(ans, %{})
    state = put_in(state.expr_shards[expr.data.id], expr.data)
    {expr, {cache, state}}
  end

  defp eval(%T{data: %Expr{op: :metadata, args: [expr, _meta]}}, {cache, state}) do
    composite_traverse(expr, state, cache)
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

  defp eval_apply(:parameter, %T{data: %Expr{id: id, args: [i]}} = expr, {cache, state}) do
    shards = Map.fetch!(state.tensor_shardings, i)
    res = put_shards(expr, shards, input_id: id)

    state = put_in(state.parameter_ids_to_index[id], i)
    state = put_in(state.expr_shards[id], res.data)

    {res, {Map.put(cache, id, res), state}}
  end

  defp eval_apply(:elem, %T{data: %Expr{id: id, args: [tuple, i]}}, {cache, state}) do
    {tuple, cache} = composite_traverse(tuple, state, cache)
    res = elem(tuple, i)
    state = put_in(state.expr_shards[id], res.data)
    {res, {Map.put(cache, id, res), state}}
  end

  defp eval_apply(op, %T{data: %Expr{id: id}} = ans, {cache, state}) do
    {args, {cache, state}} = Nx.Defn.Tree.apply_args(ans, {cache, state}, &eval/2)

    {res, state} = apply_op(op, ans, args, state)
    state = put_in(state.expr_shards[id], res.data)
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
      |> Enum.with_index(fn in_axis, out_axis ->
        out_shards = make_child_shards(Map.fetch!(shards, in_axis), out_axis)
        {out_axis, out_shards}
      end)
      |> Map.new()

    {put_shards(ans, out_shards), state}
  end

  defp apply_op(:dot, ans, [t0, c0, b0, t1, c1, b1], state) do
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
end
