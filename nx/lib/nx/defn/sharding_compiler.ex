defmodule Nx.Defn.ShardingCompiler do
  alias Nx.Defn.Composite
  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  @behaviour Nx.Defn.Compiler

  defstruct [:tensor_sharding, :input_tensor_shardings]

  defmodule Shard do
    @derive Inspect
    defstruct [:slices, :id, :axis]

    def inspect(%__MODULE__{id: id, axis: axis, slices: slices}, _inspect_opts) do
      doc =
        slices
        |> Enum.map(fn start..slice_end//1 -> "#{start}..#{slice_end}" end)
        |> Enum.intersperse(", ")
        |> Inspect.Algebra.concat()

      Inspect.Algebra.concat([
        "(#{inspect(id)}) ",
        "[",
        doc,
        "]"
      ])
    end
  end

  defmodule TensorSharding do
    defstruct [:shards, :id, :shard_interactions]

    @doc """
    config is a map of axis index or name -> slices
    """
    def new(tensor, %TensorSharding{} = config) do
      config
    end

    def new(tensor, config) when is_map(config) do
      shards =
        Map.new(config, fn
          {name, slices} when is_atom(name) ->
            axis = Nx.axis_index(tensor, name)
            id = make_ref()
            {id, %Shard{id: id, axis: axis, slices: slices}}

          {axis, slices} ->
            id = make_ref()
            {id, %Shard{id: id, axis: axis, slices: slices}}
        end)

      ids = Map.keys(shards)

      shard_interactions =
        for left <- ids, right <- ids, into: %{} do
          {{left, right}, nil}
        end

      %TensorSharding{shard_interactions: shard_interactions, shards: shards, id: make_ref()}
    end

    def inspect(%__MODULE__{shards: shards}, opts) do
      import Inspect.Algebra

      dbg(shards)

      shards =
        shards
        |> Enum.sort_by(fn {_id, %{axis: axis}} -> axis end)
        |> Enum.map(fn {_id, %Shard{axis: axis} = shard} ->
          axis_name = opts.custom_options[:axis_names][axis] || axis

          concat([
            "#{axis_name}: ",
            Shard.inspect(shard, opts)
          ])
        end)
        |> Enum.intersperse(line())
        |> concat()

      if shards != :doc_nil do
        nest(
          concat([
            color("TensorSharding<", :map, opts),
            line(),
            shards,
            line(),
            color(">", :map, opts)
          ]),
          2
        )
      else
        string("TensorSharding<>")
      end
    end

    defimpl Inspect do
      def inspect(mod, opts), do: TensorSharding.inspect(mod, opts)
    end
  end

  @impl true
  def inspect(
        %T{
          shape: shape,
          names: names,
          data: %__MODULE__{tensor_sharding: tensor_sharding}
        },
        inspect_opts
      ) do
    import Inspect.Algebra

    TensorSharding.inspect(tensor_sharding, custom_options: [axis_names: names])
  end

  def init(opts), do: opts

  def shard(tensor, config) do
    tensor_sharding = TensorSharding.new(tensor, config)

    data =
      case tensor.data do
        %__MODULE__{tensor_sharding: old_config} = data ->
          used_shards =
            Map.merge(old_config.used_shards, config.shards, fn k, old, new ->
              unless old == new do
                raise "Unexpected sharding conflict"
              end
            end)

          %__MODULE__{data | tensor_sharding: %{config | used_shards: used_shards}}

        _ ->
          %__MODULE__{tensor_sharding: config}
      end

    %{tensor | data: data}
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
          input_tensor_shardings: input_tensor_shardings,
          tensor_sharding: output_config
        }
      }
    ] =
      __compile__(key, vars, fun, sharding_config: opts[:sharding_config]).([args])

    slices =
      Enum.with_index(output_config, fn
        nil, axis -> {axis, [..]}
        %Shard{slices: slices}, axis -> {axis, slices}
      end)

    dbg(input_tensor_shardings)

    shards = cartesian_product(slices)

    sharded_args =
      for slices <- shards do
        for lazy_input <- args do
          input = lazy_input.()

          slices = Enum.sort(slices)

          starts =
            for {_axis, start.._//1} <- slices do
              start
            end

          lengths =
            for {axis, start..finish//1} <- slices do
              axis_size = Nx.axis_size(input, axis)

              cond do
                axis_size == 1 ->
                  1

                finish == -1 ->
                  axis_size - start

                true ->
                  finish - start + 1
              end
            end

          {starts, Nx.slice(input, starts, lengths)}
        end
      end

    sharding_compiler = opts[:sharding_compiler]
    sharding_compiler_options = opts[:sharding_compiler_options]

    result =
      for instance <- sharded_args do
        args = Enum.map(instance, &elem(&1, 1))
        starts = Enum.map(instance, &elem(&1, 0))

        vars =
          Enum.with_index(args, fn arg, idx ->
            arg
            # |> Expr.tensor()
            |> Expr.parameter(:root, idx)
          end)

        compiled_fun = sharding_compiler.__compile__(key, vars, fun, sharding_compiler_options)

        {
          [List.to_tuple(args)],
          fn [args] ->
            [res] =
              compiled_fun.([
                Enum.map(Tuple.to_list(args), fn arg ->
                  fn -> arg end
                end)
              ])

            res
          end,
          fn result, acc ->
            Nx.put_slice(acc, hd(starts), result)
          end
        }
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
        TensorSharding.new(var, config)
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
            contracted_tensor_shardings: %{}
          },
          %{}
        )

      [container]
    end
  end

  defp composite_eval(expr, state, cache) do
    Composite.traverse(expr, {cache, state}, &eval/2)
  end

  defp eval(%T{data: %Expr{op: :tensor, args: [t]}}, {cache, state}) do
    {t, {cache, state}}
  end

  defp eval(%T{data: %Expr{op: :constant, args: [_constant]}} = ans, {cache, state}) do
    {ans, {cache, state}}
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
    config = Map.fetch!(state.tensor_shardings, i)
    res = tensor |> Nx.devectorize() |> shard(config)
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

  # defp apply_op(:dot, ans, [arg0, contract0, [], arg1, contract1, []], state) do
  #   {config0, config1, contracted_tensor_shardings} =
  #     Enum.zip_reduce(
  #       contract0,
  #       contract1,
  #       {arg0.data.tensor_sharding, arg1.data.tensor_sharding, state.contracted_tensor_shardings},
  #       fn
  #         axis0, axis1, {config0, config1, contracted_tensor_shardings} ->
  #           entry0 = Enum.fetch!(config0, axis0)
  #           entry1 = Enum.fetch!(config1, axis1)

  #           unless is_nil(entry0) or is_nil(entry1) do
  #             raise "incompatible sharding"
  #           end

  #           dbg(entry0)

  #           contracted_tensor_shardings =
  #             if entry0 do
  #               Map.put(contracted_tensor_shardings, entry0.id, entry0)
  #             else
  #               contracted_tensor_shardings
  #             end

  #           contracted_tensor_shardings =
  #             if entry1 do
  #               Map.put(contracted_tensor_shardings, entry1.id, entry1)
  #             else
  #               contracted_tensor_shardings
  #             end

  #           {
  #             List.replace_at(config0, axis0, :delete),
  #             List.replace_at(config1, axis1, :delete),
  #             contracted_tensor_shardings
  #           }
  #       end
  #     )

  #   config = Enum.reject(config0, &(&1 == :delete)) ++ Enum.reject(config1, &(&1 == :delete))

  #   {shard(ans, config),
  #    Map.put(state, :contracted_tensor_shardings, contracted_tensor_shardings)}
  # end

  @unary_ops [:exp, :expm1, :log, :log1p, :sigmoid, :cos, :sin, :tan, :cosh, :sinh, :tanh] ++
               [:acosh, :asinh, :atanh, :sqrt, :rsqrt, :cbrt, :negate, :sign, :abs, :bitwise_not] ++
               [:is_nan, :is_infinity] ++
               [:conjugate, :population_count, :count_leading_zeros, :floor, :ceil, :round] ++
               [:erf, :erfc, :erf_inv, :acos, :asin, :atan, :bitcast, :real, :imag]

  defp apply_op(op, ans, [arg], state) when op in @unary_ops do
    {shard(ans, arg.data.tensor_sharding), state}
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
    dbg({arg0.shape, arg1.shape})
    tensor_sharding = bin_op_tensor_sharding(arg0, arg1, ans)

    dbg(tensor_sharding)
    {shard(ans, tensor_sharding), state}
  end

  defp bin_op_tensor_sharding(
         %T{
           shape: left_shape,
           data: %__MODULE__{
             tensor_sharding: left_config
           }
         },
         %T{
           shape: right_shape,
           data: %__MODULE__{
             tensor_sharding: right_config
           }
         },
         %T{shape: out_shape}
       ) do
    left_broadcast_axes = Nx.Shape.broadcast_axes(left_shape, out_shape)
    right_broadcast_axes = Nx.Shape.broadcast_axes(right_shape, out_shape)

    left_shards_by_axis = Map.new(left_config.shards, fn {_id, shard} -> {shard.axis, shard} end)

    left_shards =
      Enum.map(Nx.axes(left_shape), fn axis ->
        {axis, left_shards_by_axis[axis]}
      end)

    left_shards =
      Enum.zip_with(left_broadcast_axes, left_shards, fn new_axis, {_id, shard} ->
        {new_axis, shard}
      end)
      |> Map.new()

    right_shards_by_axis =
      Map.new(right_config.shards, fn {_id, shard} -> {shard.axis, shard} end)

    right_shards =
      Enum.map(Nx.axes(right_shape), fn axis ->
        {axis, right_shards_by_axis[axis]}
      end)

    right_shards =
      Enum.zip_with(right_broadcast_axes, right_shards, fn new_axis, {_id, shard} ->
        {new_axis, shard}
      end)
      |> Map.new()

    output_shards =
      Enum.map(Nx.axes(out_shape), fn axis ->
        left = left_shards[axis]
        right = right_shards[axis]

        if left && right do
          dbg({axis, left_broadcast_axes, right_broadcast_axes})
          dbg({left, right})
          raise "Sharding conflict on output axis #{axis}"
        end

        shard = left || right
        %{shard | axis: axis}
      end)

    shard_interactions =
      for %{id: left} <- output_shards,
          %{id: right} <- output_shards,
          into: Map.merge(left_config.shard_interactions, right_config.shard_interactions) do
        {{left, right}, nil}
      end

    %TensorSharding{
      shards: Map.new(output_shards, &{&1.id, &1}),
      shard_interactions: shard_interactions
    }
  end
end
