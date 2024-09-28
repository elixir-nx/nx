defmodule Nx.Defn.ShardingCompiler do
  alias Nx.Defn.Composite
  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  @behaviour Nx.Defn.Compiler

  defstruct [:tensor_sharding, :input_tensor_shardings]

  defmodule Shard do
    @derive Inspect
    defstruct [:slices, :id, :axis, :input_id]

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
    def new(tensor, config, opts \\ [])

    def new(_tensor, %TensorSharding{} = config, opts) do
      if input_id = opts[:input_id] do
        shards =
          Map.new(config.shards, fn {axis, shard} ->
            {axis, Map.put(shard, :input_id, shard.input_id || input_id)}
          end)

        %TensorSharding{config | shards: shards}
      else
        config
      end
    end

    def new(tensor, config, opts) when is_map(config) do
      input_id = opts[:input_id]

      shards =
        Map.new(config, fn
          {name, slices} when is_atom(name) ->
            axis = Nx.axis_index(tensor, name)
            id = make_ref()

            {axis, %Shard{id: id, axis: axis, slices: slices, input_id: input_id}}

          {axis, slices} ->
            id = make_ref()

            {axis, %Shard{id: id, axis: axis, slices: slices, input_id: input_id}}
        end)

      shards =
        Enum.reduce(Nx.axes(tensor), shards, fn axis, shards ->
          if Map.has_key?(shards, axis) do
            shards
          else
            id = make_ref()
            axis_size = Nx.axis_size(tensor, axis)
            shard = %Shard{id: id, axis: axis, slices: [], input_id: input_id}
            Map.put(shards, axis, shard)
          end
        end)

      shard_interactions = %{}

      %TensorSharding{shard_interactions: shard_interactions, shards: shards, id: make_ref()}
    end

    def inspect(%__MODULE__{shards: shards}, opts) do
      import Inspect.Algebra

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

  def shard(tensor, config, opts \\ []) do
    tensor_sharding = TensorSharding.new(tensor, config, opts)

    data =
      case tensor.data do
        %__MODULE__{tensor_sharding: old_config} = data ->
          used_shards =
            Map.merge(old_config.used_shards, config.shards, fn k, old, new ->
              unless old == new do
                raise "Unexpected sharding conflict"
              end
            end)

          %__MODULE__{data | tensor_sharding: %{tensor_sharding | used_shards: used_shards}}

        _ ->
          %__MODULE__{tensor_sharding: tensor_sharding}
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

    args_with_slices =
      Enum.with_index(args, fn lazy_input, idx ->
        input = lazy_input.()
        input_sharding = input_tensor_shardings[idx]

        shards =
          for {axis, %Shard{id: id}} <- input_sharding.shards do
            {axis, output_config.shard_interactions[id]}
          end

        dbg({idx, shards})

        # Here, shards is a list containing all possible slicing combinations.
        # We need to obtain the proper cartesian product of all these combinations.
        slices =
          cartesian_product(
            Enum.map(shards, fn {axis, shard} ->
              slices =
                if shard.slices == [] do
                  [0..(Nx.axis_size(input, axis) - 1)//1]
                else
                  shard.slices
                end

              {axis, slices}
            end)
          )

        [input, slices]
      end)

    dbg(args_with_slices)

    raise "asdf"
    # sharding_compiler = opts[:sharding_compiler]
    # sharding_compiler_options = opts[:sharding_compiler_options]

    # result =
    #   for instance <- sharded_args do
    #     args = Enum.map(instance, &elem(&1, 1))
    #     starts = Enum.map(instance, &elem(&1, 0))

    #     vars =
    #       Enum.with_index(args, fn arg, idx ->
    #         arg
    #         # |> Expr.tensor()
    #         |> Expr.parameter(:root, idx)
    #       end)

    #     compiled_fun = sharding_compiler.__compile__(key, vars, fun, sharding_compiler_options)

    #     {
    #       [List.to_tuple(args)],
    #       fn [args] ->
    #         [res] =
    #           compiled_fun.([
    #             Enum.map(Tuple.to_list(args), fn arg ->
    #               fn -> arg end
    #             end)
    #           ])

    #         res
    #       end,
    #       fn result, acc ->
    #         Nx.put_slice(acc, hd(starts), result)
    #       end
    #     }
    #   end

    # output_holder = Nx.iota(shape, type: type)

    # [{output_holder, result}]
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

      [put_in(container.data.input_tensor_shardings, state_shard_configs)]
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
    res = tensor |> Nx.devectorize() |> shard(config, input_id: id)
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
    tensor_sharding = bin_op_tensor_sharding(arg0, arg1, ans)
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

    left_shards =
      Enum.map(Nx.axes(left_shape), fn axis ->
        {axis, left_config.shards[axis]}
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
        {axis, right_config.shards[axis]}
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

    output_shard_and_interactions =
      Enum.map(Nx.axes(out_shape), fn axis ->
        left = left_shards[axis]
        right = right_shards[axis]

        dbg({left, right})

        {shard, left_slices, right_slices} =
          case {left, right} do
            {%Shard{slices: []}, right} ->
              {right, right.slices, right.slices}

            {left, %Shard{slices: []}} ->
              {left, left.slices, left.slices}

            {nil, right} ->
              {right, right.slices, right.slices}

            {left, nil} ->
              {left, left.slices, left.slices}

            {shard, shard} ->
              {shard, shard.slices, shard.slices}

            {left, right} ->
              raise "Sharding conflict on output axis #{axis}"
          end

        dbg({axis, left_slices, left_shape, left_axis_sizes})
        out_axis_size = elem(out_shape, axis)

        left_slices =
          if left_axis_sizes[axis] == 1 do
            # List.duplicate(0..0//1, out_axis_size)
            []
          else
            left_slices
          end

        dbg(left_slices)

        right_slices =
          if right_axis_sizes[axis] == 1 do
            # List.duplicate(0..0//1, out_axis_size)
            []
          else
            right_slices
          end

        {
          %Shard{shard | axis: axis},
          left && %Shard{left | slices: left_slices},
          right && %Shard{right | slices: right_slices}
        }
      end)

    interactions = Map.merge(left_config.shard_interactions, right_config.shard_interactions)

    for {shard, left_shard, right_shard} <- output_shard_and_interactions,
        reduce: %TensorSharding{shards: %{}, shard_interactions: interactions} do
      %TensorSharding{
        shards: output_shards,
        shard_interactions: interactions
      } = acc ->
        interactions =
          if left_shard do
            Map.put(interactions, left_shard.id, left_shard)
          else
            interactions
          end

        interactions =
          if right_shard do
            Map.put(interactions, right_shard.id, right_shard)
          else
            interactions
          end

        %TensorSharding{
          acc
          | shards: Map.put(output_shards, shard.axis, shard),
            shard_interactions: interactions
        }
    end
  end
end
