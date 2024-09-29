defmodule Nx.Defn.ShardingCompiler do
  alias Nx.Defn.Composite
  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  @behaviour Nx.Defn.Compiler

  defstruct [:tensor_sharding, :input_tensor_shardings]

  defmodule Shard do
    import Inspect.Algebra
    defstruct [:id, :axis, :input_id, :start, :length, :parents]

    def inspect(%__MODULE__{start: start, length: length}, inspect_opts)
        when is_nil(start) or is_nil(length) do
      color("Shard<>", :map, inspect_opts)
    end

    def inspect(%__MODULE__{id: id, axis: axis, start: start, length: length}, inspect_opts) do
      single_line = inspect_opts.custom_options[:single_line]
      print_axis = inspect_opts.custom_options[:print_axis]

      range_doc = "#{start}..#{start + length - 1}"

      if single_line do
        concat([
          color("Shard<", :map, inspect_opts),
          if(print_axis && axis, do: "#{axis}: ", else: ""),
          range_doc,
          " (#{inspect(id)})",
          color(">", :map, inspect_opts)
        ])
      else
        concat([
          color("Shard<", :map, inspect_opts),
          nest(
            concat([
              line(),
              if(print_axis && axis, do: "#{axis}: ", else: ""),
              range_doc,
              line(),
              "(#{inspect(id)})"
            ]),
            2
          ),
          line(),
          color(">", :map, inspect_opts)
        ])
      end
    end

    defimpl Inspect do
      def inspect(mod, opts), do: Shard.inspect(mod, opts)
    end
  end

  defmodule TensorSharding do
    defstruct [:shards, :id]

    @doc """
    config is a map of axis index or name -> slices
    """
    def new(tensor, config, opts \\ [])

    def new(_tensor, %TensorSharding{} = config, opts) do
      if input_id = opts[:input_id] do
        shards =
          Map.new(config.shards, fn {axis, shards} ->
            {axis, Enum.map(shards, &Map.put(&1, :input_id, &1.input_id || input_id))}
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
          {axis_or_name, slices} ->
            axis =
              if is_atom(axis_or_name) do
                Nx.axis_index(tensor, axis_or_name)
              else
                axis_or_name
              end

            shards =
              Enum.map(slices, fn start..finish//1 ->
                id = make_ref()

                %Shard{
                  id: id,
                  axis: axis,
                  start: start,
                  length: finish - start + 1,
                  input_id: input_id,
                  parents: MapSet.new()
                }
              end)

            {axis, shards}
        end)

      shards =
        Enum.reduce(Nx.axes(tensor), shards, fn axis, shards ->
          if Map.has_key?(shards, axis) do
            shards
          else
            id = make_ref()

            shard = %Shard{
              id: id,
              axis: axis,
              start: 0,
              length: Nx.axis_size(tensor, axis),
              input_id: input_id,
              parents: MapSet.new()
            }

            Map.put(shards, axis, [shard])
          end
        end)

      %TensorSharding{shards: shards, id: make_ref()}
    end

    def inspect(%__MODULE__{shards: shards}, opts) do
      import Inspect.Algebra

      shards =
        shards
        |> Enum.sort_by(fn {axis, _} -> axis end)
        |> Enum.map(fn {axis, shards} ->
          axis_name = opts.custom_options[:axis_names][axis] || axis

          shard_doc =
            shards
            |> Enum.flat_map(fn %Shard{} = shard ->
              opts = put_in(opts.custom_options[:single_line], true)
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

      if shards != :doc_nil do
        concat([
          concat([
            color("TensorSharding<", :map, opts),
            line(),
            shards
          ])
          |> nest(2),
          line(),
          color(">", :map, opts)
        ])
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

    # args_with_slices =
    #   Enum.with_index(args, fn lazy_input, idx ->
    #     input = lazy_input.()
    #     input_sharding = input_tensor_shardings[idx]

    #     shards =
    #       for {axis, %Shard{id: id}} <- input_sharding.shards do
    #         {axis, output_config.shard_interactions[id]}
    #       end

    #     dbg({idx, shards})

    #     # Here, shards is a list containing all possible slicing combinations.
    #     # We need to obtain the proper cartesian product of all these combinations.
    #     slices =
    #       cartesian_product(
    #         Enum.map(shards, fn {axis, shard} ->
    #           slices =
    #             if shard.slices == [] do
    #               [0..(Nx.axis_size(input, axis) - 1)//1]
    #             else
    #               shard.slices
    #             end

    #           {axis, slices}
    #         end)
    #       )

    #     [input, slices]
    #   end)

    # dbg(args_with_slices)

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
            parameter_ids_to_index: %{},
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

    state = put_in(state.params.parameter_ids_to_index, res.data.id, i)
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
    {tensor_sharding, state} = bin_op_tensor_sharding(arg0, arg1, ans, state)
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
         %T{shape: out_shape} = out,
         %{tensor_shardings: tensor_shardings} = state
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

    out_axes = Nx.axes(out_shape)

    left_shards_list =
      Enum.map(out_axes, fn axis ->
        left_shards[axis] || []
      end)

    right_shards_list =
      Enum.map(out_axes, fn axis ->
        right_shards[axis] || []
      end)

    dbg(out_axes)

    result =
      Enum.reduce(out_axes, {left_shards_list, right_shards_list, [], [], []}, fn
        axis,
        {[left_shards | left_acc], [right_shards | right_acc], out_acc, left_out_acc,
         right_out_acc} ->
          {out_shards, left_shards, right_shards} =
            case {left_shards, right_shards} do
              {[], shards} ->
                out_shards =
                  Enum.map(shards, fn shard ->
                    %Shard{
                      id: make_ref(),
                      axis: axis,
                      start: shard.start,
                      length: shard.length,
                      input_id: nil,
                      parents: MapSet.put(shard.parents, shard.id)
                    }
                  end)

                {out_shards, [], shards}

              {shards, []} ->
                out_shards =
                  Enum.map(shards, fn shard ->
                    %Shard{
                      id: make_ref(),
                      axis: axis,
                      start: shard.start,
                      length: shard.length,
                      input_id: nil,
                      parents: MapSet.put(shard.parents, shard.id)
                    }
                  end)

                {out_shards, shards, []}

              {[], []} ->
                {[], [], []}

              {left, right} ->
                # We can resolve a conflict iff one of the shards is sharding over the full axis.
                resolve_sharding_broadcast(
                  left,
                  left_axis_sizes[axis],
                  right,
                  right_axis_sizes[axis]
                )
            end

          {
            left_acc,
            right_acc,
            [out_shards | out_acc],
            [left_shards | left_out_acc],
            [right_shards | right_out_acc]
          }
      end)

    {[], [], out_shards_reverse, left_shards_reverse, right_shards_reverse} = result

    out_shards = Enum.reverse(out_shards_reverse)
    left_shards_broadcasted = Enum.reverse(left_shards_reverse)
    right_shards_broadcasted = Enum.reverse(right_shards_reverse)

    left_shards = Enum.map(left_broadcast_axes, &Enum.fetch!(left_shards_broadcasted, &1))
    right_shards = Enum.map(right_broadcast_axes, &Enum.fetch!(right_shards_broadcasted, &1))

    dbg(left_shards)
    dbg(right_shards)
    dbg(out_shards)

    raise "asdf"
  end

  defp resolve_sharding_broadcast(
         [%Shard{start: 0, length: 1, id: id, parents: parents}] = left,
         1,
         right,
         _right_axis_size
       ) do
    out_shards =
      Enum.map(right, fn shard ->
        %{
          shard
          | id: make_ref(),
            start: shard.start,
            length: shard.length,
            parents: shard.parents |> MapSet.union(parents) |> MapSet.put(id)
        }
      end)

    {out_shards, left, right}
  end

  defp resolve_sharding_broadcast(
         left,
         _left_axis_size,
         [%Shard{start: 0, length: 1, id: id, parents: parents}] = right,
         1
       ) do
    out_shards =
      Enum.map(left, fn shard ->
        %{
          shard
          | id: make_ref(),
            start: shard.start,
            length: shard.length,
            parents: shard.parents |> MapSet.union(parents) |> MapSet.put(id)
        }
      end)

    {out_shards, left, right}
  end

  defp resolve_sharding_broadcast(left, left_axis_size, right, right_axis_size) do
    dbg({left, right})
    # if we have a single axis-length shard on either end, we can re-slice it according to
    # the slicing on the other side
    case {left, right} do
      {[
         %Shard{
           id: id,
           axis: axis,
           input_id: input_id,
           start: 0,
           length: ^left_axis_size,
           parents: parents
         }
       ], right_shards} ->
        left_shards =
          Enum.map(right_shards, fn shard ->
            %Shard{
              id: make_ref(),
              axis: axis,
              start: shard.start,
              length: shard.length,
              input_id: input_id,
              parents: MapSet.put(parents, id)
            }
          end)

        out_shards =
          Enum.zip_with(left_shards, right_shards, fn left, right ->
            %{
              left
              | id: make_ref(),
                input_id: nil,
                parents: MapSet.union(left.parents, right.parents)
            }
          end)

        {out_shards, left_shards, right_shards}

      {left_shards,
       [
         %Shard{
           id: id,
           axis: axis,
           input_id: input_id,
           start: 0,
           length: ^right_axis_size,
           parents: parents
         }
       ]} ->
        right_shards =
          Enum.map(left_shards, fn shard ->
            %Shard{
              id: make_ref(),
              axis: axis,
              start: shard.start,
              length: shard.length,
              input_id: input_id,
              parents: MapSet.put(parents, id)
            }
          end)

        out_shards =
          Enum.zip_with(left_shards, right_shards, fn left, right ->
            %{
              right
              | id: make_ref(),
                input_id: nil,
                parents: MapSet.union(left.parents, right.parents)
            }
          end)

        {out_shards, left_shards, right_shards}

      {l, r} ->
        dbg({l, r})
        raise "incompatible sharding"
    end
    |> dbg()
  end
end
