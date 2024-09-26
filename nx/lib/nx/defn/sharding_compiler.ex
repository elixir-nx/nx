defmodule Nx.Defn.ShardingCompiler do
  alias Nx.Defn.Composite
  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  @behaviour Nx.Defn.Compiler

  defstruct [:sharding_config, :input_sharding_configs, :contracted_sharding_configs]

  defmodule AxisConfig do
    @derive Inspect
    defstruct [:slices, :id, :input_axis]

    def inspect(%__MODULE__{id: id, input_axis: input_axis, slices: slices}, _inspect_opts) do
      doc =
        slices
        |> Enum.map(fn start..slice_end//1 -> "#{start}..#{slice_end}" end)
        |> Enum.intersperse(", ")
        |> Inspect.Algebra.concat()

      Inspect.Algebra.concat([
        "(#{input_axis}, #{inspect(id)}) ",
        "[",
        doc,
        "]"
      ])
    end
  end

  @impl true
  def inspect(
        %T{names: names, data: %__MODULE__{sharding_config: sharding_config}},
        inspect_opts
      ) do
    import Inspect.Algebra

    shards =
      sharding_config
      |> Enum.with_index(fn
        nil, index ->
          axis_name = Enum.at(names, index) || index
          string("#{axis_name}: [0..-2]")

        axis_config, index ->
          axis_name = Enum.at(names, index) || index

          concat([
            "#{axis_name}: ",
            AxisConfig.inspect(axis_config, inspect_opts)
          ])
      end)
      |> Enum.intersperse(line())
      |> concat()

    if shards != :doc_nil do
      nest(
        concat([
          "Shards",
          line(),
          shards,
          line()
        ]),
        2
      )
    else
      string("Shards<>")
    end
  end

  @doc """
  config is a map of axis index or name -> slices
  """
  def build_sharding_config(_tensor, config) when is_list(config) do
    config
  end

  def build_sharding_config(tensor, config) when is_map(config) do
    config =
      Map.new(config, fn
        {name, v} when is_atom(name) ->
          {Nx.axis_index(tensor, name), v}

        item ->
          item
      end)

    Enum.map(Nx.axes(tensor), fn axis ->
      if slices = Map.get(config, axis) do
        %AxisConfig{id: make_ref(), input_axis: axis, slices: slices}
      end
    end)
  end

  def init(opts), do: opts

  def shard(tensor, config) do
    sharding_config = build_sharding_config(tensor, config)
    put_in(tensor.data, %__MODULE__{sharding_config: sharding_config})
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
          input_sharding_configs: input_sharding_configs,
          contracted_sharding_configs: contracted_sharding_configs,
          sharding_config: output_config
        }
      }
    ] =
      __compile__(key, vars, fun, opts).([args])

    dbg({output_config, contracted_sharding_configs})

    slices =
      Enum.with_index(output_config, fn
        nil, axis -> {axis, [..]}
        %AxisConfig{slices: slices}, axis -> {axis, slices}
      end)

    dbg(input_sharding_configs)

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
    expr = fun.(vars)

    state_shard_configs =
      opts
      |> Keyword.get(:sharding_config)
      |> Enum.zip_with(vars, fn config, var ->
        build_sharding_config(var, config)
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
            sharding_configs: state_shard_configs,
            contracted_sharding_configs: %{}
          },
          %{}
        )

      container =
        put_in(container.data.contracted_sharding_configs, state.contracted_sharding_configs)

      container = put_in(container.data.input_sharding_configs, state_shard_configs)
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

  defp eval(%T{data: %Expr{op: op}} = ans, {cache, state}) do
    {res, {cache, state}} = eval_apply(op, ans, {cache, state})
    state.gc? && :erlang.garbage_collect(self())
    {res, {cache, state}}
  end

  defp eval(other, {cache, state}) do
    {other, {cache, state}}
  end

  defp eval_apply(:parameter, %T{data: %Expr{id: id, args: [i]}}, {cache, state}) do
    %T{} = tensor = Map.fetch!(state.params, i).()
    config = Map.fetch!(state.sharding_configs, i)
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

  defp apply_op(:dot, ans, [arg0, contract0, [], arg1, contract1, []], state) do
    dbg({arg0, contract0})
    dbg({arg1, contract1})

    {config0, config1, contracted_sharding_configs} =
      Enum.zip_reduce(
        contract0,
        contract1,
        {arg0.data.sharding_config, arg1.data.sharding_config, state.contracted_sharding_configs},
        fn
          axis0, axis1, {config0, config1, contracted_sharding_configs} ->
            entry0 = Enum.fetch!(config0, axis0)
            entry1 = Enum.fetch!(config1, axis1)

            unless is_nil(entry0) or is_nil(entry1) do
              raise "incompatible sharding"
            end

            dbg(entry0)

            contracted_sharding_configs =
              if entry0 do
                Map.put(contracted_sharding_configs, entry0.id, entry0)
              else
                contracted_sharding_configs
              end

            contracted_sharding_configs =
              if entry1 do
                Map.put(contracted_sharding_configs, entry1.id, entry1)
              else
                contracted_sharding_configs
              end

            {
              List.replace_at(config0, axis0, :delete),
              List.replace_at(config1, axis1, :delete),
              contracted_sharding_configs
            }
        end
      )

    config = Enum.reject(config0, &(&1 == :delete)) ++ Enum.reject(config1, &(&1 == :delete))

    {shard(ans, config),
     Map.put(state, :contracted_sharding_configs, contracted_sharding_configs)}
  end

  @unary_ops [:exp, :expm1, :log, :log1p, :sigmoid, :cos, :sin, :tan, :cosh, :sinh, :tanh] ++
               [:acosh, :asinh, :atanh, :sqrt, :rsqrt, :cbrt, :negate, :sign, :abs, :bitwise_not] ++
               [:is_nan, :is_infinity] ++
               [:conjugate, :population_count, :count_leading_zeros, :floor, :ceil, :round] ++
               [:erf, :erfc, :erf_inv, :acos, :asin, :atan, :bitcast, :real, :imag]

  defp apply_op(op, ans, [arg], state) when op in @unary_ops do
    {shard(ans, arg.data.sharding_config), state}
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
    sharding_config = merge_sharding_config(arg0, arg1)
    dbg({sharding_config, arg0, arg1})
    {shard(ans, sharding_config), state}
  end

  defp merge_sharding_config(
         %T{data: %__MODULE__{sharding_config: config1}},
         %T{data: %__MODULE__{sharding_config: config2}}
       ) do
    Enum.zip_with(config1, config2, fn
      c, nil ->
        c

      nil, c ->
        c

      c, c ->
        c

      %AxisConfig{}, %AxisConfig{} ->
        raise "conflict resolution not supported yet"
    end)
  end
end
