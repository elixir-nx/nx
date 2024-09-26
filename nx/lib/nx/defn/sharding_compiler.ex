defmodule Nx.Defn.ShardingCompiler do
  alias Nx.Defn.Composite
  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  @behaviour Nx.Defn.Compiler

  defstruct [:sharding_config]

  defmodule AxisConfig do
    @derive Inspect
    defstruct [:slices]

    def inspect(%__MODULE__{slices: slices}, _inspect_opts) do
      doc =
        slices
        |> Enum.map(fn start..slice_end//1 -> "#{start}..#{slice_end}" end)
        |> Enum.intersperse(", ")
        |> Inspect.Algebra.concat()

      Inspect.Algebra.concat([
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
        %AxisConfig{slices: slices}
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
    [args] = args

    [%T{shape: shape, type: type, data: %__MODULE__{sharding_config: output_config}}] =
      __compile__(key, vars, fun, opts).([args])

    slices =
      Enum.with_index(output_config, fn
        nil, axis -> {axis, [..]}
        %AxisConfig{slices: slices}, axis -> {axis, slices}
      end)

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

    result =
      for instance <- sharded_args do
        args = Enum.map(instance, &elem(&1, 1))
        starts = Enum.map(instance, &elem(&1, 0))
        compiled_fun = Nx.Defn.Evaluator.__compile__(key, args, fun, [])

        {
          [args],
          fn args ->
            [res] =
              compiled_fun.([
                Enum.map(args, fn arg ->
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
  def __compile__(key, vars, fun, opts) do
    expr = fun.(vars)

    state_shard_configs =
      opts
      |> Keyword.get(:sharding_config)
      |> Enum.with_index(fn x, idx -> {idx, x} end)
      |> Map.new()

    fn [params] ->
      state_params = params |> Enum.with_index(fn x, idx -> {idx, x} end) |> Map.new()

      {container, _cache} =
        composite_eval(
          expr,
          %{
            gc?: opts[:garbage_collect] || false,
            params: state_params,
            sharding_configs: state_shard_configs
          },
          %{}
        )

      [container]
    end
  end

  defp composite_eval(expr, state, cache) do
    Composite.traverse(expr, cache, &eval(&1, state, &2))
  end

  defp eval(%T{data: %Expr{op: :tensor, args: [t]}}, _state, cache) do
    {t, cache}
  end

  defp eval(%T{data: %Expr{op: :constant, args: [_constant]}} = ans, _state, cache) do
    {ans, cache}
  end

  defp eval(%T{data: %Expr{op: :metadata, args: [expr, _meta]}}, state, cache) do
    composite_eval(expr, state, cache)
  end

  defp eval(%T{data: %Expr{op: op}} = ans, state, cache) do
    {res, cache} = eval_apply(op, ans, state, cache)
    state.gc? && :erlang.garbage_collect(self())
    {res, cache}
  end

  defp eval(other, _state, [_ | _] = cache) do
    {other, cache}
  end

  defp eval_apply(:parameter, %T{data: %Expr{id: id, args: [i]}}, state, cache) do
    %T{} = tensor = Map.fetch!(state.params, i).()
    config = Map.fetch!(state.sharding_configs, i)
    res = tensor |> Nx.devectorize() |> shard(config)
    {res, Map.put(cache, id, res)}
  end

  defp eval_apply(:elem, %T{data: %Expr{id: id, args: [tuple, i]}}, state, cache) do
    {tuple, cache} = composite_eval(tuple, state, cache)
    res = elem(tuple, i)
    {res, Map.put(cache, id, res)}
  end

  defp eval_apply(op, %T{data: %Expr{id: id}} = ans, state, cache) do
    {args, cache} = Nx.Defn.Tree.apply_args(ans, cache, &eval(&1, state, &2))

    res = apply_op(op, ans, args)
    {res, Map.put(cache, id, res)}
  end

  @unary_ops [:exp, :expm1, :log, :log1p, :sigmoid, :cos, :sin, :tan, :cosh, :sinh, :tanh] ++
               [:acosh, :asinh, :atanh, :sqrt, :rsqrt, :cbrt, :negate, :sign, :abs, :bitwise_not] ++
               [:is_nan, :is_infinity] ++
               [:conjugate, :population_count, :count_leading_zeros, :floor, :ceil, :round] ++
               [:erf, :erfc, :erf_inv, :acos, :asin, :atan, :bitcast, :real, :imag]

  defp apply_op(op, ans, [arg]) when op in @unary_ops do
    shard(ans, arg.data.sharding_config)
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

  defp apply_op(op, ans, [arg0, arg1]) when op in @binary_ops do
    sharding_config = merge_sharding_config(arg0, arg1)
    shard(ans, sharding_config)
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
