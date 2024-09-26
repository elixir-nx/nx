defmodule Nx.Defn.ShardingCompiler.ShardingBackend do
  @behaviour Nx.Backend

  @derive Inspect
  # sharding_config is a list of AxisShardingConfig
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
        %Nx.Tensor{names: names, data: %__MODULE__{sharding_config: sharding_config}},
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

  def init(opts), do: opts

  def shard(tensor, config) do
    sharding_config = build_sharding_config(tensor, config)
    put_in(tensor.data, %__MODULE__{sharding_config: sharding_config})
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

  @impl true
  def constant(out, _number, opts) do
    put_in(out.data, %__MODULE__{sharding_config: []})
  end

  @impl true
  def from_binary(out, _binary, opts) do
    sharding_config = Keyword.get(opts, :sharding_config) || build_sharding_config(out, %{})

    put_in(out.data, %__MODULE__{sharding_config: sharding_config})
  end

  unary_ops =
    [:exp, :expm1, :log, :log1p, :sigmoid, :cos, :sin, :tan, :cosh, :sinh, :tanh] ++
      [:acosh, :asinh, :atanh, :sqrt, :rsqrt, :cbrt, :negate, :sign, :abs, :bitwise_not] ++
      [:is_nan, :is_infinity] ++
      [:conjugate, :population_count, :count_leading_zeros, :floor, :ceil, :round] ++
      [:erf, :erfc, :erf_inv, :acos, :asin, :atan, :bitcast, :real, :imag]

  for op <- unary_ops do
    @impl true
    def unquote(op)(out, tensor) do
      unary_op(out, tensor)
    end
  end

  binary_ops =
    [:add, :subtract, :multiply, :pow, :remainder, :divide, :atan2, :min, :max, :quotient] ++
      [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift] ++
      [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal] ++
      [:logical_and, :logical_or, :logical_xor]

  for op <- binary_ops do
    @impl true
    def unquote(op)(out, t1, t2) do
      binary_op(out, t1, t2)
    end
  end

  defp unary_op(out, tensor) do
    shard(out, tensor.data.sharding_config)
  end

  defp binary_op(out, t1, t2) do
    sharding_config = merge_sharding_config(t1, t2)
    shard(out, sharding_config)
  end

  defp merge_sharding_config(
         %Nx.Tensor{data: %__MODULE__{sharding_config: config1}},
         %Nx.Tensor{data: %__MODULE__{sharding_config: config2}}
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
