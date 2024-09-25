defmodule Nx.Defn.ShardingCompiler.ShardingBackend do
  @behaviour Nx.Backend

  @derive Inspect
  # sharding_config is a list of AxisShardingConfig
  defstruct [:sharding_config]

  defmodule ShardingConfig do
    @derive Inspect
    defstruct [:num_shards, :ref]
  end

  def inspect(
        %Nx.Tensor{data: %__MODULE__{sharding_config: sharding_config}} = tensor,
        inspect_opts
      ) do
    Inspect.Algebra.concat([
      "ShardingBackend<",
      Inspect.Algebra.line(),
      Kernel.inspect(sharding_config),
      Inspect.Algebra.line(),
      ">"
    ])
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
    put_in(out.data.sharding_config, tensor.data.sharding_config)
  end

  defp binary_op(out, t1, t2) do
    sharding_config = merge_sharding_config(t1, t2)
    put_in(out.data.sharding_config, sharding_config)
  end

  defp merge_sharding_config(
         %Nx.Tensor{data: %__MODULE__{sharding_config: config1}} = t1,
         %Nx.Tensor{data: %__MODULE__{sharding_config: config2}} = t2
       ) do
    Enum.zip_with(config1, config2, fn
      c, nil ->
        c

      nil, c ->
        c

      c, c ->
        c

      %ShardingConfig{} = c1, %ShardingConfig{} = c2 ->
        raise "conflict resolution not supported yet"
    end)
  end
end
