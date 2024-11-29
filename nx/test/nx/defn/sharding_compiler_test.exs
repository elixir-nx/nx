defmodule Nx.Defn.ShardingCompilerTest do
  use ExUnit.Case, async: true

  test "end to end test with single stage" do
    arg0 =
      Nx.tensor([
        [1, 2, 3],
        [4, 5, 6]
      ])

    arg1 =
      Nx.tensor([
        [1, 2],
        [3, 4],
        [5, 6]
      ])

    fun = fn arg0, arg1, arg2 ->
      x = Nx.add(arg0, 1)
      y = Nx.subtract(arg1, 2)

      Nx.multiply(x, Nx.transpose(y))
      |> Nx.add(arg2)
    end

    result =
      Nx.Defn.jit(fun,
        compiler: Nx.Defn.ShardingCompiler,
        sharding_config: [%{0 => 1, 1 => 3}, %{0 => 3, 1 => 1}, %{}]
      ).(arg0, arg1, 1)

    assert result == fun.(arg0, arg1, 1)
  end

  test "composed test" do
    fun = fn l, r ->
      x = Nx.add(l, Nx.tensor([[1]]))
      x = Nx.transpose(x, axes: [0, 2, 1])
      y = Nx.subtract(r, 1)
      y = Nx.squeeze(y, axes: [0, 1])
      Nx.dot(x, [2, 1], y, [1, 0])
    end

    in0 = Nx.iota({2, 2, 3}, type: :f32)
    in1 = Nx.add(Nx.iota({1, 1, 3, 2, 2}), 10)

    inputs = [in0, in1]

    arg0_sharding = %{0 => 2, 1 => 2}
    arg1_sharding = %{4 => 1}

    sharding = [arg0_sharding, arg1_sharding]

    result = assert_sharded_results(fun, inputs, sharding)

    assert result.shape == {2, 2}
    assert result.type == {:f, 32}
  end

  test "shards a binary op" do
    fun = &Nx.add/2

    t = Nx.iota({3, 3})

    inputs = [t, t]

    arg0_sharding = %{0 => 1}
    arg1_sharding = %{1 => 1}

    sharding = [arg0_sharding, arg1_sharding]

    result = assert_sharded_results(fun, inputs, sharding)

    assert result.shape == {3, 3}
    assert result.type == {:s, 32}
  end

  test "shards a unary op" do
    fun = &Nx.cos/1

    t = Nx.iota({3, 3})

    inputs = [t]

    arg0_sharding = %{0 => 1, 1 => 3}

    sharding = [arg0_sharding]

    result = assert_sharded_results(fun, inputs, sharding)

    assert result.shape == {3, 3}
    assert result.type == {:f, 32}
  end

  test "shards dot product" do
    fun = &Nx.dot/2

    t0 = Nx.iota({3, 2})
    t1 = Nx.iota({2, 3})

    inputs = [t0, t1]

    arg0_sharding = %{0 => 3}
    arg1_sharding = %{1 => 3}

    sharding = [arg0_sharding, arg1_sharding]

    result = assert_sharded_results(fun, inputs, sharding)

    assert result.shape == {3, 3}
    assert result.type == {:s, 32}
  end

  test "works with literal scalars" do
    fun = fn x -> Nx.add(x, 1) end

    assert_sharded_results(fun, [Nx.iota({4})], [%{}])
  end

  test "works with literal tensors" do
    fun = fn x -> Nx.add(x, Nx.tensor([1])) end

    assert_sharded_results(fun, [Nx.iota({4})], [%{}])

    # This case raises because all literal tensors are forced
    # to be not sharded. We can in the future fix this by adding
    # fan-in and fan-out operations to the sharding compiler.
    fun = fn x -> Nx.add(x, Nx.tensor([1, 1, 1, 1])) end

    assert_raise RuntimeError, "incompatible sharding", fn ->
      assert_sharded_results(fun, [Nx.iota({4})], [%{}])
    end
  end

  test "squeeze" do
    fun = &Nx.squeeze(&1, axes: [0, 1])
    assert_sharded_results(fun, [Nx.iota({1, 1, 4})], [%{}])
  end

  test "transpose" do
    fun = &Nx.transpose(&1, axes: [2, 0, 1])
    result = assert_sharded_results(fun, [Nx.iota({2, 3, 4})], [%{}])
    assert result.shape == {4, 2, 3}
  end

  defp assert_sharded_results(fun, inputs, sharding) do
    sharded_result =
      Nx.Defn.jit_apply(
        fun,
        inputs,
        compiler: Nx.Defn.ShardingCompiler,
        sharding_config: sharding,
        sharding_compiler: Nx.Defn.Evaluator,
        sharding_compiler_options: []
      )

    assert sharded_result == apply(fun, inputs)
    sharded_result
  end
end
