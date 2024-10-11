defmodule Nx.Defn.ShardingCompilerTest do
  use ExUnit.Case, async: true

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

    arg0_sharding = %{0 => [0..1], 1 => [0..1]}
    arg1_sharding = %{4 => [0..0, 1..1]}

    sharding = [arg0_sharding, arg1_sharding]

    {output_holder, shards} = assert_sharded_results(fun, inputs, sharding)

    assert output_holder.shape == {2, 2}
    assert output_holder.type == {:f, 32}

    # sharding on arg0, axis 1 is discarded by dot/4
    assert [{[shard_args0], _fun0, _caster0}, {[shard_args1], _fun1, _caster1}] = shards

    assert {in0, in1[[.., .., .., .., 0..0]]} == shard_args0
    assert {in0, in1[[.., .., .., .., 1..1]]} == shard_args1
  end

  test "shards a binary op" do
    fun = &Nx.add/2

    t = Nx.iota({3, 3})

    inputs = [t, t]

    arg0_sharding = %{0 => [0..0, 1..1, 2..2]}
    arg1_sharding = %{1 => [0..0, 1..1, 2..2]}

    sharding = [arg0_sharding, arg1_sharding]

    {output_holder, shards} = assert_sharded_results(fun, inputs, sharding)

    assert output_holder.shape == {3, 3}
    assert output_holder.type == {:s, 32}

    assert length(shards) == 9

    assert Enum.with_index(shards, fn {[{arg0, arg1}], _fun, _caster}, idx ->
             assert arg0 == Nx.tensor([[idx]])
             assert arg1 == Nx.tensor([[idx]])
           end)
  end

  test "shards a unary op" do
    fun = &Nx.cos/1

    t = Nx.iota({3, 3})

    inputs = [t]

    arg0_sharding = %{0 => [0..0, 1..1, 2..2], 1 => [0..2]}

    sharding = [arg0_sharding]

    {output_holder, shards} = assert_sharded_results(fun, inputs, sharding)

    assert output_holder.shape == {3, 3}
    assert output_holder.type == {:f, 32}

    assert length(shards) == 3

    assert Enum.with_index(shards, fn {[{arg0}], _fun, _caster}, idx ->
             assert arg0 == t[[idx..idx, ..]]
           end)
  end

  test "shards dot product" do
    fun = &Nx.dot/2

    t0 = Nx.iota({3, 2})
    t1 = Nx.iota({2, 3})

    inputs = [t0, t1]

    arg0_sharding = %{0 => [0..0, 1..1, 2..2]}
    arg1_sharding = %{1 => [0..0, 1..1, 2..2]}

    sharding = [arg0_sharding, arg1_sharding]

    {output_holder, shards} = assert_sharded_results(fun, inputs, sharding)

    assert output_holder.shape == {3, 3}
    assert output_holder.type == {:s, 32}

    assert length(shards) == 9

    # ensure that the shards are appearing in the expected order
    # (although this doesn't have any practical effect, it's an important regression)
    for idx0 <- 0..2, idx1 <- 0..2, reduce: shards do
      [shard | shards] ->
        {[{arg0, arg1}], _fun, _caster} = shard

        assert arg0 == t0[[idx0..idx0, ..]]
        assert arg1 == t1[[.., idx1..idx1]]

        shards
    end
  end

  test "works with literal scalars" do
    fun = fn x -> Nx.add(x, 1) end

    {_, shards} = assert_sharded_results(fun, [Nx.iota({4})], [%{}])
    assert length(shards) == 4
  end

  test "works with literal tensors" do
    fun = fn x -> Nx.add(x, Nx.tensor([1])) end

    {_, shards} = assert_sharded_results(fun, [Nx.iota({4})], [%{}])
    assert length(shards) == 4

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
    {_, shards} = assert_sharded_results(fun, [Nx.iota({1, 1, 4})], [%{}])
    assert length(shards) == 4
  end

  test "transpose" do
    fun = &Nx.transpose(&1, axes: [2, 0, 1])
    {output_holder, shards} = assert_sharded_results(fun, [Nx.iota({2, 3, 4})], [%{}])
    assert output_holder.shape == {4, 2, 3}
    assert length(shards) == 2 * 3 * 4
  end

  defp assert_sharded_results(fun, inputs, sharding) do
    {output_holder, shards} =
      Nx.Defn.jit_apply(
        fun,
        inputs,
        compiler: Nx.Defn.ShardingCompiler,
        sharding_config: sharding,
        sharding_compiler: Nx.Defn.Evaluator,
        sharding_compiler_options: []
      )

    sharded_result =
      Enum.reduce(shards, output_holder, fn {arg, fun, caster}, acc ->
        result = fun.(arg)
        caster.(result, acc)
      end)

    assert sharded_result == apply(fun, inputs)

    {output_holder, shards}
  end
end
