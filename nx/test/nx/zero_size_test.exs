defmodule Nx.ZeroSizeTest do
  use ExUnit.Case, async: true

  import Nx.Testing

  defp empty(shape, type \\ :f32), do: Nx.broadcast(Nx.tensor(0, type: type), shape)

  describe "creation and conversion" do
    test "shapes accept 0, window dimensions do not" do
      assert Nx.Shape.validate!({2, 0, 3}, :shape) == {2, 0, 3}

      assert_raise ArgumentError, ~r/must be a positive integer/, fn ->
        Nx.Shape.validate!({0, 2}, :window_dimensions)
      end
    end

    test "broadcast, iota and eye" do
      assert Nx.shape(empty({2, 0})) == {2, 0}
      assert Nx.shape(Nx.iota({0})) == {0}
      assert Nx.shape(Nx.iota({2, 0, 3}, axis: 2)) == {2, 0, 3}
      assert Nx.shape(Nx.eye({0, 0})) == {0, 0}
    end

    test "tensor and from_binary" do
      assert Nx.shape(Nx.tensor([])) == {0}
      assert Nx.type(Nx.tensor([])) == {:f, 32}
      assert Nx.shape(Nx.tensor([[], []])) == {2, 0}
      assert Nx.shape(Nx.tensor([], type: :s64)) == {0}
      assert Nx.shape(Nx.from_binary(<<>>, :f32)) == {0}
    end

    test "to_binary, to_flat_list, to_list and inspect" do
      assert Nx.to_binary(empty({2, 0})) == <<>>
      assert Nx.to_flat_list(empty({2, 0})) == []
      assert Nx.to_list(empty({2, 0})) == [[], []]
      assert Nx.to_list(empty({0, 2})) == []
      assert inspect(empty({2, 0})) =~ "f32[2][0]"
    end

    test "to_number raises" do
      assert_raise ArgumentError, fn -> Nx.to_number(empty({0})) end
    end
  end

  describe "elementwise" do
    test "unary and binary operations keep the shape" do
      e = empty({2, 0})
      assert Nx.shape(Nx.exp(e)) == {2, 0}
      assert Nx.shape(Nx.negate(e)) == {2, 0}
      assert Nx.shape(Nx.add(e, 1)) == {2, 0}
      assert Nx.shape(Nx.add(e, e)) == {2, 0}
      assert Nx.shape(Nx.equal(e, 0)) == {2, 0}
      assert Nx.shape(Nx.select(Nx.greater(e, 0), e, 1.0)) == {2, 0}
      assert Nx.shape(Nx.clip(e, 0, 1)) == {2, 0}
      assert Nx.shape(Nx.as_type(e, :s32)) == {2, 0}
      assert Nx.shape(Nx.bitcast(e, :s32)) == {2, 0}
    end

    test "a dimension of 1 broadcasts to 0" do
      assert Nx.shape(Nx.add(Nx.iota({2, 1}), empty({2, 0}))) == {2, 0}
      assert Nx.shape(Nx.add(empty({0, 1}), Nx.iota({1, 3}))) == {0, 3}
    end

    test "a zero-size tensor broadcasts to more axes" do
      assert Nx.shape(Nx.broadcast(empty({2, 0}), {3, 2, 0})) == {3, 2, 0}
      assert Nx.shape(Nx.add(empty({2, 0}), empty({3, 2, 0}))) == {3, 2, 0}
    end

    test "a dimension of 0 does not broadcast to other sizes" do
      assert_raise ArgumentError, fn -> Nx.add(empty({0}), Nx.iota({3})) end
    end
  end

  describe "structure" do
    test "reshape" do
      assert Nx.shape(Nx.reshape(Nx.iota({0}), {0, 5})) == {0, 5}
      assert Nx.shape(Nx.reshape(empty({2, 0}), {0})) == {0}
      assert Nx.shape(Nx.reshape(empty({2, 0}), {:auto, 2})) == {0, 2}

      assert_raise ArgumentError, ~r/:auto/, fn -> Nx.reshape(empty({2, 0}), {:auto, 0}) end
    end

    test "transpose, new_axis, squeeze and reverse" do
      assert Nx.shape(Nx.transpose(empty({2, 0}))) == {0, 2}
      assert Nx.shape(Nx.transpose(empty({2, 0, 3}), axes: [2, 0, 1])) == {3, 2, 0}
      assert Nx.shape(Nx.new_axis(empty({2, 0}), 0)) == {1, 2, 0}
      assert Nx.shape(Nx.squeeze(empty({1, 0}))) == {0}
      assert Nx.shape(Nx.reverse(empty({2, 0}))) == {2, 0}
    end

    test "concatenate skips empty parts" do
      x = Nx.iota({2, 3})
      assert_equal(Nx.concatenate([empty({2, 0}, :s32), x], axis: 1), x)
      assert Nx.shape(Nx.concatenate([empty({2, 0}), empty({2, 0})], axis: 1)) == {2, 0}
      assert Nx.shape(Nx.concatenate([empty({0, 3}), Nx.iota({2, 3})])) == {2, 3}
    end

    test "stack" do
      assert Nx.shape(Nx.stack([empty({2, 0}), empty({2, 0})])) == {2, 2, 0}
    end

    test "slice and slice_along_axis with a length of 0" do
      x = Nx.iota({2, 3})
      assert Nx.shape(Nx.slice(x, [0, 1], [2, 0])) == {2, 0}
      assert Nx.shape(Nx.slice_along_axis(x, 3, 0, axis: 1)) == {2, 0}
      assert Nx.shape(Nx.slice(empty({2, 0}), [1, 0], [1, 0])) == {1, 0}
      assert Nx.shape(Nx.slice(Nx.iota({4}), [1], [0])) == {0}
      assert Nx.shape(Nx.slice(x, [1, 0], [0, 3])) == {0, 3}
    end

    test "random tensors" do
      key = Nx.Random.key(42)

      for shape <- [{0, 3}, {3, 0}, {0}] do
        {uniform, _key} = Nx.Random.uniform(key, shape: shape)
        {normal, _key} = Nx.Random.normal(key, shape: shape)
        assert Nx.shape(uniform) == shape
        assert Nx.shape(normal) == shape
      end
    end

    test "put_slice into and from empty tensors" do
      x = Nx.iota({2, 3})
      assert_equal(Nx.put_slice(x, [0, 1], empty({2, 0}, :s32)), x)
      assert Nx.shape(Nx.put_slice(empty({2, 0}), [0, 0], empty({2, 0}))) == {2, 0}
    end

    test "pad" do
      assert_equal(Nx.pad(empty({2, 0}), 1.0, [{0, 0, 0}, {1, 1, 0}]), Nx.broadcast(1.0, {2, 2}))
      assert Nx.shape(Nx.pad(empty({2, 0}), 1.0, [{1, 0, 0}, {0, 0, 0}])) == {3, 0}
    end

    test "tile" do
      assert Nx.shape(Nx.tile(Nx.iota({2, 3}), [1, 0])) == {2, 0}
      assert Nx.shape(Nx.tile(empty({0, 2}), [3, 1])) == {0, 2}
    end
  end

  describe "reductions" do
    test "with an initial value give it for an empty axis" do
      e = empty({2, 0})
      assert_equal(Nx.sum(e), Nx.tensor(0.0))
      assert_equal(Nx.sum(e, axes: [1]), Nx.tensor([0.0, 0.0]))
      assert Nx.shape(Nx.sum(e, axes: [0])) == {0}
      assert_equal(Nx.product(e, axes: [1]), Nx.tensor([1.0, 1.0]))
      assert_equal(Nx.all(e, axes: [1]), Nx.tensor([1, 1], type: :u8))
      assert_equal(Nx.any(e, axes: [1]), Nx.tensor([0, 0], type: :u8))
      assert_equal(Nx.reduce(e, 5.0, [axes: [1]], &Nx.add/2), Nx.tensor([5.0, 5.0]))
    end

    test "that divide by the size give NaN, logsumexp gives -inf" do
      e = empty({2, 0})
      assert Nx.to_number(Nx.mean(e)) == :nan
      assert Nx.to_list(Nx.mean(e, axes: [1])) == [:nan, :nan]
      assert Nx.to_number(Nx.variance(e)) == :nan
      assert Nx.to_list(Nx.logsumexp(e, axes: [1])) == [:neg_infinity, :neg_infinity]
    end

    test "max, min, argmax and argmin start from the limits of the type" do
      e = empty({2, 0})
      assert Nx.to_list(Nx.reduce_max(e, axes: [1])) == [:neg_infinity, :neg_infinity]
      assert Nx.to_list(Nx.reduce_min(e, axes: [1])) == [:infinity, :infinity]
      assert Nx.to_number(Nx.reduce_max(empty({0}, :s32))) == -2_147_483_648
      assert Nx.to_list(Nx.argmax(e, axis: 1)) == [0, 0]
      assert Nx.to_number(Nx.argmin(e)) == 0
    end

    test "keep an empty output" do
      e = empty({2, 0})
      assert Nx.shape(Nx.reduce_max(e, axes: [0])) == {0}
      assert Nx.shape(Nx.argmax(e, axis: 0)) == {0}
    end

    test "cumulative operations keep the shape" do
      assert Nx.shape(Nx.cumulative_sum(empty({2, 0}), axis: 1)) == {2, 0}
      assert Nx.shape(Nx.cumulative_max(empty({2, 0}), axis: 0)) == {2, 0}
    end
  end

  describe "sorting" do
    test "sort, argsort and top_k" do
      assert Nx.shape(Nx.sort(empty({2, 0}), axis: 1)) == {2, 0}
      assert Nx.shape(Nx.argsort(empty({0, 3}), axis: 1)) == {0, 3}
      {values, indices} = Nx.top_k(Nx.iota({2, 3}, type: :f32), k: 0)
      assert Nx.shape(values) == {2, 0}
      assert Nx.shape(indices) == {2, 0}
    end
  end

  describe "dot and conv" do
    test "dot over an empty contraction axis gives zeros" do
      assert_equal(Nx.dot(empty({2, 0}), empty({0, 3})), Nx.broadcast(0.0, {2, 3}))
      assert Nx.shape(Nx.dot(empty({0, 3}), Nx.iota({3, 2}, type: :f32))) == {0, 2}
      assert Nx.shape(Nx.dot(empty({0, 2, 3}), [2], [0], empty({0, 3, 4}), [1], [0])) == {0, 2, 4}
    end

    test "conv with an empty batch or empty input channels" do
      assert Nx.shape(Nx.conv(empty({0, 1, 4}), Nx.broadcast(1.0, {1, 1, 2}))) == {0, 1, 3}
      assert_equal(Nx.conv(empty({1, 0, 4}), empty({2, 0, 2})), Nx.broadcast(0.0, {1, 2, 3}))
    end
  end

  describe "indexing" do
    test "take and take_along_axis with empty indices" do
      x = Nx.iota({2, 3})
      none = Nx.tensor([], type: :s64)
      assert Nx.shape(Nx.take(x, none, axis: 1)) == {2, 0}
      assert Nx.shape(Nx.take(empty({0, 3}), none)) == {0, 3}
      assert Nx.shape(Nx.take_along_axis(x, empty({2, 0}, :s64), axis: 1)) == {2, 0}
    end

    test "gather with empty indices" do
      assert Nx.shape(Nx.gather(Nx.iota({2, 3}), empty({0, 2}, :s64))) == {0}
    end

    test "indexed_put and indexed_add with empty indices return the target" do
      x = Nx.iota({2, 3})
      assert_equal(Nx.indexed_put(x, empty({0, 2}, :s64), empty({0}, :s32)), x)
      assert_equal(Nx.indexed_add(x, empty({0, 2}, :s64), empty({0}, :s32)), x)
    end

    test "window operations on an empty input" do
      assert Nx.shape(Nx.window_sum(empty({0, 4}), {1, 2})) == {0, 3}
      assert Nx.shape(Nx.window_max(empty({2, 0}), {1, 1})) == {2, 0}
    end
  end

  describe "defn" do
    test "jit with the evaluator" do
      e = empty({2, 0})
      assert Nx.shape(Nx.Defn.jit(&Nx.add(&1, 1)).(e)) == {2, 0}
      assert_equal(Nx.Defn.jit(&Nx.sum(&1, axes: [1])).(e), Nx.tensor([0.0, 0.0]))
      assert_equal(Nx.Defn.jit(&Nx.dot/2).(e, empty({0, 3})), Nx.broadcast(0.0, {2, 3}))
    end

    test "grad through a broadcast to a zero-size shape" do
      grad = Nx.Defn.grad(Nx.tensor(2.0), &Nx.sum(Nx.multiply(Nx.broadcast(&1, {0}), 3)))
      assert_equal(grad, Nx.tensor(0.0))
    end

    test "grad of a zero-size input has its shape" do
      assert Nx.shape(Nx.Defn.grad(empty({2, 0}), &Nx.sum(Nx.exp(&1)))) == {2, 0}

      grad = Nx.Defn.grad(empty({2, 0}), &Nx.sum(Nx.dot(&1, Nx.broadcast(1.0, {0, 3}))))
      assert Nx.shape(grad) == {2, 0}
    end

    test "grad through dot over an empty contraction axis" do
      grad = Nx.Defn.grad(Nx.iota({2, 3}, type: :f32), &Nx.sum(Nx.dot(empty({0, 2}), &1)))
      assert_equal(grad, Nx.broadcast(0.0, {2, 3}))
    end
  end
end
