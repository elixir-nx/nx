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
end
