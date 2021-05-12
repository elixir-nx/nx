defmodule Nx.Defn.VmapTest do
  use ExUnit.Case, async: true

  import Nx.Defn

  describe "simple" do
    defn vmap_itself(t), do: vmap(fn t -> t end, [t])
    defn vmap_tensor(t), do: vmap(fn _t -> Nx.tensor(1.0) end, [t])
    defn vmap_tuple(t1, t2), do: vmap(fn t1, t2 -> {t1, t2} end, [t1, t2])

    test "vectorizes in simple cases" do
      assert vmap_itself(Nx.tensor([1])) == Nx.tensor([1])
      assert vmap_tensor(Nx.tensor([1])) == Nx.tensor(1.0)
      assert vmap_tuple(Nx.tensor([1]), Nx.tensor([1])) == {Nx.tensor([1]), Nx.tensor([1])}
    end
  end

  describe "error cases" do
    defn vmap_invalid_in_axis(t1, t2), do: vmap(fn t1, t2 -> Nx.add(t1, t2) end, [t1, t2], [0, 2])

    test "invalid in axis" do
      assert_raise ArgumentError, ~r/vmap input axes cannot exceed rank/, fn ->
        vmap_invalid_in_axis(Nx.tensor(1), Nx.tensor(1))
      end
    end

    defn vmap_too_many_axes(t), do: vmap(fn t -> t end, [t], [0, 0])

    test "too many axes" do
      assert_raise ArgumentError, ~r/length of vmap input axes/, fn ->
        vmap_too_many_axes(Nx.tensor([1]))
      end
    end

    defn vmap_too_few_axes(t1, t2), do: vmap(fn t1, _ -> t1 end, [t1, t2], [0])

    test "too few axes" do
      assert_raise ArgumentError, ~r/length of vmap input axes/, fn ->
        vmap_too_few_axes(Nx.tensor([1]), Nx.tensor([1]))
      end
    end

    defn vmap_batch_sizes_not_matching(t1, t2), do: vmap(fn t1, t2 -> {t1, t2} end, [t1, t2], [0, 0])

    test "batch sizes do not match" do
      assert_raise ArgumentError, ~r/batch sizes must match/, fn ->
        vmap_batch_sizes_not_matching(Nx.iota({2, 3, 2}), Nx.iota({1, 2, 3}))
      end
    end

    defn vmap_no_non_nil(t1), do: vmap(fn t1 -> t1 end, [t1], [nil])

    test "at least one non-nil axis" do
      assert_raise ArgumentError, ~r/at least 1 input axis passed to vmap must be non-nil/, fn ->
        vmap_no_non_nil(Nx.iota({2, 3, 2}))
      end
    end
  end

  describe "nested vmap" do
    defn vmap_nested(t1, t2), do: vmap(&Nx.cos(Nx.dot(&1, [1], [], &2, [1], [])), [t1, t2])

    test "vectorizes over nested expression" do
      t = Nx.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
      lhs = vmap_nested(t, t)
      rhs = Nx.tensor([[[0.13673722, 0.83422336], [0.83422336, -0.03097503]]])

      compare_tensors!(lhs, rhs)
    end
  end

  describe "element-wise unary ops" do
    defn vmap_cos(t), do: vmap(fn t -> Nx.cos(t) end, [t])

    test "vectorizes over cos" do
      assert vmap_cos(Nx.iota({4, 4, 4})) == Nx.cos(Nx.iota({4, 4, 4}))
    end
  end

  describe "dot product" do
    defn vmap_dot(t1, t2), do: vmap(&Nx.dot(&1, &2), [t1, t2])
    defn vmap_dot_general(t1, t2), do: vmap(&Nx.dot(&1, [1], [], &2, [1], []), [t1, t2])

    test "vectorizes over both sides" do
      t = Nx.tensor([[[1, 2, 3], [4, 5, 6]]])
      # assert vmap_dot_general(t, t) == Nx.tensor([[[14, 32], [32, 77]]])

      t1 = Nx.iota({2, 3, 2})
      t2 = Nx.iota({2, 2, 3})

      assert vmap_dot(t1, t2) == Nx.tensor(1)
    end
  end

  defp compare_tensors!(left, right) do
    atol = 1.0e-7
    rtol = 1.0e-4

    try do
      assert Nx.all_close?(left, right, atol: atol, rtol: rtol) == Nx.tensor(1, type: {:u, 8})
    rescue
      # So we can see the diff
      _ -> assert left == right
    end
  end
end