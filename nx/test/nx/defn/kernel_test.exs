defmodule Nx.Defn.KernelTest do
  use ExUnit.Case, async: true

  describe "doctests" do
    use Nx.Defn.Kernel
    doctest Nx.Defn.Kernel
  end

  describe "with numbers" do
    test "+" do
      assert Nx.Defn.Kernel.+(1, 2) == 3
    end

    test "-" do
      assert Nx.Defn.Kernel.-(1, 2) == -1
    end

    test "*" do
      assert Nx.Defn.Kernel.*(1, 2) == 2
    end

    test "/" do
      assert Nx.Defn.Kernel./(1, 2) == 0.5
    end

    test "and" do
      assert Nx.Defn.Kernel.and(0, 0) == 0
      assert Nx.Defn.Kernel.and(1, 0) == 0
      assert Nx.Defn.Kernel.and(0, 2) == 0
      assert Nx.Defn.Kernel.and(1, 1) == 1

      assert Nx.Defn.Kernel.and(0, 0.0) == 0.0
      assert Nx.Defn.Kernel.and(1, 0.0) == 0.0
      assert Nx.Defn.Kernel.and(0, 2.0) == 0.0
      assert Nx.Defn.Kernel.and(1, 1.0) == 1.0
    end

    test "or" do
      assert Nx.Defn.Kernel.or(0, 0) == 0
      assert Nx.Defn.Kernel.or(1, 0) == 1
      assert Nx.Defn.Kernel.or(0, 2) == 1
      assert Nx.Defn.Kernel.or(1, 1) == 1

      assert Nx.Defn.Kernel.or(0, 0.0) == 0.0
      assert Nx.Defn.Kernel.or(1, 0.0) == 1.0
      assert Nx.Defn.Kernel.or(0, 2.0) == 1.0
      assert Nx.Defn.Kernel.or(1, 1.0) == 1.0
    end

    test "not" do
      assert Nx.Defn.Kernel.not(0) == 1
      assert Nx.Defn.Kernel.not(1) == 0
      assert Nx.Defn.Kernel.not(2) == 0

      assert Nx.Defn.Kernel.not(0.0) == 1.0
      assert Nx.Defn.Kernel.not(1.0) == 0.0
      assert Nx.Defn.Kernel.not(2.0) == 0.0
    end

    test "&&&" do
      assert Nx.Defn.Kernel.&&&(1, 2) == 0
    end

    test "|||" do
      assert Nx.Defn.Kernel.|||(1, 2) == 3
    end

    test "<<<" do
      assert Nx.Defn.Kernel.<<<(1, 2) == 4
    end

    test ">>>" do
      assert Nx.Defn.Kernel.>>>(1, 2) == 0
    end

    test "unary +/-" do
      assert Nx.Defn.Kernel.+(1) == 1
      assert Nx.Defn.Kernel.-(1) == -1
    end

    test "~~~" do
      assert Nx.Defn.Kernel.~~~(1) == -2
    end

    test "min/max" do
      assert Nx.Defn.Kernel.min(0, 1) == 0
      assert Nx.Defn.Kernel.max(0, 1) == 1
    end

    test ".." do
      assert Nx.Defn.Kernel.".."(1, 2) == 1..2
    end
  end

  describe "inside defn" do
    import Nx.Defn

    defn assert_square_matrix(tensor) do
      assert_shape_pattern(tensor, {x, x})
    end

    test "assert_shape_pattern" do
      assert_square_matrix(Nx.tensor([[1, 2], [3, 4]]))

      assert_raise ArgumentError,
                   "expected tensor to match shape {x, x}, got tensor with shape {1, 2}",
                   fn -> assert_square_matrix(Nx.tensor([[1, 2]])) end
    end
  end
end
