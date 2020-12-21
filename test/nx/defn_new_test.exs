defmodule DefnNewTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  alias Nx.Defn.Expr
  alias Nx.Defn.Translation

  @default_defn_compiler Nx.Defn.New

  describe "unary ops" do
    defn exp(t), do: Nx.exp(t)

    test "to expr" do
      assert %Expr{op: :exp, args: [_], shape: {3}} = exp(Nx.tensor([1, 2, 3]))
    end
  end

  describe "binary ops" do
    defn add(t1, t2), do: Nx.add(t1, t2)

    test "to expr" do
      assert %Expr{op: :add, args: [_, _], shape: {3}} = add(Nx.tensor([1, 2, 3]), Nx.tensor(1))
      assert %Expr{op: :add, args: [_, _], shape: {2, 2}} = add(Nx.tensor([[1, 2], [3, 4]]), Nx.tensor([1, 2]))
    end
  end

  describe "aggregate ops" do
    defn sum_all(t), do: Nx.sum(t)
    defn sum_1(t), do: Nx.sum(t, axis: 1)
    defn sum_2(t), do: Nx.sum(t, axes: [0, 1])

    test "to expr" do
      assert %Expr{op: :sum, args: [_, _], shape: {}} = sum_all(Nx.tensor([1, 2, 3]))
      assert %Expr{op: :sum, args: [_, _], shape: {2}} = sum_1(Nx.tensor([[1, 2, 3], [1, 2, 3]]))
      assert %Expr{op: :sum, args: [_, _], shape: {}} = sum_2(Nx.tensor([[1, 2, 3], [1, 2, 3]]))
    end
  end

  describe "broadcasting semantics" do

    test "broadcasts scalars" do
      assert Translation.broadcast({}, {}) == {}
      assert Translation.broadcast({}, {4, 2, 1, 5}) == {4, 2, 1, 5}
      assert Translation.broadcast({4, 2, 1, 5}, {}) == {4, 2, 1, 5}
    end

    test "broadcasts correctly" do
      assert Translation.broadcast({8, 1, 6, 1}, {7, 1, 5}) == {8, 7, 6, 5}
      assert Translation.broadcast({7, 1, 5}, {8, 1, 6, 1}) == {8, 7, 6, 5}
      assert Translation.broadcast({5, 4}, {1}) == {5, 4}
      assert Translation.broadcast({5, 4}, {4}) == {5, 4}
      assert Translation.broadcast({15, 3, 5}, {15, 1, 5}) == {15, 3, 5}
      assert Translation.broadcast({3, 1}, {15, 3, 5}) == {15, 3, 5}
    end

    test "raises on bad dims" do
      assert_raise ArgumentError, "could not broadcast shapes because dimensions are" <>
                                  " incompatible, expected dimensions to be equal or" <>
                                  " either dimension to be 1, got: 4 and 3", fn ->
            Translation.broadcast({4, 2, 5}, {3, 2, 5})
      end
    end
  end
end