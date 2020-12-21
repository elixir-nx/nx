defmodule DefnNewTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  alias Nx.Defn.Expr

  @default_defn_compiler Nx.Defn.New

  describe "rank, shape, size" do
    defn rank(t), do: Nx.rank(t)
    defn shape(t), do: Nx.shape(t)
    defn size(t), do: Nx.size(t)

    test "rank" do
      assert 2 == rank(Nx.tensor([[1, 2, 3], [1, 2, 3]]))
    end

    test "shape" do
      assert {2, 3} == shape(Nx.tensor([[1, 2, 3], [1, 2, 3]]))
    end

    test "size" do
      assert 6 == size(Nx.tensor([[1, 2, 3], [1, 2, 3]]))
    end
  end

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

  describe "creation ops" do
    defn iota(t), do: Nx.iota(t)
    defn random_uniform(t), do: Nx.random_uniform(t, 0.0, 2.0)
    defn random_normal(t), do: Nx.random_normal(t, 0.0, 1.0)

    test "iota" do
      assert %Expr{op: :iota, args: [{3}, []], shape: {3}} = iota(Nx.tensor([1, 2, 3]))
    end

    test "random uniform" do
      assert %Expr{op: :random_uniform, args: [{3}, 0.0, 2.0, []], shape: {3}} = random_uniform(Nx.tensor([1, 2, 3]))
    end

    test "random normal" do
      assert %Expr{op: :random_normal, args: [{3}, 0.0, 1.0, []], shape: {3}} = random_normal(Nx.tensor([1, 2, 3]))
    end
  end

  describe "tensor ops" do
    defn dot(t1, t2), do: Nx.dot(t1, t2)
    defn outer(t1, t2), do: Nx.outer(t1, t2)
    defn transpose(t), do: Nx.transpose(t)
    defn reshape(t), do: Nx.reshape(t, {2, 3})
    defn broadcast(t), do: Nx.broadcast(t, {3, 3, 3})

    test "dot product" do
      assert %Expr{op: :dot, args: [_, _], shape: {2, 2}} = dot(Nx.tensor([[1, 2, 3], [1, 2, 3]]), Nx.tensor([[1, 2], [3, 4], [5, 6]]))
    end

    test "outer product" do
      assert %Expr{op: :outer, args: [_, _], shape: {3, 3}} = outer(Nx.tensor([1, 2, 3]), Nx.tensor([1, 2, 3]))
    end

    test "transpose" do
      assert %Expr{op: :transpose, args: [_, _], shape: {3, 2}} = transpose(Nx.tensor([[1, 2, 3], [1, 2, 3]]))
    end

    test "reshape" do
      assert %Expr{op: :reshape, args: [_, _], shape: {2, 3}} = reshape(Nx.tensor([[1, 2], [3, 4], [5, 6]]))
    end

    test "broadcast" do
      assert %Expr{op: :broadcast, args: [_, _], shape: {3, 3, 3}} = broadcast(Nx.tensor([1, 2, 3]))
    end
  end

  describe "conditional ops" do
    defn select(t1, t2, t3), do: Nx.select(t1, t2, t3)

    test "select" do
      assert %Expr{op: :select, args: [_, _, _], shape: {2, 2}} = select(Nx.tensor([[1, 1], [0, 0]]), Nx.tensor(1), Nx.tensor(0))
    end

  end
end