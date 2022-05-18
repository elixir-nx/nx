defmodule Nx.Defn.CompositeTest do
  use ExUnit.Case, async: true

  import Nx, only: :sigils

  alias Nx.Defn.Composite

  doctest Nx.Defn.Composite

  describe "compatible?/3" do
    test "non-composite types" do
      result = ~V[2+2i]
      assert Nx.reshape(result, {}) == Composite.compatible?(Complex.new(0, 2), 2, &Nx.add/2)
      assert Nx.reshape(result, {}) == Composite.compatible?(2, Complex.new(0, 2), &Nx.add/2)
      assert result == Composite.compatible?(Complex.new(0, 2), ~V[2], &Nx.add/2)
      assert result == Composite.compatible?(~V[2], Complex.new(0, 2), &Nx.add/2)
      assert result == Composite.compatible?(2, ~V[2i], &Nx.add/2)
      assert result == Composite.compatible?(~V[2i], 2, &Nx.add/2)

      assert result == Composite.compatible?(~V[2], ~V[2i], &Nx.add/2)

      assert Nx.reshape(result, {}) ==
               Composite.compatible?(Complex.new(2), Complex.new(0, 2), &Nx.add/2)

      assert Nx.tensor(2) == Composite.compatible?(2, 0, &Nx.add/2)
    end

    test "tuple" do
      assert true == Composite.compatible?({Complex.new(0, 2)}, {2}, &Nx.add/2)
      assert true == Composite.compatible?({2}, {Complex.new(0, 2)}, &Nx.add/2)
      assert true == Composite.compatible?({Complex.new(0, 2)}, {~V[2]}, &Nx.add/2)
      assert true == Composite.compatible?({~V[2]}, {Complex.new(0, 2)}, &Nx.add/2)
      assert true == Composite.compatible?({2}, {~V[2i]}, &Nx.add/2)
      assert true == Composite.compatible?({~V[2i]}, {2}, &Nx.add/2)
    end

    test "structs" do
      args = {1, Complex.new(1), Nx.tensor(1)}
      c = %Container{a: args, b: args}

      assert true == Composite.compatible?(c, c, &Nx.add/2)
    end
  end

  describe "traverse/2" do
    test "works with mix of complex and tensor and number" do
      assert {
               Nx.tensor(2),
               Nx.tensor(3, type: {:c, 64}),
               Nx.tensor(4)
             } ==
               Composite.traverse(
                 {1, Complex.new(2), Nx.tensor(3)},
                 &Nx.add(&1, 1)
               )
    end
  end

  describe "traverse/3" do
    test "works with mix of complex and tensor and number" do
      assert {{
                Nx.tensor(1),
                Nx.tensor(3, type: {:c, 64}),
                Nx.tensor(4, type: {:c, 64})
              },
              Nx.tensor(2, type: {:c, 64})} ==
               Composite.traverse(
                 {1, Complex.new(2), Nx.tensor(3)},
                 0,
                 &{Nx.add(&1, &2), Nx.subtract(&1, &2)}
               )
    end
  end

  describe "reduce/3" do
    test "works with complex and tensor and number" do
      assert Nx.tensor(6, type: {:c, 64}) ==
               Composite.reduce({1, {Nx.tensor(3), {Complex.new(2)}}}, 1, &Nx.multiply/2)
    end
  end

  describe "flatten_list/3" do
    test "flattens with default args" do
      assert [1, Complex.new(2), Nx.tensor(3)] ==
               Composite.flatten_list([1, {Complex.new(2), Nx.tensor(3)}])
    end

    test "flattens with custom tail" do
      assert [1, Complex.new(2), Nx.tensor(3), 4, 5, 6] ==
               Composite.flatten_list([1, {Complex.new(2), Nx.tensor(3)}], [4, 5, 6])
    end

    test "flattens with custom function" do
      assert [Nx.tensor(1), Nx.tensor(Complex.new(2)), Nx.tensor(3)] ==
               Composite.flatten_list([1, {Complex.new(2), Nx.tensor(3)}], [], &Nx.tensor/1)
    end
  end
end
