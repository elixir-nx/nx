defmodule Nx.Defn.CompositeTest do
  use ExUnit.Case, async: true

  alias Nx.Defn.Composite
  doctest Nx.Defn.Composite

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
              }, Nx.tensor(2, type: {:c, 64})} ==
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
  end
end
