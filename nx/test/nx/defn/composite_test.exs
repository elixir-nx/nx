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
      assert Nx.reshape(result, {}) == Composite.compatible?(Complex.new(2), Complex.new(0, 2), &Nx.add/2)
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

end
