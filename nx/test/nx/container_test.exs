defmodule Nx.ContainerTest do
  use ExUnit.Case, async: true

  test "to_template" do
    assert Nx.to_template(%Container{a: 1, b: 2, c: 3, d: 4}) ==
             %Container{a: Nx.template({}, {:s, 64}), b: Nx.template({}, {:s, 64}), c: nil, d: 4}

    assert Nx.to_template(%Container{a: 1, b: {2, 3.0}, c: 4, d: 5}) ==
             %Container{
               a: Nx.template({}, {:s, 64}),
               b: {Nx.template({}, {:s, 64}), Nx.template({}, {:f, 32})},
               c: nil,
               d: 5
             }
  end

  test "compatible?" do
    assert Nx.compatible?(%Container{a: 1, b: 2, c: 3}, %Container{a: 4, b: 5, c: 6})
    assert Nx.compatible?(%Container{a: 1, b: 2, c: 3}, %Container{a: 4, b: 5, c: 6.0})
    refute Nx.compatible?(%Container{a: 1, b: 2, c: 3}, %Container{a: 4.0, b: 5.0, c: 6.0})
    refute Nx.compatible?(%Container{a: 1, b: 2, c: 3}, %URI{})
  end

  describe "containers" do
    import Nx.Defn

    alias Nx.Defn.Expr
    alias Nx.Tensor, as: T
    alias Container, as: C
    @default_defn_compiler Nx.Defn.Identity

    defn match_signature(%Container{a: a, b: b}) do
      a + b
    end

    defn match_alias(%C{a: a, b: b}) do
      a + b
    end

    defn match_in_body(var) do
      %C{a: a, b: b} = var
      a + b
    end

    defn dot(var) do
      var.a + var.b
    end

    defn return_struct(x, y) do
      %C{a: x + y, b: x - y}
    end

    defn update_struct(var, x) do
      %C{var | b: x}
    end

    defn update_map(var, x) do
      %{var | b: x}
    end

    defn dot_assert_fields(var) do
      transform(var, &assert_fields!/1)
      var.a + var.b
    end

    defp assert_fields!(%C{c: nil, d: :keep}), do: 1

    test "matches in signature" do
      inp = %Container{a: Nx.tensor(1), b: Nx.tensor(2)}

      assert %T{shape: {}, type: {:s, 64}, data: %Expr{op: :add, args: [left, right]}} =
               match_signature(inp)

      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:s, 64}} = right
    end

    test "matches alias" do
      inp = %Container{a: 1, b: 2}

      assert %T{shape: {}, type: {:s, 64}, data: %Expr{op: :add, args: [left, right]}} =
               match_alias(inp)

      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:s, 64}} = right
    end

    test "matches in body" do
      inp = %Container{a: 1, b: 2}

      assert %T{shape: {}, type: {:s, 64}, data: %Expr{op: :add, args: [left, right]}} =
               match_in_body(inp)

      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:s, 64}} = right
    end

    test "uses dot" do
      inp = %Container{a: 1, b: 2}

      assert %T{shape: {}, type: {:s, 64}, data: %Expr{op: :add, args: [left, right]}} = dot(inp)
      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:s, 64}} = right
    end

    test "can be returned" do
      assert %Container{a: a, b: b} = return_struct(1, 2.0)
      assert %T{shape: {}, type: {:f, 32}, data: %Expr{op: :add, args: [left, right]}} = a
      assert %T{shape: {}, type: {:f, 32}, data: %Expr{op: :subtract, args: [^left, ^right]}} = b
    end

    test "can be updated" do
      inp = %Container{a: 1, b: 2.0}

      assert %Container{a: a, b: b} = update_struct(inp, 8)
      assert %T{shape: {}, type: {:s, 64}, data: %Expr{op: :parameter, args: [0]}} = a
      assert %T{shape: {}, type: {:s, 64}, data: %Expr{op: :parameter, args: [2]}} = b

      assert %Container{a: a, b: b} = update_map(inp, 8)
      assert %T{shape: {}, type: {:s, 64}, data: %Expr{op: :parameter, args: [0]}} = a
      assert %T{shape: {}, type: {:s, 64}, data: %Expr{op: :parameter, args: [2]}} = b
    end

    test "keeps fields" do
      inp = %Container{a: 1, b: 2, c: :reset, d: :keep}

      assert %T{shape: {}, type: {:s, 64}, data: %Expr{op: :add, args: [left, right]}} =
               dot_assert_fields(inp)

      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:s, 64}} = right
    end
  end
end
