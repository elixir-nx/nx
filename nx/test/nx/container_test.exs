defmodule Nx.ContainerTest do
  use ExUnit.Case, async: true

  test "to_template" do
    assert Nx.to_template(%Container{a: 1, b: 2, c: 3, d: 4}) ==
             %Container{a: Nx.template({}, {:s, 64}), b: Nx.template({}, {:s, 64}), c: %{}, d: 4}

    assert Nx.to_template(%Container{a: 1, b: {2, 3.0}, c: 4, d: 5}) ==
             %Container{
               a: Nx.template({}, {:s, 64}),
               b: {Nx.template({}, {:s, 64}), Nx.template({}, {:f, 32})},
               c: %{},
               d: 5
             }
  end

  test "compatible?" do
    assert Nx.compatible?(%Container{a: 1, b: 2, c: 3}, %Container{a: 4, b: 5, c: 6})
    assert Nx.compatible?(%Container{a: 1, b: 2, c: 3}, %Container{a: 4, b: 5, c: 6.0})
    refute Nx.compatible?(%Container{a: 1, b: 2, c: 3}, %Container{a: 4.0, b: 5.0, c: 6.0})
    refute Nx.compatible?(%Container{a: 1, b: 2, c: 3}, %URI{})
  end

  test "backend_transfer" do
    assert Nx.backend_transfer(%Container{a: Nx.tensor(1), b: 2}) ==
             %Container{a: Nx.tensor(1), b: Nx.tensor(2)}
  end

  test "backend_copy" do
    assert Nx.backend_copy(%Container{a: Nx.tensor(1), b: 2}) ==
             %Container{a: Nx.tensor(1), b: Nx.tensor(2)}
  end

  test "backend_deallocate" do
    assert Nx.backend_deallocate(%Container{a: Nx.tensor(1), b: 2}) == :ok
  end

  describe "inside defn" do
    import Nx.Defn

    alias Nx.Defn.Expr
    alias Nx.Tensor, as: T
    alias Container, as: C

    setup do
      Nx.Defn.default_options(compiler: Nx.Defn.Identity)
      :ok
    end

    defn match_signature(%Container{a: a, b: b}) do
      a + b
    end

    test "matches in signature" do
      inp = %Container{a: Nx.tensor(1), b: Nx.tensor(2)}

      assert %T{shape: {}, type: {:s, 64}, data: %Expr{op: :add, args: [left, right]}} =
               match_signature(inp)

      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:s, 64}} = right
    end

    defn match_alias(%C{a: a, b: b}) do
      a + b
    end

    test "matches alias" do
      inp = %Container{a: 1, b: 2}

      assert %T{shape: {}, type: {:s, 64}, data: %Expr{op: :add, args: [left, right]}} =
               match_alias(inp)

      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:s, 64}} = right
    end

    defn match_in_body(var) do
      %C{a: a, b: b} = var
      a + b
    end

    test "matches in body" do
      inp = %Container{a: 1, b: 2}

      assert %T{shape: {}, type: {:s, 64}, data: %Expr{op: :add, args: [left, right]}} =
               match_in_body(inp)

      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:s, 64}} = right
    end

    defn dot(var) do
      var.a + var.b
    end

    test "uses dot" do
      inp = %Container{a: 1, b: 2}

      assert %T{shape: {}, type: {:s, 64}, data: %Expr{op: :add, args: [left, right]}} = dot(inp)
      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:s, 64}} = right
    end

    defn return_struct(x, y) do
      %C{a: x + y, b: x - y}
    end

    test "can be returned" do
      assert %Container{a: a, b: b} = return_struct(1, 2.0)
      assert %T{shape: {}, type: {:f, 32}, data: %Expr{op: :add, args: [left, right]}} = a
      assert %T{shape: {}, type: {:f, 32}, data: %Expr{op: :subtract, args: [^left, ^right]}} = b
    end

    defn update_struct(var, x) do
      %C{var | b: x}
    end

    defn update_map(var, x) do
      %{var | b: x}
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

    defn dot_assert_fields(var) do
      transform(var, &assert_fields!/1)
      var.a + var.b
    end

    defp assert_fields!(%C{c: %{}, d: :keep}), do: 1

    test "keeps fields" do
      inp = %Container{a: 1, b: 2, c: :reset, d: :keep}

      assert %T{shape: {}, type: {:s, 64}, data: %Expr{op: :add, args: [left, right]}} =
               dot_assert_fields(inp)

      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:s, 64}} = right
    end

    defn dot_assert_fields_2(var) do
      transform(var, fn %C{c: %{}, d: %{}} -> 1 end)
      var.a + var.b
    end

    test "keeps empty maps" do
      inp = %Container{a: 1, b: 2, c: :reset, d: %{}}

      assert %T{shape: {}, type: {:s, 64}, data: %Expr{op: :add, args: [left, right]}} =
               dot_assert_fields_2(inp)

      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:s, 64}} = right
    end
  end
end
