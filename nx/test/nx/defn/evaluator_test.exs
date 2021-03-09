defmodule Nx.Defn.EvaluatorTest do
  use ExUnit.Case, async: true

  alias Nx.Tensor, as: T
  import Nx.Defn

  @defn_compiler Nx.Defn.Evaluator
  defn add(a, b), do: {a + b, a - b}

  # Check the attribute has been reset
  nil = Module.get_attribute(__MODULE__, :defn_compiler)

  test "can be set explicitly set" do
    assert add(1, 2) == {Nx.tensor(3), Nx.tensor(-1)}
  end

  test "is the default compiler" do
    defmodule DefaultCompiler do
      import Nx.Defn
      defn add(a, b), do: a + b
    end

    assert DefaultCompiler.add(1, 2) == Nx.tensor(3)
  end

  defn add_two_int(t), do: Nx.add(t, 2)
  defn add_two_float(t), do: Nx.add(t, 2)

  test "constant" do
    assert %T{shape: {3}, type: {:u, 8}} = add_two_int(Nx.tensor([1, 2, 3], type: {:u, 8}))

    assert %T{shape: {3}, type: {:bf, 16}} = add_two_float(Nx.tensor([1, 2, 3], type: {:bf, 16}))
  end

  defn iota(), do: Nx.iota({2, 2})

  test "iota" do
    assert %T{shape: {2, 2}, type: {:s, 64}} = iota()
  end

  defn concatenate(a, b), do: Nx.concatenate([a, b])

  test "concatenate" do
    assert concatenate(Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])) ==
             Nx.tensor([1, 2, 3, 4, 5, 6])
  end

  defn reshape(t), do: Nx.reshape(t, {3, 2})

  test "reshape" do
    assert %T{shape: {3, 2}, type: {:s, 64}} = reshape(Nx.iota({2, 3}))
  end

  defn lu(t), do: Nx.lu(t)

  test "lu" do
    assert {p, l, u} = lu(Nx.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
    assert p == Nx.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert l == Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert u == Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
  end

  defn qr(t), do: Nx.qr(t)

  test "qr" do
    assert {q, r} = qr(Nx.iota({3, 2}))
    assert q == Nx.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    assert r == Nx.tensor([[2.0, 3.0], [0.0, 1.0]])
  end

  defn svd(t), do: Nx.svd(t)

  test "svd" do
    assert {u, s, vt} = svd(Nx.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
    assert u == Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert s == Nx.tensor([1.0, 1.0, 1.0])
    assert vt == Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
  end

  defn if3(a, b, c), do: if(a, do: b, else: c)

  test "if" do
    assert if3(Nx.tensor(0), Nx.tensor(1, type: {:s, 16}), Nx.tensor(2, type: {:f, 32})) ==
             Nx.tensor(2, type: {:f, 32})

    assert if3(Nx.tensor(1), Nx.tensor(1, type: {:s, 16}), Nx.tensor(2, type: {:f, 32})) ==
             Nx.tensor(1, type: {:f, 32})

    assert if3(Nx.tensor(2), Nx.tensor(1, type: {:s, 16}), Nx.tensor(2, type: {:f, 32})) ==
             Nx.tensor(1, type: {:f, 32})

    assert if3(Nx.tensor(0), Nx.tensor([1, 2]), Nx.tensor([[3], [4]])) ==
             Nx.tensor([[3, 3], [4, 4]])

    assert if3(Nx.tensor(1), Nx.tensor([1, 2]), Nx.tensor([[3], [4]])) ==
             Nx.tensor([[1, 2], [1, 2]])
  end

  defn if_tuple(a, b, c), do: if(a, do: {{a, b}, c}, else: {{c, b}, a})

  test "if with tuples" do
    assert if_tuple(Nx.tensor(0), Nx.tensor(10), Nx.tensor(20)) ==
             {{Nx.tensor(20), Nx.tensor(10)}, Nx.tensor(0)}

    assert if_tuple(Nx.tensor(1), Nx.tensor(10), Nx.tensor(20)) ==
             {{Nx.tensor(1), Nx.tensor(10)}, Nx.tensor(20)}

    assert if_tuple(Nx.tensor(0), Nx.tensor(10), Nx.tensor([20, 30])) ==
             {{Nx.tensor([20, 30]), Nx.tensor(10)}, Nx.tensor([0, 0])}

    assert if_tuple(Nx.tensor(1), Nx.tensor(10), Nx.tensor([20, 30])) ==
             {{Nx.tensor([1, 1]), Nx.tensor(10)}, Nx.tensor([20, 30])}
  end

  defn if_tuple_match(a, b, c) do
    {{x, y}, z} = if(a, do: {{a, b}, c}, else: {{c, b}, a})
    x * y - z
  end

  test "if with matched tuples" do
    assert if_tuple_match(Nx.tensor(0), Nx.tensor(10), Nx.tensor(20)) == Nx.tensor(200)
    assert if_tuple_match(Nx.tensor(1), Nx.tensor(10), Nx.tensor(20)) == Nx.tensor(-10)
  end

  defn if_tuple_return(a, b, c) do
    {xy, _} = if(a, do: {{a, b}, c}, else: {{c, b}, a})
    xy
  end

  test "if with return tuple" do
    assert if_tuple_return(Nx.tensor(0), Nx.tensor(10), Nx.tensor(20)) ==
             {Nx.tensor(20), Nx.tensor(10)}

    assert if_tuple_return(Nx.tensor(1), Nx.tensor(10), Nx.tensor(20)) ==
             {Nx.tensor(1), Nx.tensor(10)}
  end
end
