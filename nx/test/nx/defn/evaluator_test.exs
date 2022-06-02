defmodule Nx.Defn.EvaluatorTest do
  use ExUnit.Case, async: true

  alias Nx.Tensor, as: T
  import Nx.Defn

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

    assert_raise RuntimeError, "cannot perform operations on a Nx.TemplateBackend tensor", fn ->
      concatenate(Nx.template({3}, {:f, 32}), Nx.template({3}, {:f, 32}))
    end
  end

  defn reshape(t), do: Nx.reshape(t, {3, 2})

  test "reshape" do
    assert %T{shape: {3, 2}, type: {:s, 64}} = reshape(Nx.iota({2, 3}))
  end

  defn reduce_window(t1, acc),
    do: Nx.window_reduce(t1, acc, {2}, [padding: :valid], fn x, acc -> x + acc end)

  test "window reduce" do
    assert reduce_window(Nx.tensor([1, 2, 3]), 0) == Nx.tensor([3, 5])
  end

  describe "decompositions" do
    defn lu(t), do: Nx.LinAlg.lu(t)

    test "lu" do
      assert {p, l, u} = lu(Nx.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
      assert p == Nx.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
      assert l == Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
      assert u == Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    end

    defn qr(t), do: Nx.LinAlg.qr(t)

    test "qr" do
      assert {q, r} = qr(Nx.iota({3, 2}))

      assert q ==
               Nx.tensor([
                 [0.0, 0.9128709435462952],
                 [0.4472135901451111, 0.3651483654975891],
                 [0.8944271802902222, -0.18257418274879456]
               ])

      assert r ==
               Nx.tensor([
                 [4.4721360206604, 5.813776969909668],
                 [0.0, 1.095445156097412]
               ])
    end

    defn svd(t), do: Nx.LinAlg.svd(t)

    test "svd" do
      assert {u, s, vt} = svd(Nx.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
      assert u == Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
      assert s == Nx.tensor([1.0, 1.0, 1.0])
      assert vt == Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    end
  end

  describe "if" do
    defn if3(a, b, c), do: if(a, do: b, else: c)

    test "simple" do
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

    test "with tuples" do
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

    test "with matched tuples" do
      assert if_tuple_match(Nx.tensor(0), Nx.tensor(10), Nx.tensor(20)) == Nx.tensor(200)
      assert if_tuple_match(Nx.tensor(1), Nx.tensor(10), Nx.tensor(20)) == Nx.tensor(-10)
    end

    defn if_tuple_return(a, b, c) do
      {xy, _} = if(a, do: {{a, b}, c}, else: {{c, b}, a})
      xy
    end

    test "with return tuple" do
      assert if_tuple_return(Nx.tensor(0), Nx.tensor(10), Nx.tensor(20)) ==
               {Nx.tensor(20), Nx.tensor(10)}

      assert if_tuple_return(Nx.tensor(1), Nx.tensor(10), Nx.tensor(20)) ==
               {Nx.tensor(1), Nx.tensor(10)}
    end

    defn if_map(a, b, c), do: if(a, do: {%{a: a, b: b, c: 1}, c}, else: {%{a: c, b: b, c: 2}, a})

    test "with map" do
      assert if_map(Nx.tensor(0), Nx.tensor(10), Nx.tensor(20)) ==
               {%{a: Nx.tensor(20), b: Nx.tensor(10), c: Nx.tensor(2)}, Nx.tensor(0)}

      assert if_map(Nx.tensor(1), Nx.tensor(10), Nx.tensor(20)) ==
               {%{a: Nx.tensor(1), b: Nx.tensor(10), c: Nx.tensor(1)}, Nx.tensor(20)}

      assert if_map(Nx.tensor(0), Nx.tensor(10), Nx.tensor([20, 30])) ==
               {%{a: Nx.tensor([20, 30]), b: Nx.tensor(10), c: Nx.tensor(2)}, Nx.tensor([0, 0])}

      assert if_map(Nx.tensor(1), Nx.tensor(10), Nx.tensor([20, 30])) ==
               {%{a: Nx.tensor([1, 1]), b: Nx.tensor(10), c: Nx.tensor(1)}, Nx.tensor([20, 30])}
    end

    defn if_map_match(a, b, c) do
      {%{a: x, b: y}, z} = if(a, do: {%{a: a, b: b}, c}, else: {%{a: c, b: b}, a})
      x * y - z
    end

    test "with matched map" do
      assert if_map_match(Nx.tensor(0), Nx.tensor(10), Nx.tensor(20)) == Nx.tensor(200)
      assert if_map_match(Nx.tensor(1), Nx.tensor(10), Nx.tensor(20)) == Nx.tensor(-10)
    end
  end

  describe "while/3" do
    defn upto10(x) do
      while x, Nx.less(x, 10) do
        x + 1
      end
    end

    test "simple" do
      assert upto10(0) == Nx.tensor(10)
      assert upto10(5) == Nx.tensor(10)
    end

    defn factorial_tuple(x) do
      factorial = Nx.tensor(1, type: Nx.type(x))

      {factorial, _} =
        while {factorial, x}, Nx.greater(x, 1) do
          {factorial * x, x - 1}
        end

      factorial
    end

    test "factorial tuple" do
      assert factorial_tuple(5) == Nx.tensor(120)
      assert factorial_tuple(10.0) == Nx.tensor(3_628_800.0)
    end

    defn factorial_map(x) do
      factorial = Nx.tensor(1, type: Nx.type(x))

      %{factorial: factorial} =
        while map = %{factorial: factorial, x: x}, Nx.greater(map.x, 1) do
          %{map | factorial: map.factorial * map.x, x: map.x - 1}
        end

      factorial
    end

    test "factorial map" do
      assert factorial_map(5) == Nx.tensor(120)
      assert factorial_map(10.0) == Nx.tensor(3_628_800.0)
    end

    defn factorial_map_input(map) do
      %{factorial: factorial} =
        while map, Nx.greater(map.x, 1) do
          %{map | factorial: map.factorial * map.x, x: map.x - 1}
        end

      factorial
    end

    test "factorial map input" do
      assert factorial_map_input(%{factorial: 1, x: 5}) == Nx.tensor(120)
      assert factorial_map_input(%{factorial: 1.0, x: 10.0}) == Nx.tensor(3_628_800.0)
    end
  end

  describe "argsort/2" do
    defn argsort(x), do: Nx.argsort(x)

    test "simple" do
      t = Nx.tensor([3, 1, 2])
      assert argsort(t) == Nx.tensor([1, 2, 0])
    end
  end

  describe "sort/2" do
    defn sort(x), do: Nx.sort(x)

    test "simple" do
      t = Nx.tensor([3, 1, 2])
      assert sort(t) == Nx.tensor([1, 2, 3])
    end
  end

  describe "anonymous functions" do
    defn calls_binary_fun(fun, a, b), do: fun.(a, b)

    test "calls external anonymous function directly" do
      assert calls_binary_fun(&Nx.add/2, 1, 2.0) == Nx.tensor(3.0)
    end

    defn calls_reduce_fun(fun, t), do: Nx.reduce(t, 0, fun)

    test "calls external anonymous function via reduce" do
      assert calls_reduce_fun(&Nx.add/2, Nx.tensor([1, 2, 3])) == Nx.tensor(6)
    end

    defn calls_map_fun(t) do
      Nx.map(t, fn x ->
        if Nx.equal(x, 0), do: 1, else: -x
      end)
    end

    test "calls internal anonymous function via map" do
      assert calls_map_fun(Nx.tensor([0, 1, 2])) == Nx.tensor([1, -1, -2])
    end
  end

  describe "access" do
    defn slice1(t), do: t[1][0]

    test "supports correct access" do
      assert slice1(Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])) == Nx.tensor(4)
    end
  end

  defn container_cond(var, cond) do
    if cond do
      %{var | a: var.a + 1}
    else
      %{var | b: var.b - 1}
    end
  end

  describe "containers" do
    test "input, output, and cond" do
      container = %Container{a: 1, b: -1, c: :reset, d: :kept}

      assert container_cond(container, 1) ==
               %Container{a: Nx.tensor(2), b: Nx.tensor(-1), c: %{}, d: :kept}

      assert container_cond(container, 0) ==
               %Container{a: Nx.tensor(1), b: Nx.tensor(-2), c: %{}, d: :kept}
    end
  end

  defn labelled_inspect(a, b), do: inspect_value(a + b, label: "add")

  test "inspect_value/2" do
    assert ExUnit.CaptureIO.capture_io(fn -> labelled_inspect(1, 2) end) ==
             """
             add: #Nx.Tensor<
               s64
               3
             >
             """
  end

  describe "hooks" do
    defp send_to_self(value), do: send(self(), value)

    defn basic_hook(a, b), do: hook(a + b, :example, &send_to_self({:default, &1}))

    test "basic hook with overriddes" do
      assert basic_hook(1, 2) == Nx.tensor(3)
      assert_received {:default, tensor}
      assert tensor == Nx.tensor(3)

      assert Nx.Defn.jit(&basic_hook/2, [1, 2]) == Nx.tensor(3)
      assert_received {:default, tensor}
      assert tensor == Nx.tensor(3)

      assert Nx.Defn.jit(&basic_hook/2, [1, 2], hooks: %{example: &send_to_self({:custom, &1})}) ==
               Nx.tensor(3)

      assert_received {:custom, tensor}
      assert tensor == Nx.tensor(3)
    end

    defn container_hook(a, b), do: hook({a, b}, :example, &send_to_self({:default, &1}))

    test "container hook with overriddes" do
      assert container_hook(1, 2) == {Nx.tensor(1), Nx.tensor(2)}
      assert_received {:default, tuple}
      assert tuple == {Nx.tensor(1), Nx.tensor(2)}

      assert Nx.Defn.jit(&container_hook/2, [1, 2]) == {Nx.tensor(1), Nx.tensor(2)}
      assert_received {:default, tuple}
      assert tuple == {Nx.tensor(1), Nx.tensor(2)}

      assert Nx.Defn.jit(&container_hook/2, [1, 2],
               hooks: %{example: &send_to_self({:custom, &1})}
             ) == {Nx.tensor(1), Nx.tensor(2)}

      assert_received {:custom, tuple}
      assert tuple == {Nx.tensor(1), Nx.tensor(2)}
    end

    defn side_effect_hooks(a, b) do
      token = create_token()
      {token, _} = hook_token(token, b, :b)
      {token, _} = hook_token(token, a, :a)
      attach_token(token, {a, b})
    end

    test "side effect hooks" do
      side_effect_hooks(1, 2)
      refute_received _

      hooks = %{a: &send_to_self({:a, &1})}
      Nx.Defn.jit(&side_effect_hooks/2, [1, 2], hooks: hooks)
      assert_received {:a, tensor}
      assert tensor == Nx.tensor(1)
      refute_received _

      hooks = %{b: &send_to_self({:b, &1})}
      Nx.Defn.jit(&side_effect_hooks/2, [1, 2], hooks: hooks)
      assert_received {:b, tensor}
      assert tensor == Nx.tensor(2)
      refute_received _

      hooks = %{a: &send_to_self({:a, &1}), b: &send_to_self({:b, &1})}
      Nx.Defn.jit(&side_effect_hooks/2, [1, 2], hooks: hooks)
      {:messages, [b: _, a: _]} = Process.info(self(), :messages)
      assert_received {:b, tensor}
      assert tensor == Nx.tensor(2)
      assert_received {:a, tensor}
      assert tensor == Nx.tensor(1)
    end

    defn side_effect_nested_hooks(a, b) do
      token = create_token()
      {token, _} = hook_token(token, b, :b)
      a = attach_token(token, a)
      hook(a, :a)
    end

    test "side effect nested hooks" do
      side_effect_nested_hooks(1, 2)
      refute_received _

      hooks = %{a: &send_to_self({:a, &1})}
      Nx.Defn.jit(&side_effect_nested_hooks/2, [1, 2], hooks: hooks)
      assert_received {:a, tensor}
      assert tensor == Nx.tensor(1)
      refute_received _

      hooks = %{b: &send_to_self({:b, &1})}
      Nx.Defn.jit(&side_effect_nested_hooks/2, [1, 2], hooks: hooks)
      assert_received {:b, tensor}
      assert tensor == Nx.tensor(2)
      refute_received _

      hooks = %{a: &send_to_self({:a, &1}), b: &send_to_self({:b, &1})}
      Nx.Defn.jit(&side_effect_nested_hooks/2, [1, 2], hooks: hooks)
      {:messages, [b: _, a: _]} = Process.info(self(), :messages)
      assert_received {:b, tensor}
      assert tensor == Nx.tensor(2)
      assert_received {:a, tensor}
      assert tensor == Nx.tensor(1)
    end

    defn side_effect_nested_hook_with_default(a, b) do
      token = create_token()
      {token, _} = hook_token(token, b, :b, &send_to_self({:b, &1}))
      a = attach_token(token, a)
      hook(a, :a)
    end

    test "side effect nested hooks with default" do
      side_effect_nested_hook_with_default(1, 2)
      assert_received {:b, tensor}
      assert tensor == Nx.tensor(2)

      hooks = %{a: &send_to_self({:a, &1})}
      Nx.Defn.jit(&side_effect_nested_hook_with_default/2, [1, 2], hooks: hooks)
      {:messages, [b: _, a: _]} = Process.info(self(), :messages)
      assert_received {:b, tensor}
      assert tensor == Nx.tensor(2)
      assert_received {:a, tensor}
      assert tensor == Nx.tensor(1)

      hooks = %{b: &send_to_self({:custom, &1})}
      Nx.Defn.jit(&side_effect_nested_hook_with_default/2, [1, 2], hooks: hooks)
      assert_received {:custom, tensor}
      assert tensor == Nx.tensor(2)

      refute_received _
    end

    defn hook_upto10(x) do
      while x, Nx.less(x, 10) do
        hook(x + 1, :while)
      end
    end

    test "inside loops" do
      assert hook_upto10(5) == Nx.tensor(10)
      refute_received _

      assert Nx.Defn.jit(&hook_upto10/1, [5], hooks: %{while: &send_to_self({:while, &1})}) ==
               Nx.tensor(10)

      assert_received {:while, tensor}
      assert tensor == Nx.tensor(6)
      assert_received {:while, tensor}
      assert tensor == Nx.tensor(7)
      assert_received {:while, tensor}
      assert tensor == Nx.tensor(8)
      assert_received {:while, tensor}
      assert tensor == Nx.tensor(9)
      assert_received {:while, tensor}
      assert tensor == Nx.tensor(10)
      refute_received _
    end
  end
end
