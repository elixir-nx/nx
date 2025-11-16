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
    assert %T{shape: {2, 2}, type: {:s, 32}} = iota()
  end

  defn concatenate(a, b), do: Nx.concatenate([a, b])

  test "concatenate" do
    assert concatenate(Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])) ==
             Nx.tensor([1, 2, 3, 4, 5, 6])

    assert_raise RuntimeError, "cannot perform operations on a Nx.TemplateBackend tensor", fn ->
      # Check we will pick Nx.Template from list
      concatenate(Nx.template({3}, {:f, 32}), Nx.template({3}, {:f, 32}))
    end
  end

  defn slice(a, b), do: Nx.slice(a, [b], [1])

  test "slice" do
    assert slice(Nx.tensor([1, 2, 3]), Nx.tensor(0)) == Nx.tensor([1])

    assert_raise RuntimeError, "cannot perform operations on a Nx.TemplateBackend tensor", fn ->
      # Check we will pick Nx.Template from the slice
      slice(Nx.tensor([1, 2, 3]), Nx.template({}, {:s, 32}))
    end
  end

  defn reshape(t), do: Nx.reshape(t, {3, 2})

  test "reshape" do
    assert %T{shape: {3, 2}, type: {:s, 32}} = reshape(Nx.iota({2, 3}))
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
                 [0.4472135901451111, 0.3651483356952667],
                 [0.8944271802902222, -0.1825741082429886]
               ])

      assert r ==
               Nx.tensor([
                 [4.4721360206604, 5.81377649307251],
                 [2.433349663988338e-7, 1.0954453945159912]
               ])
    end

    defn svd(t), do: Nx.LinAlg.svd(t)

    test "svd" do
      assert {u, s, vt} = svd(Nx.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
      assert u == Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
      assert s == Nx.tensor([1.0, 1.0, 1.0])
      assert vt == Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
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

  defn labelled_inspect(a, b), do: print_value(a + b, label: "add")

  test "print_value/2 with options" do
    assert ExUnit.CaptureIO.capture_io(fn -> labelled_inspect(1, 2) end) ==
             """
             add: #Nx.Tensor<
               s32
               3
             >
             """
  end

  defn mapped_inspect(a, b), do: print_value(a + b, fn x -> x * 2 end)

  test "print_value/2 with mapping function" do
    assert ExUnit.CaptureIO.capture_io(fn ->
             assert mapped_inspect(1, 2) == Nx.tensor(3)
           end) ==
             """
             #Nx.Tensor<
               s32
               6
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

      assert Nx.Defn.jit(&basic_hook/2).(1, 2) == Nx.tensor(3)
      assert_received {:default, tensor}
      assert tensor == Nx.tensor(3)

      fun = Nx.Defn.jit(&basic_hook/2, hooks: %{example: &send_to_self({:custom, &1})})
      assert fun.(1, 2) == Nx.tensor(3)

      assert_received {:custom, tensor}
      assert tensor == Nx.tensor(3)
    end

    defn container_hook(a, b), do: hook({a, b}, :example, &send_to_self({:default, &1}))

    test "container hook with overriddes" do
      assert container_hook(1, 2) == {Nx.tensor(1), Nx.tensor(2)}
      assert_received {:default, tuple}
      assert tuple == {Nx.tensor(1), Nx.tensor(2)}

      assert Nx.Defn.jit(&container_hook/2).(1, 2) == {Nx.tensor(1), Nx.tensor(2)}
      assert_received {:default, tuple}
      assert tuple == {Nx.tensor(1), Nx.tensor(2)}

      fun = Nx.Defn.jit(&container_hook/2, hooks: %{example: &send_to_self({:custom, &1})})
      assert fun.(1, 2) == {Nx.tensor(1), Nx.tensor(2)}

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
      Nx.Defn.jit(&side_effect_hooks/2, hooks: hooks).(1, 2)
      assert_received {:a, tensor}
      assert tensor == Nx.tensor(1)
      refute_received _

      hooks = %{b: &send_to_self({:b, &1})}
      Nx.Defn.jit(&side_effect_hooks/2, hooks: hooks).(1, 2)
      assert_received {:b, tensor}
      assert tensor == Nx.tensor(2)
      refute_received _

      hooks = %{a: &send_to_self({:a, &1}), b: &send_to_self({:b, &1})}
      Nx.Defn.jit(&side_effect_hooks/2, hooks: hooks).(1, 2)
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
      Nx.Defn.jit(&side_effect_nested_hooks/2, hooks: hooks).(1, 2)
      assert_received {:a, tensor}
      assert tensor == Nx.tensor(1)
      refute_received _

      hooks = %{b: &send_to_self({:b, &1})}
      Nx.Defn.jit(&side_effect_nested_hooks/2, hooks: hooks).(1, 2)
      assert_received {:b, tensor}
      assert tensor == Nx.tensor(2)
      refute_received _

      hooks = %{a: &send_to_self({:a, &1}), b: &send_to_self({:b, &1})}
      Nx.Defn.jit(&side_effect_nested_hooks/2, hooks: hooks).(1, 2)
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
      Nx.Defn.jit(&side_effect_nested_hook_with_default/2, hooks: hooks).(1, 2)
      {:messages, [b: _, a: _]} = Process.info(self(), :messages)
      assert_received {:b, tensor}
      assert tensor == Nx.tensor(2)
      assert_received {:a, tensor}
      assert tensor == Nx.tensor(1)

      hooks = %{b: &send_to_self({:custom, &1})}
      Nx.Defn.jit(&side_effect_nested_hook_with_default/2, hooks: hooks).(1, 2)
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

      assert Nx.Defn.jit(&hook_upto10/1, hooks: %{while: &send_to_self({:while, &1})}).(5) ==
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

  describe "cond cache" do
    # The goal of those tests is to show that expressions inside cond are cached,
    # regardless of evaluation order.
    defn cond_cache_left(bool, a, b) do
      res = hook(a + b, :example, &send_to_self({:hook, &1}))

      cond =
        if bool do
          res
        else
          0
        end

      cond * res
    end

    test "on lhs" do
      assert cond_cache_left(0, 1, 2) == Nx.tensor(0)
      assert_received {:hook, _}
      refute_received {:hook, _}

      assert cond_cache_left(1, 1, 2) == Nx.tensor(9)
      assert_received {:hook, _}
      refute_received {:hook, _}
    end

    defn cond_cache_right(bool, a, b) do
      res = hook(a + b, :example, &send_to_self({:hook, &1}))

      cond =
        if bool do
          res
        else
          0
        end

      res * cond
    end

    test "on rhs" do
      assert cond_cache_right(0, 1, 2) == Nx.tensor(0)
      assert_received {:hook, _}
      refute_received {:hook, _}

      assert cond_cache_right(1, 1, 2) == Nx.tensor(9)
      assert_received {:hook, _}
      refute_received {:hook, _}
    end

    defn cond_cache_both(bool, a, b) do
      res = hook(a + b, :example, &send_to_self({:hook, &1}))

      left =
        if bool do
          res
        else
          -res
        end

      right =
        if bool do
          res * 2
        else
          res
        end

      left * right
    end

    test "on both" do
      assert cond_cache_both(0, 4, 5) == Nx.tensor(-81)
      assert_received {:hook, _}
      refute_received {:hook, _}

      assert cond_cache_both(1, 4, 5) == Nx.tensor(162)
      assert_received {:hook, _}
      refute_received {:hook, _}
    end

    defn cond_cache_map(state) do
      state =
        if state.iteration < 4 do
          if state.iteration != 1 do
            state
          else
            state
          end
        else
          state
        end

      %{state | iteration: state.iteration + 1}
    end

    test "with nested map" do
      assert cond_cache_map(%{iteration: 0}) ==
               %{iteration: Nx.tensor(1)}

      # Use :abc/:xyz so we try different key orderings.
      assert cond_cache_map(%{iteration: 0, abc: 1}) ==
               %{iteration: Nx.tensor(1), abc: Nx.tensor(1)}

      assert cond_cache_map(%{iteration: 0, xyz: 1}) ==
               %{iteration: Nx.tensor(1), xyz: Nx.tensor(1)}

      assert cond_cache_map(%{iteration: 4}) ==
               %{iteration: Nx.tensor(5)}

      # Use :abc/:xyz so we try different key orderings.
      assert cond_cache_map(%{iteration: 4, abc: 1}) ==
               %{iteration: Nx.tensor(5), abc: Nx.tensor(1)}

      assert cond_cache_map(%{iteration: 4, xyz: 1}) ==
               %{iteration: Nx.tensor(5), xyz: Nx.tensor(1)}
    end

    defn cond_nested_condition_cache(state) do
      {state, prev} =
        if state.iteration < 12 do
          sign = if(state.iteration < 2, do: 1, else: -1)

          factor =
            if sign >= 0 do
              2
            else
              1
            end

          {state, factor}
        else
          {state, 1}
        end

      %{state | iteration: state.iteration + prev}
    end

    test "with nested condition" do
      assert cond_nested_condition_cache(%{iteration: 0}) == %{iteration: Nx.tensor(2)}
      assert cond_nested_condition_cache(%{iteration: 12}) == %{iteration: Nx.tensor(13)}
    end
  end

  describe "vectorization" do
    defn vectorize_within_defn(t, a) do
      t =
        t
        |> Nx.vectorize(a: 1, b: 2)
        |> Nx.add(1)

      a = Nx.vectorize(a, a: 1)

      Nx.select(a, t, a)
    end

    test "vectorize works inside defn" do
      t = Nx.tensor([[1, 2]])

      assert vectorize_within_defn(t, Nx.tensor([1])) == Nx.vectorize(Nx.add(t, 1), a: 1, b: 2)

      assert vectorize_within_defn(t, Nx.tensor([0])) ==
               Nx.vectorize(Nx.tensor([[0, 0]]), a: 1, b: 2)
    end

    defn vectorized_while(a, b, i \\ 0) do
      while({a, b, i}, i < 2) do
        {a + b, b, i + 1}
      end
    end

    defn vectorized_cond(cond_1, on_1, cond_2, on_2, on_3) do
      cond do
        cond_1 -> on_1
        cond_2 -> on_2
        true -> on_3
      end
    end

    defn vectorized_metadata(t) do
      x = Nx.devectorize(t)
      x = Nx.vectorize(x, t.vectorized_axes)

      stop_grad(x)
    end

    test "while" do
      t = Nx.iota({2, 3}, vectorized_axes: [a: 1])

      assert {result, one, i} = vectorized_while(t, 1)

      assert result == Nx.add(t, 2)
      assert one == Nx.tensor(1)
      assert i == Nx.tensor(2)
    end

    test "while raises on incompatible body/initial" do
      t = Nx.iota({2, 3}, vectorized_axes: [a: 1], type: :s32)

      message = """
      test/nx/defn/evaluator_test.exs:650: the do-block in while must return tensors with the same shape, type, and names as the initial arguments.

      {\e[32m
       <<<<< Body (do-block) <<<<<
       #Nx.Tensor<
         vectorized[a: 2]
         s32[2][3]
       >
       ==========
       \e[31m#Nx.Tensor<
         vectorized[a: 1]
         s32[2][3]
       >
       >>>>>     Initial     >>>>>
       \e[0m, #Nx.Tensor<
         vectorized[a: 2]
         s32[2][3]
       >, #Nx.Tensor<
         s32
       >}
      """

      assert_raise CompileError, message, fn ->
        vectorized_while(t, Nx.iota({2, 3}, vectorized_axes: [a: 2], type: :s32))
      end
    end

    test "while raises on vectorized condition" do
      t = Nx.iota({2, 3}, vectorized_axes: [a: 1])

      error =
        """
        test/nx/defn/evaluator_test.exs:650: condition must be a scalar tensor, got: #Nx.Tensor<
          vectorized[x: 1]
          u8[1]
        \s\s
          Nx.Defn.Expr
          parameter a:2   s32[1][1]
          b = reshape 2   s32[1][1]
          c = less a, b   u8[1][1]
        >, consider using Nx.all/1 or Nx.any/1 to obtain a scalar predicate from tensor
        """
        |> String.trim()

      assert_raise CompileError, error, fn ->
        vectorized_while(t, t, Nx.iota({1}, vectorized_axes: [x: 1]))
      end
    end

    test "cond" do
      a = Nx.tensor(3)
      b = Nx.tensor(2)
      c = Nx.vectorize(Nx.tensor([[1]]), x: 1, y: 1)
      d = Nx.vectorize(Nx.tensor([[1, 2]]), x: 1, y: 2)

      assert {c, c} == vectorized_cond(0, {c, c}, 0, {c, c}, {c, c})
      assert Nx.add(c, 2) == vectorized_cond(1, a, 0, b, c)
      assert Nx.add(c, 1) == vectorized_cond(0, a, 1, b, c)

      # vectorization edge cases

      [c_vec_d, _] = Nx.broadcast_vectors([c, d])

      assert c_vec_d == vectorized_cond(0, a, 1, c, d)
      assert d == vectorized_cond(0, a, 0, c, d)
    end

    test "metadata with expr" do
      a = Nx.iota({1}, vectorized_axes: [x: 2])
      assert a == vectorized_metadata(a)
    end
  end

  defn debug_test_fun(x, y) do
    a = Nx.add(x, y)
    b = Nx.multiply(a, 2)
    Nx.subtract(b, 1)
  end

  defn reuse_fun(x) do
    a = Nx.add(x, 1)
    Nx.add(a, a)
  end

  def debug_test_fun(x, y, opts), do: Nx.Defn.jit(&debug_test_fun/2, opts).(x, y)
  def reuse_fun(x, opts), do: Nx.Defn.jit(&reuse_fun/1, opts).(x)

  describe "debug_options" do
    test "prints node info to stdout" do
      x = Nx.tensor([1, 2])
      y = Nx.tensor([3, 4])
      opts = [compiler: Nx.Defn.Evaluator, debug_options: [inspect_limit: 5]]
      output = ExUnit.CaptureIO.capture_io(fn -> debug_test_fun(x, y, opts) end)

      node_id_regex = ~r/node_id = \"(.*)\"/

      assert [id0, id1, id2, id3, id4] =
               Regex.scan(node_id_regex, output, capture: :all_but_first)

      output =
        output
        |> String.replace(id0, "::id0::")
        |> String.replace(id1, "::id1::")
        |> String.replace(id2, "::id2::")
        |> String.replace(id3, "::id3::")
        |> String.replace(id4, "::id4::")

      assert output == ~S"""
             node_id = "::id0::"
             operation = :parameter

             args = [
               "0"
             ]

             # Result:
             result = Nx.from_binary(<<1, 0, 0, 0, 2, 0, 0, 0>>, {:s, 32}, backend: {Nx.BinaryBackend, []}) |> Nx.reshape({2})

             node_id = "::id1::"
             operation = :parameter

             args = [
               "1"
             ]

             # Result:
             result = Nx.from_binary(<<3, 0, 0, 0, 4, 0, 0, 0>>, {:s, 32}, backend: {Nx.BinaryBackend, []}) |> Nx.reshape({2})

             node_id = "::id2::"
             operation = :add

             args = [
               "#Nx.Tensor<\n  s32[2]\n  \n  Nx.Defn.Expr<::id0::>\n  parameter a:0   s32[2]\n>",
               "#Nx.Tensor<\n  s32[2]\n  \n  Nx.Defn.Expr<::id1::>\n  parameter a:1   s32[2]\n>"
             ]

             # Result:
             result = Nx.from_binary(<<4, 0, 0, 0, 6, 0, 0, 0>>, {:s, 32}, backend: {Nx.BinaryBackend, []}) |> Nx.reshape({2})

             node_id = "::id3::"
             operation = :multiply

             args = [
               "#Nx.Tensor<\n  s32\n  \n  Nx.Defn.Expr\n  2\n>",
               "#Nx.Tensor<\n  s32[2]\n  \n  Nx.Defn.Expr<::id2::>\n  parameter a:0   s32[2]\n  parameter b:1   s32[2]\n  c = add a, b    s32[2]\n>"
             ]

             # Result:
             result = Nx.from_binary(<<8, 0, 0, 0, 12, 0, 0, 0>>, {:s, 32}, backend: {Nx.BinaryBackend, []}) |> Nx.reshape({2})

             node_id = "::id4::"
             operation = :subtract

             args = [
               "#Nx.Tensor<\n  s32[2]\n  \n  Nx.Defn.Expr<::id3::>\n  parameter a:0       s32[2]\n  parameter b:1       s32[2]\n  c = add a, b        s32[2]\n  d = multiply 2, c   s32[2]\n>",
               "#Nx.Tensor<\n  s32\n  \n  Nx.Defn.Expr\n  1\n>"
             ]

             # Result:
             result = Nx.from_binary(<<7, 0, 0, 0, 11, 0, 0, 0>>, {:s, 32}, backend: {Nx.BinaryBackend, []}) |> Nx.reshape({2})

             """
    end

    test "saves node info to files" do
      x = Nx.tensor([1, 2])
      y = Nx.tensor([3, 4])
      tmp_dir = Path.join(System.tmp_dir!(), "nx_debug_test_#{System.unique_integer()}")
      on_exit(fn -> File.rm_rf!(tmp_dir) end)
      opts = [compiler: Nx.Defn.Evaluator, debug_options: [inspect_limit: 5, save_path: tmp_dir]]

      debug_test_fun(x, y, opts)
      files = File.ls!(tmp_dir)
      assert Enum.any?(files, &String.starts_with?(&1, "node_"))
      contents = Enum.map(files, &File.read!(Path.join(tmp_dir, &1)))
      assert {[_], rest} = Enum.split_with(contents, &(&1 =~ "operation = :add"))
      assert {[_], rest} = Enum.split_with(rest, &(&1 =~ "operation = :multiply"))
      assert {[_], rest} = Enum.split_with(rest, &(&1 =~ "operation = :subtract"))
      assert length(rest) == 2
    end

    test "node info for reused node only once" do
      x = Nx.tensor([1, 2])
      opts = [compiler: Nx.Defn.Evaluator, debug_options: [inspect_limit: 5]]
      output = ExUnit.CaptureIO.capture_io(fn -> reuse_fun(x, opts) end)

      node_id_regex = ~r/node_id = (.*)/

      assert [id0, id1, id2, id3] =
               Regex.scan(node_id_regex, output, capture: :all_but_first)

      output =
        output
        |> String.replace(id0, "::id0::")
        |> String.replace(id1, "::id1::")
        |> String.replace(id2, "::id2::")
        |> String.replace(id3, "::id3::")

      # ensure that each node id is printed exactly once
      assert output =~ ~r/.*(?:node_id = ::id0::){1}.*/
      assert output =~ ~r/.*(?:node_id = ::id1::){1}.*/
      assert output =~ ~r/.*(?:node_id = ::id2::){1}.*/
      assert output =~ ~r/.*(?:node_id = ::id3::){1}.*/
    end

    test "respects inspect_limit" do
      x = Nx.tensor(Enum.to_list(1..20))
      y = Nx.tensor(Enum.to_list(21..40))
      opts = [compiler: Nx.Defn.Evaluator, debug_options: [inspect_limit: 2]]
      output = ExUnit.CaptureIO.capture_io(fn -> debug_test_fun(x, y, opts) end)

      assert output =~ "..."
    end

    test "does nothing when feature is disabled" do
      x = Nx.tensor([1, 2])
      y = Nx.tensor([3, 4])
      output = ExUnit.CaptureIO.capture_io(fn -> debug_test_fun(x, y) end)
      assert output == ""
    end
  end
end
