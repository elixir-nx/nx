defmodule Nx.DefnTest do
  use ExUnit.Case, async: true

  alias Nx.Tensor, as: T
  alias Nx.Defn.{Expr, Identity, Evaluator}
  alias Nx.DefnTest.Sample

  import Nx.Defn
  doctest Nx.Defn

  defmacrop location(plus) do
    file = Path.relative_to_cwd(__CALLER__.file)
    quote do: "#{unquote(file)}:#{unquote(__CALLER__.line) + unquote(plus)}"
  end

  @default_defn_compiler Identity

  describe "constants" do
    @tensor [1, 2, 3]
    defn list_constant, do: Nx.tensor(@tensor)

    test "from list" do
      assert %T{data: %Expr{op: :tensor}} = list_constant()
    end

    @tensor Nx.to_binary(Nx.tensor([1, 2, 3]))
    defn binary_constant, do: Nx.from_binary(@tensor, {:s, 64})

    test "from binary" do
      assert %T{data: %Expr{op: :tensor}} = binary_constant()
    end
  end

  describe "tuple" do
    defn tuple_shape_match_signature({a, b}) do
      a + b
    end

    test "allows pattern matching on the tuple shape on signature" do
      assert %T{shape: {}, type: {:f, 32}, data: %Expr{op: :add, args: [left, right]}} =
               tuple_shape_match_signature({1, 2.0})

      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:f, 32}} = right
    end

    defn tuple_shape_match_alias({_, _} = var) do
      {a, b} = var
      a + b
    end

    test "allows pattern matching on the tuple shape with alias and underscores" do
      assert %T{shape: {}, type: {:f, 32}, data: %Expr{op: :add, args: [left, right]}} =
               tuple_shape_match_alias({1, 2.0})

      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:f, 32}} = right
    end

    defn tuple_shape_match_inside_body(var) do
      {a, b} = var
      a + b
    end

    test "allows pattern matching on the tuple shape inside body" do
      assert %T{shape: {}, type: {:f, 32}, data: %Expr{op: :add, args: [left, right]}} =
               tuple_shape_match_inside_body({1, 2.0})

      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:f, 32}} = right
    end
  end

  describe "map" do
    defn map_shape_match_signature(%{a: a, b: b}) do
      a + b
    end

    test "allows pattern matching on the map shape on signature" do
      assert %T{shape: {}, type: {:f, 32}, data: %Expr{op: :add, args: [left, right]}} =
               map_shape_match_signature(%{a: 1, b: 2.0})

      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:f, 32}} = right
    end

    defn map_shape_match_alias(%{} = var) do
      %{a: a, b: b} = var
      a + b
    end

    test "allows pattern matching on the map shape with alias and underscores" do
      assert %T{shape: {}, type: {:f, 32}, data: %Expr{op: :add, args: [left, right]}} =
               map_shape_match_alias(%{a: 1, b: 2.0})

      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:f, 32}} = right
    end

    defn map_shape_match_inside_body(var) do
      %{a: a, b: b} = var
      a + b
    end

    test "allows pattern matching on the map shape inside body" do
      assert %T{shape: {}, type: {:f, 32}, data: %Expr{op: :add, args: [left, right]}} =
               map_shape_match_inside_body(%{a: 1, b: 2.0})

      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:f, 32}} = right
    end

    defn map_shape_access(var) do
      var[:a] + var[:b]
    end

    test "allows access on maps" do
      assert %T{shape: {}, type: {:f, 32}, data: %Expr{op: :add, args: [left, right]}} =
               map_shape_access(%{a: 1, b: 2.0})

      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:f, 32}} = right
    end

    defn map_shape_dot(var) do
      var.a + var.b
    end

    test "allows dot on maps" do
      assert %T{shape: {}, type: {:f, 32}, data: %Expr{op: :add, args: [left, right]}} =
               map_shape_dot(%{a: 1, b: 2.0})

      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:f, 32}} = right
    end

    defn map_return(a, b) do
      %{add: a + b, subtract: a - b}
    end

    test "returns a map" do
      assert %{add: add, subtract: subtract} = map_return(1, 2.0)

      assert %T{shape: {}, type: {:f, 32}, data: %Expr{op: :add, args: [left, right]}} = add

      assert %T{shape: {}, type: {:f, 32}, data: %Expr{op: :subtract, args: [^left, ^right]}} =
               subtract
    end

    defn map_update(map) do
      %{map | b: map.a + map.b}
    end

    test "updates a map" do
      assert %{a: a, b: add} = map_update(%{a: 1, b: 2.0})

      assert %T{shape: {}, type: {:f, 32}, data: %Expr{op: :add, args: [^a, b]}} = add

      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = a
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:f, 32}} = b
    end

    defn verify_maps(map) do
      transform(map, fn map ->
        for {k, v} <- map do
          assert Elixir.Kernel.==(v.shape, {String.to_integer(k)})
        end

        map
      end)
    end

    test "keeps map ordering across different sizes" do
      size = 50
      map = for i <- 1..size, into: %{}, do: {"#{i}", Nx.iota({i})}
      map = verify_maps(map)
      assert map_size(map) == size

      for {k, v} <- map do
        assert v.shape == {String.to_integer(k)}
      end
    end
  end

  describe "anonymous functions args" do
    defn calls_binary_fun(fun, a, b), do: fun.(a, b)

    test "calls anonymous function directly" do
      assert %T{shape: {}, type: {:f, 32}, data: %Expr{op: :add, args: [left, right]}} =
               calls_binary_fun(&Nx.add/2, 1, 2.0)

      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:f, 32}} = right
    end

    defn calls_reduce_fun(fun, t), do: Nx.reduce(t, 0, fun)

    test "calls anonymous function via reduce" do
      assert %T{shape: {}, type: {:s, 64}, data: %Expr{op: :reduce}} =
               calls_reduce_fun(&Nx.add/2, Nx.tensor([1, 2, 3]))
    end

    defn calls_tuple_fun({funa, funb}, a, b), do: {funa.(a, b), funb.(a, b)}

    test "calls anonymous function from tuples" do
      assert {%T{shape: {}, type: {:f, 32}, data: %Expr{op: :add, args: [left, right]}},
              %T{shape: {}, type: {:f, 32}, data: %Expr{op: :subtract, args: [left, right]}}} =
               calls_tuple_fun({&Nx.add/2, &Nx.subtract/2}, 1, 2.0)

      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:f, 32}} = right

      assert_raise ArgumentError,
                   ~r"Anonymous functions are only allowed as direct arguments to defn",
                   fn -> calls_tuple_fun({&Nx.add/2, 0}, 1, 2.0) end
    end
  end

  describe "unary ops" do
    defn exp(t), do: Nx.exp(t)

    test "to expr" do
      assert %T{shape: {3}, type: {:f, 32}, data: %Expr{op: :exp, args: [_]}} =
               exp(Nx.tensor([1, 2, 3]))

      assert %T{shape: {3}, type: {:f, 32}, data: %Expr{op: :exp, args: [_]}} =
               exp(Nx.tensor([1, 2, 3], type: {:f, 32}))

      assert %T{shape: {3}, type: {:bf, 16}, data: %Expr{op: :exp, args: [_]}} =
               exp(Nx.tensor([1, 2, 3], type: {:bf, 16}))
    end
  end

  describe "binary ops" do
    defn add(t1, t2), do: Nx.add(t1, t2)
    defn add_two_int(t), do: Nx.add(t, 2)
    defn add_two_float(t), do: Nx.add(t, 2)

    test "to expr" do
      assert %T{shape: {3}, type: {:s, 64}, data: %Expr{op: :add, args: [_, _]}} =
               add(Nx.tensor([1, 2, 3]), Nx.tensor(1))

      assert %T{shape: {2, 2}, type: {:s, 64}, data: %Expr{op: :add, args: [_, _]}} =
               add(Nx.tensor([[1, 2], [3, 4]]), Nx.tensor([1, 2]))

      assert %T{shape: {2, 2}, type: {:f, 32}, data: %Expr{op: :add, args: [_, _]}} =
               add(Nx.tensor([[1, 2], [3, 4]], type: {:f, 32}), Nx.tensor([1, 2]))

      assert %T{shape: {2, 2}, type: {:f, 32}, data: %Expr{op: :add, args: [_, _]}} =
               add(Nx.tensor([[1, 2], [3, 4]], type: {:f, 32}), Nx.tensor([1, 2], type: {:s, 32}))
    end

    test "constant" do
      assert %T{shape: {3}, type: {:u, 8}, data: %Expr{op: :add, args: [_, _]}} =
               add_two_int(Nx.tensor([1, 2, 3], type: {:u, 8}))

      assert %T{shape: {3}, type: {:bf, 16}, data: %Expr{op: :add, args: [_, _]}} =
               add_two_float(Nx.tensor([1, 2, 3], type: {:bf, 16}))
    end
  end

  describe "aggregate axes ops" do
    defn sum_all(t), do: Nx.sum(t)
    defn sum_pos(t), do: Nx.sum(t, axes: [0, 1])
    defn sum_neg(t), do: Nx.sum(t, axes: [-1, -2])
    defn sum_keep(t), do: Nx.sum(t, axes: [0, 1], keep_axes: true)

    test "to expr" do
      assert %T{
               shape: {},
               type: {:s, 64},
               data: %Expr{op: :sum, args: [_, [axes: nil, keep_axes: false]]}
             } = sum_all(Nx.tensor([1, 2, 3]))

      assert %T{
               shape: {},
               type: {:s, 64},
               data: %Expr{op: :sum, args: [_, [axes: [0, 1], keep_axes: false]]}
             } = sum_pos(Nx.tensor([[1, 2, 3], [1, 2, 3]], type: {:s, 8}))

      assert %T{
               shape: {3},
               type: {:f, 32},
               data: %Expr{op: :sum, args: [_, [axes: [0, 1], keep_axes: false]]}
             } = sum_pos(Nx.tensor([[[1, 2, 3], [1, 2, 3]]], type: {:f, 32}))

      assert %T{
               shape: {},
               type: {:u, 64},
               data: %Expr{op: :sum, args: [_, [axes: [1, 0], keep_axes: false]]}
             } = sum_neg(Nx.tensor([[1, 2, 3], [1, 2, 3]], type: {:u, 8}))

      assert %T{
               shape: {1},
               type: {:bf, 16},
               data: %Expr{op: :sum, args: [_, [axes: [2, 1], keep_axes: false]]}
             } = sum_neg(Nx.tensor([[[1, 2, 3], [1, 2, 3]]], type: {:bf, 16}))

      assert %T{
               shape: {1, 1, 3},
               type: {:f, 32},
               data: %Expr{op: :sum, args: [_, [axes: [0, 1], keep_axes: true]]}
             } = sum_keep(Nx.tensor([[[1, 2, 3], [1, 2, 3]]], type: {:f, 32}))
    end
  end

  describe "creation ops" do
    defn iota(t), do: Nx.iota(t)
    defn eye, do: Nx.eye(2)
    defn random_uniform(t), do: Nx.random_uniform(t, 0.0, 2.0)
    defn random_normal(t), do: Nx.random_normal(t, 0.0, 1.0)

    test "iota" do
      assert %T{shape: {3}, data: %Expr{op: :iota, args: [nil]}} = iota(Nx.tensor([1, 2, 3]))
    end

    test "eye" do
      assert %T{shape: {2, 2}, data: %Expr{op: :eye, args: []}} = eye()
    end

    test "random uniform" do
      assert %T{
               shape: {3},
               data: %Expr{op: :random_uniform, args: [%T{shape: {}}, %T{shape: {}}]}
             } = random_uniform(Nx.tensor([1, 2, 3]))
    end

    test "random normal" do
      assert %T{shape: {3}, data: %Expr{op: :random_normal, args: [%T{shape: {}}, %T{shape: {}}]}} =
               random_normal(Nx.tensor([1, 2, 3]))
    end
  end

  describe "tensor ops" do
    defn dot2(t1, t2), do: Nx.dot(t1, t2)
    defn dot6(t1, t2), do: Nx.dot(t1, [-2], [], t2, [-1], [])
    defn outer(t1, t2), do: Nx.outer(t1, t2)
    defn transpose_1(t), do: Nx.transpose(t)
    defn transpose_2(t), do: Nx.transpose(t, axes: [-1, -2])
    defn reshape(t), do: Nx.reshape(t, {2, 3})

    test "dot product" do
      assert %T{data: %Expr{op: :dot, args: [_, [0], _, _, [0], _]}, shape: {2}} =
               dot2(Nx.tensor([1, 2, 3]), Nx.tensor([[1, 2], [3, 4], [5, 6]]))

      assert %T{data: %Expr{op: :dot, args: [_, [1], _, _, [0], _]}, shape: {2}} =
               dot2(Nx.tensor([[1, 2, 3], [1, 2, 3]]), Nx.tensor([1, 2, 3]))

      assert %T{data: %Expr{op: :dot, args: [_, [1], _, _, [0], _]}, shape: {2, 2}} =
               dot2(Nx.tensor([[1, 2, 3], [1, 2, 3]]), Nx.tensor([[1, 2], [3, 4], [5, 6]]))

      assert %T{data: %Expr{op: :dot, args: [_, [0], [], _, [1], []]}, shape: {3, 3}} =
               dot6(Nx.tensor([[1, 2, 3], [1, 2, 3]]), Nx.tensor([[1, 2], [3, 4], [5, 6]]))
    end

    test "outer product" do
      assert %T{data: %Expr{op: :outer, args: [_, _]}, shape: {3, 3}} =
               outer(Nx.tensor([1, 2, 3]), Nx.tensor([1, 2, 3]))
    end

    test "transpose" do
      assert %T{data: %Expr{op: :transpose, args: [_, [1, 0]]}, shape: {3, 2}} =
               transpose_1(Nx.tensor([[1, 2, 3], [1, 2, 3]]))

      assert %T{data: %Expr{op: :transpose, args: [_, [1, 0]]}, shape: {3, 2}} =
               transpose_2(Nx.tensor([[1, 2, 3], [1, 2, 3]]))
    end

    test "reshape" do
      assert %T{data: %Expr{op: :reshape, args: [_, _]}, shape: {2, 3}} =
               reshape(Nx.tensor([[1, 2], [3, 4], [5, 6]]))
    end
  end

  describe "broadcast" do
    defn broadcast(t), do: Nx.broadcast(t, {3, 3, 3})
    defn broadcast_axes(t), do: Nx.broadcast(t, {3, 2}, axes: [-2])

    test "with and without axes" do
      assert %T{data: %Expr{op: :broadcast, args: [_, _, [2]]}, shape: {3, 3, 3}} =
               broadcast(Nx.tensor([1, 2, 3]))

      assert %T{data: %Expr{op: :broadcast, args: [_, _, [0]]}, shape: {3, 2}} =
               broadcast_axes(Nx.tensor([1, 2, 3]))
    end

    defn broadcast_collapse1(t),
      do: t |> Nx.broadcast({5, 3}) |> Nx.broadcast({7, 5, 3})

    defn broadcast_collapse2(t),
      do: t |> Nx.broadcast({3, 5}, axes: [0]) |> Nx.broadcast({3, 5, 7}, axes: [0, 1])

    defn broadcast_collapse3(t),
      do: t |> Nx.broadcast({3, 5}, axes: [0]) |> Nx.broadcast({3, 7, 5}, axes: [0, 2])

    defn broadcast_collapse4(t),
      do: t |> Nx.broadcast({3, 5}, axes: [0]) |> Nx.broadcast({7, 3, 5}, axes: [1, 2])

    defn broadcast_collapse5(t),
      do: t |> Nx.broadcast({5, 3}) |> Nx.broadcast({7, 5, 3, 9}, axes: [1, 2])

    defn broadcast_collapse6(t),
      do: t |> Nx.broadcast({5, 3, 7}, axes: [1]) |> Nx.broadcast({9, 5, 3, 7}, axes: [1, 2, 3])

    defn broadcast_collapse7(t),
      do:
        t |> Nx.broadcast({3, 5, 7}, axes: [0, 2]) |> Nx.broadcast({3, 9, 5, 7}, axes: [0, 2, 3])

    test "collapses" do
      assert %T{data: %Expr{op: :broadcast, args: [_, {7, 5, 3}, [1]]}, shape: {7, 5, 3}} =
               broadcast_collapse1(Nx.tensor([1, 2, 3]))

      assert %T{data: %Expr{op: :broadcast, args: [_, {3, 5, 7}, [0]]}, shape: {3, 5, 7}} =
               broadcast_collapse2(Nx.tensor([1, 2, 3]))

      assert %T{data: %Expr{op: :broadcast, args: [_, {3, 7, 5}, [0, 2]]}} =
               broadcast_collapse3(Nx.tensor([1, 2, 3]))

      assert %T{data: %Expr{op: :broadcast, args: [_, {7, 3, 5}, [1, 2]]}} =
               broadcast_collapse4(Nx.tensor([1, 2, 3]))

      assert %T{data: %Expr{op: :broadcast, args: [_, {7, 5, 3, 9}, [1, 2]]}} =
               broadcast_collapse5(Nx.tensor([1, 2, 3]))

      assert %T{data: %Expr{op: :broadcast, args: [_, {9, 5, 3, 7}, [1, 2, 3]]}} =
               broadcast_collapse6(Nx.tensor([1, 2, 3]))

      assert %T{data: %Expr{op: :broadcast, args: [_, {3, 9, 5, 7}, [0, 2, 3]]}} =
               broadcast_collapse7(Nx.iota({3, 7}))
    end
  end

  describe "squeeze" do
    defn squeeze(t), do: Nx.squeeze(t)

    test "sized one dimensions" do
      assert %T{data: %Expr{op: :squeeze, args: [_, [0, 2, 4]]}, shape: {3, 2}} =
               squeeze(Nx.iota({1, 3, 1, 2, 1}))
    end

    defn squeeze_collapse1(t), do: t |> Nx.squeeze(axes: [0, 2]) |> Nx.squeeze(axes: [0, 2])
    defn squeeze_collapse2(t), do: t |> Nx.squeeze(axes: [3, 1]) |> Nx.squeeze(axes: [2])
    defn squeeze_collapse3(t), do: t |> Nx.squeeze(axes: [2]) |> Nx.squeeze(axes: [3, 1])

    test "with explicit dimensions are collapsed" do
      assert %T{data: %Expr{op: :squeeze, args: [_, [0, 1, 2, 4]]}, shape: {1}, names: [:d]} =
               squeeze_collapse1(Nx.iota({1, 1, 1, 1, 1}, names: [:a, :b, :c, :d, :e]))

      assert %T{data: %Expr{op: :squeeze, args: [_, [1, 3, 4]]}, shape: {1, 1}, names: [:a, :c]} =
               squeeze_collapse2(Nx.iota({1, 1, 1, 1, 1}, names: [:a, :b, :c, :d, :e]))

      assert %T{data: %Expr{op: :squeeze, args: [_, [1, 2, 4]]}, shape: {1, 1}, names: [:a, :d]} =
               squeeze_collapse3(Nx.iota({1, 1, 1, 1, 1}, names: [:a, :b, :c, :d, :e]))
    end
  end

  describe "conditional ops" do
    defn select(t1, t2, t3), do: Nx.select(t1, t2, t3)

    test "select with tensor predicate" do
      assert %{data: %Expr{op: :select, args: [_, _, _]}, shape: {2, 2}} =
               select(Nx.tensor([[1, 1], [0, 0]]), Nx.tensor(1), Nx.tensor(0))
    end

    test "select with scalar predicate" do
      assert %{data: %Expr{op: :select, args: [_, _, _]}, shape: {5}} =
               select(Nx.tensor(1), Nx.tensor([1, 2, 3, 4, 5]), Nx.tensor(0))

      assert %{data: %Expr{op: :select, args: [_, _, _]}, shape: {2, 2}} =
               select(Nx.tensor(1), Nx.tensor([[1], [2]]), Nx.tensor([[1, 2]]))
    end
  end

  describe "reduce ops" do
    defn reduce(t1, acc), do: Nx.reduce(t1, acc, fn x, y -> x + y end)

    defn reduce_static(t1, acc), do: Nx.reduce(t1, acc, fn _, _ -> 0 end)

    defn reduce_invalid(t1, amplifier), do: Nx.reduce(t1, 0, fn x, y -> x * amplifier + y end)

    defn reduce_non_scalar(t1), do: Nx.reduce(t1, 0, fn x, y -> Nx.broadcast(x * y, {1, 1}) end)

    defn reduce_with_opts(t1, acc),
      do: Nx.reduce(t1, acc, [type: {:f, 64}, axes: [-1]], fn x, y -> x + y end)

    test "reduces with function" do
      assert %{
               data: %Expr{op: :reduce, args: [_, _, [axes: nil, keep_axes: false], fun]},
               type: {:s, 64},
               shape: {}
             } = reduce(Nx.tensor([1, 2, 3]), 0)

      assert %T{data: %Expr{op: :fun}} = fun

      assert %{
               data: %Expr{op: :reduce, args: [_, _, [axes: [1], keep_axes: false], fun]},
               type: {:f, 64},
               shape: {3}
             } = reduce_with_opts(Nx.tensor([[1], [2], [3]]), 0)

      assert %T{data: %Expr{op: :fun}} = fun
    end

    test "reduces with constant" do
      assert %{
               data: %Expr{op: :reduce, args: [_, _, [axes: nil, keep_axes: false], fun]},
               type: {:s, 64},
               shape: {}
             } = reduce_static(Nx.tensor([1, 2, 3]), 0)

      assert %T{data: %Expr{op: :fun}} = fun

      assert %{
               data: %Expr{op: :reduce, args: [_, _, [axes: nil, keep_axes: false], fun]},
               type: {:f, 32},
               shape: {}
             } = reduce_static(Nx.tensor([1, 2, 3]), 0.0)

      assert %T{data: %Expr{op: :fun}} = fun
    end

    test "reduces raises on invalid expression" do
      assert_raise RuntimeError,
                   ~r"cannot build defn because expressions come from different contexts",
                   fn -> reduce_invalid(Nx.tensor([1, 2, 3]), 0) end
    end

    test "reduces raises on non scalar functions" do
      assert_raise RuntimeError,
                   "reduce function must return a scalar tensor, got: {1, 1}",
                   fn -> reduce_non_scalar(Nx.tensor([1, 2, 3])) end
    end
  end

  describe "operators" do
    defn add_two(a, b), do: a + b

    test "+" do
      assert %T{data: %Expr{op: :add, args: [_, _]}} = add_two(1, 2)
    end

    defn subtract_two(a, b), do: a - b

    test "-" do
      assert %T{data: %Expr{op: :subtract, args: [_, _]}} = subtract_two(1, 2)
    end

    defn multiply_two(a, b), do: a * b

    test "*" do
      assert %T{data: %Expr{op: :multiply, args: [_, _]}} = multiply_two(1, 2)
    end

    defn divide_two(a, b), do: a / b

    test "/" do
      assert %T{data: %Expr{op: :divide, args: [_, _]}} = divide_two(1, 2)
    end

    defn land_two(a, b), do: a and b

    defn land_true(a) do
      val = transform({}, fn _ -> true end)
      val and a
    end

    test "and" do
      assert %T{data: %Expr{op: :logical_and, args: [_, _]}} = land_two(1, 2)

      assert_raise ArgumentError, ~r/boolean value passed to Nx.Defn.Kernel.and\/2/, fn ->
        land_true(2)
      end
    end

    defn lor_two(a, b), do: a or b

    defn lor_true(a) do
      val = transform({}, fn _ -> true end)
      val or a
    end

    test "or" do
      assert %T{data: %Expr{op: :logical_or, args: [_, _]}} = lor_two(1, 2)

      assert_raise ArgumentError, ~r/boolean value passed to Nx.Defn.Kernel.or\/2/, fn ->
        lor_true(2)
      end
    end

    defn lnot(a), do: not a
    defn lnot_true(), do: not transform({}, fn _ -> true end)

    test "not" do
      assert %T{data: %Expr{op: :equal, args: [_, _]}} = lnot(1)

      assert_raise ArgumentError, ~r/boolean value passed to Nx.Defn.Kernel.not\/1/, fn ->
        lnot_true()
      end
    end

    defn band_two(a, b), do: a &&& b

    test "&&&" do
      assert %T{data: %Expr{op: :bitwise_and, args: [_, _]}} = band_two(1, 2)
    end

    defn bor_two(a, b), do: a ||| b

    test "|||" do
      assert %T{data: %Expr{op: :bitwise_or, args: [_, _]}} = bor_two(1, 2)
    end

    defn bsl_two(a, b), do: a <<< b

    test "<<<" do
      assert %T{data: %Expr{op: :left_shift, args: [_, _]}} = bsl_two(1, 2)
    end

    defn bsr_two(a, b), do: a >>> b

    test ">>>" do
      assert %T{data: %Expr{op: :right_shift, args: [_, _]}} = bsr_two(1, 2)
    end

    defn add_two_with_pipe(a, b), do: a |> Nx.add(b)

    test "|>" do
      assert %T{data: %Expr{op: :add, args: [_, _]}} = add_two_with_pipe(1, 2)
    end

    defn unary_plus(a), do: +a
    defn unary_minus(a), do: -a

    test "unary plus and minus" do
      assert %T{data: %Expr{op: :parameter, args: [_]}} = unary_plus(1)
      assert %T{data: %Expr{op: :negate, args: [_]}} = unary_minus(1)
    end

    defn unary_bnot(a), do: ~~~a

    test "~~~" do
      assert %T{data: %Expr{op: :bitwise_not, args: [_]}} = unary_bnot(1)
    end

    defn equality(a, b), do: a == b

    test "==" do
      assert %T{data: %Expr{op: :equal, args: [_, _]}} = equality(1, 2)
    end

    defn inequality(a, b), do: a != b

    test "!=" do
      assert %T{data: %Expr{op: :not_equal, args: [_, _]}} = inequality(1, 2)
    end

    defn less_than(a, b), do: a < b

    test "<" do
      assert %T{data: %Expr{op: :less, args: [_, _]}} = less_than(1, 2)
    end

    defn greater_than(a, b), do: a > b

    test ">" do
      assert %T{data: %Expr{op: :greater, args: [_, _]}} = greater_than(1, 2)
    end

    defn less_than_or_equal(a, b), do: a <= b

    test "<=" do
      assert %T{data: %Expr{op: :less_equal, args: [_, _]}} = less_than_or_equal(1, 2)
    end

    defn greater_than_or_equal(a, b), do: a >= b

    test ">=" do
      assert %T{data: %Expr{op: :greater_equal, args: [_, _]}} = greater_than_or_equal(1, 2)
    end
  end

  describe "kernel functions/macros" do
    defn max_two(a, b) do
      max(a, b)
    end

    test "max/2" do
      assert %T{data: %Expr{op: :max, args: [_, _]}} = max_two(1, 2)
    end

    defn min_two(a, b) do
      min(a, b)
    end

    test "min/2" do
      assert %T{data: %Expr{op: :min, args: [_, _]}} = min_two(1, 2)
    end

    defn maxu(a), do: rewrite_types(a, max_unsigned_type: {:u, 32})
    defn maxs(a), do: rewrite_types(a, max_signed_type: {:s, 32})
    defn maxf(a), do: rewrite_types(a, max_float_type: {:f, 32})

    test "max_*_type/2" do
      assert %T{data: %Expr{op: :as_type, args: [_]}} = maxu(Nx.tensor(1, type: {:u, 64}))
      assert %T{data: %Expr{op: :as_type, args: [_]}} = maxs(Nx.tensor(1, type: {:s, 64}))
      assert %T{data: %Expr{op: :as_type, args: [_]}} = maxf(Nx.tensor(1, type: {:f, 64}))
    end
  end

  describe "access" do
    defn single_access(t), do: {t[0], t[-1]}

    test "single dimensional single access" do
      {zero, minus_one} = single_access(Nx.tensor([1, 2, 3, 4, 5]))
      assert %T{data: %Expr{op: :squeeze, args: [slice, [0]]}, shape: {}} = zero
      assert %T{data: %Expr{op: :slice, args: [_, [0], [1], [1]]}, shape: {1}} = slice

      assert %T{data: %Expr{op: :squeeze, args: [slice, [0]]}, shape: {}} = minus_one
      assert %T{data: %Expr{op: :slice, args: [_, [4], [1], [1]]}, shape: {1}} = slice
    end

    test "multi dimensional single access" do
      {zero, minus_one} = single_access(Nx.iota({3, 4, 5}))
      assert %T{data: %Expr{op: :squeeze, args: [slice, [0]]}, shape: {4, 5}} = zero

      assert %T{
               data: %Expr{op: :slice, args: [_, [0, 0, 0], [1, 4, 5], [1, 1, 1]]},
               shape: {1, 4, 5}
             } = slice

      assert %T{data: %Expr{op: :squeeze, args: [slice, [0]]}, shape: {4, 5}} = minus_one

      assert %T{
               data: %Expr{op: :slice, args: [_, [2, 0, 0], [1, 4, 5], [1, 1, 1]]},
               shape: {1, 4, 5}
             } = slice
    end

    defn multi_access(t), do: t[1][2][3]

    test "multi dimensional multi-access with integers is collapsed" do
      assert %T{data: %Expr{op: :squeeze, args: [slice, [0, 1, 2]]}, shape: {}} =
               multi_access(Nx.iota({3, 4, 5}))

      assert %T{
               data: %Expr{op: :slice, args: [_, [1, 2, 3], [1, 1, 1], [1, 1, 1]]},
               shape: {1, 1, 1}
             } = slice
    end

    defn range_access(t), do: t[1][1..2]

    test "multi dimensional multi-access with ranges is collapsed" do
      assert %T{data: %Expr{op: :squeeze, args: [slice, [0]]}, shape: {2, 5}} =
               range_access(Nx.iota({3, 4, 5}))

      assert %T{
               data: %Expr{op: :slice, args: [_, [1, 1, 0], [1, 2, 5], [1, 1, 1]]},
               shape: {1, 2, 5}
             } = slice
    end

    defn keyword_access(t), do: t[[z: 1..-2//1]][[y: 1..2]]

    test "multi dimensional multi-access with keywords is collapsed" do
      assert %T{
               data: %Expr{op: :slice, args: [_, [0, 1, 1], [3, 2, 3], [1, 1, 1]]},
               shape: {3, 2, 3}
             } = keyword_access(Nx.iota({3, 4, 5}, names: [:x, :y, :z]))
    end

    defn elixir_access(a, opts \\ []), do: Nx.sum(a, axes: opts[:axes])

    test "also works for other Elixir data structures" do
      assert %T{data: %Expr{op: :sum, args: [_, [axes: [1], keep_axes: false]]}} =
               elixir_access(Nx.iota({2, 2}), axes: [1])
    end
  end

  describe "lu" do
    defn lu(t), do: Nx.LinAlg.lu(t)

    test "returns tuples" do
      assert {p, l, u} = lu(Nx.iota({3, 3}))

      assert %T{data: %Expr{op: :elem, args: [lu_expr, 0, 3]}, shape: {3, 3}} = p
      assert %T{data: %Expr{op: :elem, args: [^lu_expr, 1, 3]}, shape: {3, 3}} = l
      assert %T{data: %Expr{op: :elem, args: [^lu_expr, 2, 3]}, shape: {3, 3}} = u
    end
  end

  describe "qr" do
    defn qr(t), do: Nx.LinAlg.qr(t)

    test "returns tuples" do
      assert {left, right} = qr(Nx.iota({3, 2}))

      assert %T{data: %Expr{op: :elem, args: [qr_expr, 0, 2]}, shape: {3, 2}} = left
      assert %T{data: %Expr{op: :elem, args: [^qr_expr, 1, 2]}, shape: {2, 2}} = right
    end
  end

  describe "svd" do
    defn svd(t), do: Nx.LinAlg.svd(t)

    test "returns tuples" do
      assert {u, s, vt} = svd(Nx.iota({3, 3}))

      assert %T{data: %Expr{op: :elem, args: [svd_expr, 0, 3]}, shape: {3, 3}} = u
      assert %T{data: %Expr{op: :elem, args: [^svd_expr, 1, 3]}, shape: {3}} = s
      assert %T{data: %Expr{op: :elem, args: [^svd_expr, 2, 3]}, shape: {3, 3}} = vt
    end
  end

  describe "macros" do
    defmodule Macros do
      defmacro add(a, b) do
        use Nx.Defn.Kernel

        quote do
          unquote(a) + unquote(b)
        end
      end
    end

    defn add_two_from_external_macro(a, b) do
      require Macros
      Macros.add(a, b)
    end

    test "external" do
      assert %T{data: %Expr{op: :add, args: [_, _]}} = add_two_from_external_macro(1, 2)
    end

    defmacrop add_internal(a, b) do
      use Nx.Defn.Kernel

      quote do
        unquote(a) + unquote(b)
      end
    end

    defn add_two_from_internal_macro(a, b) do
      add_internal(a, b)
    end

    test "internal" do
      assert %T{data: %Expr{op: :add, args: [_, _]}} = add_two_from_external_macro(1, 2)
    end

    defn add_two_from_alias(a, b) do
      alias Nx, as: N
      N.add(a, b)
    end

    test "aliases" do
      assert %T{data: %Expr{op: :add, args: [_, _]}} = add_two_from_alias(1, 2)
    end

    defn add_two_from_kernel_alias(a, b) do
      a |> Kernel.+(b)
    end

    test "kernel alias" do
      assert %T{data: %Expr{op: :add, args: [_, _]}} = add_two_from_kernel_alias(1, 2)
    end

    dynamic_name = String.to_atom(Enum.join(~w(dynamic name add two), "_"))
    operator = :add
    defn unquote(dynamic_name)(left, right), do: Nx.unquote(operator)(left, right)

    test "dynamic name" do
      assert %T{data: %Expr{op: :add, args: [_, _]}} = dynamic_name_add_two(1, 2)
    end
  end

  describe "local functions" do
    defn add_two_from_public(a, b) do
      add_two_from_public_impl(a, b)
    end

    defn add_two_from_public_impl(a, b) do
      a + b
    end

    test "public" do
      assert %T{data: %Expr{op: :add, args: [_, _]}} = add_two_from_public(1, 2)
    end

    defn add_two_from_private(a, b) do
      add_two_from_private_impl(a, b)
    end

    defn add_two_from_private_impl(a, b) do
      a + b
    end

    test "private" do
      assert %T{data: %Expr{op: :add, args: [_, _]}} = add_two_from_private(1, 2)
    end
  end

  describe "remote functions" do
    defmodule Remote do
      defn add_two(c, d), do: c + d
    end

    defn add_two_remote(a, b), do: Remote.add_two(a, b)

    test "public" do
      assert %T{data: %Expr{op: :add, args: [_, _]}} = add_two_remote(1, 2)
    end

    defn add_two_unknown(a, b), do: Nx.DefnTest.unknown(a, b)

    def not_defn(a, b), do: Nx.add(a, b)
    defn add_two_not_defn(a, b), do: Nx.DefnTest.not_defn(a, b)

    test "undefined remote" do
      assert_raise UndefinedFunctionError,
                   "function Nx.DefnTest.unknown/2 is undefined or private",
                   fn -> add_two_unknown(1, 2) end
    end

    test "not defn remote" do
      assert_raise RuntimeError,
                   "cannot invoke Nx.DefnTest.not_defn/2 inside defn because it was not defined with defn",
                   fn -> add_two_not_defn(1, 2) end
    end
  end

  describe "if" do
    defn if3(a, b, c), do: if(a, do: b, else: c)
    defn if2(a, b), do: if(a, do: b)

    test "converges types" do
      assert %T{data: %Expr{op: :cond}, shape: {}, type: {:f, 32}} =
               if3(Nx.tensor(0), Nx.tensor(0, type: {:s, 16}), Nx.tensor(0, type: {:f, 32}))

      assert %T{data: %Expr{op: :cond}, shape: {}, type: {:f, 32}} =
               if3(Nx.tensor(0), Nx.tensor(0, type: {:s, 32}), Nx.tensor(0, type: {:f, 32}))

      assert %T{data: %Expr{op: :cond}, shape: {}, type: {:u, 16}} =
               if2(Nx.tensor(0), Nx.tensor(0, type: {:u, 16}))

      assert %T{data: %Expr{op: :cond}, shape: {}, type: {:f, 32}} =
               if2(Nx.tensor(0), Nx.tensor(0, type: {:f, 32}))
    end

    test "converges shapes and names" do
      assert %T{data: %Expr{op: :cond}, shape: {2, 2}, names: [:x, :y]} =
               if3(
                 Nx.tensor(0),
                 Nx.tensor([1, 2], names: [:y]),
                 Nx.tensor([[3], [4]], names: [:x, nil])
               )
    end
  end

  describe "cond" do
    defn cond4(a, b, c, d) do
      cond do
        Nx.greater(a, 0) -> b + 1
        Nx.less(a, 0) -> c - 1
        true -> d * 2
      end
    end

    test "supports multiple clauses" do
      assert %T{data: %Expr{op: :cond, args: [clauses, last]}, shape: {}, type: {:s, 64}} =
               cond4(Nx.tensor(0), Nx.tensor(1), Nx.tensor(2), Nx.tensor(3))

      [{first_head, first_body}, {second_head, second_body}] = clauses
      assert %T{data: %Expr{op: :greater}} = first_head
      assert %T{data: %Expr{op: :add}} = first_body
      assert %T{data: %Expr{op: :less}} = second_head
      assert %T{data: %Expr{op: :subtract}} = second_body
      assert %T{data: %Expr{op: :multiply}} = last
    end

    test "converges types" do
      assert %T{data: %Expr{op: :cond}, shape: {}, type: {:s, 32}} =
               cond4(
                 Nx.tensor(0),
                 Nx.tensor(1, type: {:s, 8}),
                 Nx.tensor(2, type: {:s, 16}),
                 Nx.tensor(3, type: {:s, 32})
               )
    end

    test "converges shapes and names" do
      assert %T{data: %Expr{op: :cond}, shape: {2, 2}, names: [:x, :y]} =
               cond4(
                 Nx.tensor(0),
                 Nx.tensor([1, 2], names: [:y]),
                 Nx.tensor([[3], [4]], names: [:x, nil]),
                 Nx.tensor(5)
               )
    end

    defn cond_lit(a) do
      if Nx.any?(a), do: 1, else: -1
    end

    test "supports literals" do
      assert cond_lit(Nx.tensor(0)), do: Nx.tensor(-1)
      assert cond_lit(Nx.tensor(1)), do: Nx.tensor(1)
    end

    test "raises if cond is missing last atom clause" do
      assert_raise CompileError, ~r"expected the last clause of cond to match on an atom", fn ->
        defmodule InvalidCond do
          defn badcond(a) do
            cond do
              Nx.any?(a) -> +a
              Nx.all?(a) -> -a
            end
          end
        end
      end
    end

    test "raises if given a non-scalar as condition" do
      assert_raise CompileError, ~r"condition must be a scalar tensor, got:", fn ->
        cond4(Nx.tensor([0, 0]), Nx.tensor(1), Nx.tensor(2), Nx.tensor(3))
      end
    end
  end

  describe "while/3" do
    defn upto10(x) do
      while x, Nx.less(x, 10) do
        x + 1
      end
    end

    test "simple" do
      assert %T{
               data: %Expr{op: :while, args: [initial, arg, condition, body]},
               shape: {},
               type: {:s, 64}
             } = upto10(Nx.tensor(0))

      assert %T{data: %Expr{op: :parameter}, shape: {}, type: {:s, 64}} = initial
      assert %T{data: %Expr{op: :parameter}, shape: {}, type: {:s, 64}} = arg
      assert %T{data: %Expr{op: :less}, shape: {}, type: {:u, 8}} = condition
      assert %T{data: %Expr{op: :add}, shape: {}, type: {:s, 64}} = body
    end

    defn while_constant(x) do
      while x, Nx.less(x, 10) do
        1
      end
    end

    test "constant" do
      assert %T{
               data: %Expr{op: :while, args: [initial, arg, condition, body]},
               shape: {},
               type: {:s, 64}
             } = while_constant(Nx.tensor(0))

      assert %T{data: %Expr{op: :parameter}, shape: {}, type: {:s, 64}} = initial
      assert %T{data: %Expr{op: :parameter}, shape: {}, type: {:s, 64}} = arg
      assert %T{data: %Expr{op: :less}, shape: {}, type: {:u, 8}} = condition
      assert %T{data: %Expr{op: :scalar, args: [1]}, shape: {}, type: {:s, 64}} = body
    end

    defn factorial(x) do
      {factorial, _} =
        while {factorial = 1, x}, Nx.greater(x, 1) do
          {factorial * x, x - 1}
        end

      factorial
    end

    test "factorial" do
      assert %T{} = factorial(5)

      assert_raise CompileError,
                   ~r/the do-block in while must return the shape, type, and names as the initial arguments. Got body \{f32, f32\} and initial \{s64, f32\}/,
                   fn -> factorial(10.0) end
    end

    defn while_mixed_return(a, b) do
      while {a, b}, Nx.less(a, 10) do
        %{a: a, b: b}
      end
    end

    test "raises on mixed return" do
      assert_raise CompileError,
                   ~r/the do-block in while must return the shape, type, and names as the initial arguments. Got body %\{:a => s64, :b => s64\} and initial \{s64, s64\}/,
                   fn -> while_mixed_return(Nx.tensor(0), Nx.tensor(1)) end
    end

    defn while_mixed_context(a, b) do
      while a, Nx.less(a, 10) do
        a + b
      end
    end

    test "raises on mixed context" do
      assert_raise RuntimeError,
                   ~r"cannot build defn because expressions come from different contexts: :root and :while",
                   fn -> while_mixed_context(Nx.tensor(0), Nx.tensor(1)) end
    end

    test "raises if non-variable is given as pattern" do
      assert_raise ArgumentError,
                   ~r"invalid initial argument for \"while\". Expected a variable, a variable assignment, or a tuple of the same",
                   fn ->
                     defmodule InvalidWhile do
                       defn upto(a) do
                         while :foo, Nx.less(a, 10) do
                           a + 1
                         end
                       end
                     end
                   end
    end

    test "raises if non-block is given" do
      assert_raise ArgumentError,
                   ~r"expected third argument to \"while\" to be a do-block, got: a \+ 1",
                   fn ->
                     defmodule InvalidWhile do
                       defn upto(a) do
                         while(:foo, Nx.less(a, 10), a + 1)
                       end
                     end
                   end
    end

    test "raises if given a non-scalar as condition" do
      assert_raise CompileError, ~r"condition must be a scalar tensor, got:", fn ->
        upto10(Nx.tensor([0, 0]))
      end
    end
  end

  describe "transform" do
    defn transform_inspect(a, b) do
      (Nx.tanh(a) + Nx.power(b, 3)) |> inspect_expr()
    end

    defn transform_inspect_label(a, b) do
      (Nx.tanh(a) + Nx.power(b, 3)) |> inspect_expr(label: "HELLO")
    end

    test "executes the transformation" do
      assert ExUnit.CaptureIO.capture_io(fn -> transform_inspect(1, 2) end) == """
             #Nx.Tensor<
               f32
             \s\s
               Nx.Defn.Expr
               parameter a         s64
               parameter c         s64
               b = tanh [ a ]      f32
               d = power [ c, 3 ]  s64
               e = add [ b, d ]    f32
             >
             """

      assert ExUnit.CaptureIO.capture_io(fn -> transform_inspect_label(1, 2) end) == """
             HELLO: #Nx.Tensor<
               f32
             \s\s
               Nx.Defn.Expr
               parameter a         s64
               parameter c         s64
               b = tanh [ a ]      f32
               d = power [ c, 3 ]  s64
               e = add [ b, d ]    f32
             >
             """
    end

    @defn_compiler Evaluator
    defn transform_back_and_forth(a) do
      Nx.exp(transform(Nx.negate(a), &private_back_and_forth/1))
    end

    defp private_back_and_forth(a) do
      Evaluator = Nx.Defn.Compiler.current()
      final_back_and_forth(a)
    end

    defn final_back_and_forth(a), do: Nx.tanh(a)

    test "back and forth between Elixir and defn" do
      assert transform_back_and_forth(Nx.tensor(1)) ==
               Nx.tensor(1) |> Nx.negate() |> Nx.tanh() |> Nx.exp()
    end

    @defn_compiler Evaluator
    defn transform_variable_access(a, b) do
      transform(:ok, fn :ok -> a + b end)
    end

    test "supports variable access" do
      assert transform_variable_access(Nx.tensor(1), Nx.tensor(2)) == Nx.tensor(3)
    end
  end

  describe "jit" do
    defn defn_jit({a, b}, c), do: a + b - c

    test "compiles defn function" do
      assert Nx.Defn.jit(&defn_jit/2, [{4, 5}, 3]) == Nx.tensor(6)
      assert Nx.Defn.jit(&defn_jit/2, [{4, 5}, Nx.tensor(3)]) == Nx.tensor(6)
      assert Nx.Defn.jit(&defn_jit(&1, 3), [{4, 5}]) == Nx.tensor(6)

      assert %T{data: %Expr{op: :subtract}} =
               Nx.Defn.jit(&defn_jit/2, [{1, 2}, 3], compiler: Identity)
    end

    @defn_compiler Evaluator
    defn defn_jit_or_apply(ab, c), do: Nx.Defn.jit_or_apply(&defn_jit/2, [ab, c])

    test "jits or applies" do
      assert Nx.Defn.jit_or_apply(&defn_jit/2, [{4, 5}, 3]) == Nx.tensor(6)
      assert defn_jit_or_apply({4, 5}, 3) == Nx.tensor(6)
    end

    def elixir_jit({a, b}, c) do
      true = Process.get(Nx.Defn.Compiler) in [Evaluator, Identity]
      a |> Nx.add(b) |> Nx.subtract(c)
    end

    test "compiles elixir function" do
      assert Nx.Defn.jit(&elixir_jit/2, [{4, 5}, 3]) == Nx.tensor(6)
      assert Nx.Defn.jit(&elixir_jit/2, [{4, 5}, Nx.tensor(3)]) == Nx.tensor(6)
      assert Nx.Defn.jit(&elixir_jit(&1, 3), [{4, 5}]) == Nx.tensor(6)

      assert %T{data: %Expr{op: :subtract}} =
               Nx.Defn.jit(&elixir_jit/2, [{4, 5}, 3], compiler: Identity)
    end

    test "raises if it doesn't return an expression" do
      assert_raise ArgumentError,
                   "defn must return a tensor expression or a tuple, got: :ok",
                   fn ->
                     Nx.Defn.jit(fn -> :ok end, [], compiler: Evaluator).()
                   end
    end

    defn jit_iota(), do: Nx.iota({3, 3})

    @tag :capture_log
    test "uses the default backend on iota" do
      Nx.default_backend(UnknownBackend)
      assert_raise UndefinedFunctionError, fn -> Nx.Defn.jit(&jit_iota/0, []) end
      assert_raise UndefinedFunctionError, fn -> Nx.Defn.jit(fn -> Nx.iota({3, 3}) end, []) end
    end
  end

  describe "compilation errors" do
    test "invalid numerical expression" do
      assert_raise CompileError, ~r"#{location(+5)}: invalid numerical expression", fn ->
        defmodule Sample do
          import Nx.Defn

          defn add(_a, _b) do
            receive do
              :ok -> :ok
            end
          end
        end
      end
    end

    test "non-variables used as arguments" do
      assert_raise CompileError,
                   ~r"#{location(+4)}: only variables, tuples, and maps are allowed as patterns in defn",
                   fn ->
                     defmodule Sample do
                       import Nx.Defn
                       defn add(1, 2), do: 3
                     end
                   end
    end

    test "dup vars used as arguments" do
      assert_raise CompileError,
                   ~r"#{location(+4)}: variable \"a\" appears twice in pattern \[a, a\]",
                   fn ->
                     defmodule Sample do
                       import Nx.Defn
                       defn add(a, a), do: 3
                     end
                   end
    end

    test "dup vars used as patterns" do
      assert_raise CompileError,
                   ~r"#{location(+4)}: variable \"b\" appears twice in pattern \{b, b\}",
                   fn ->
                     defmodule Sample do
                       import Nx.Defn
                       defn add(a), do: {b, b} = a
                     end
                   end
    end

    test "invalid defn compiler" do
      assert_raise ArgumentError,
                   ~r"expected @defn_compiler/@default_defn_compiler to be an atom or",
                   fn ->
                     defmodule Sample do
                       @defn_compiler "unknown"
                       import Nx.Defn
                       defn add(a, b), do: a + b
                     end
                   end
    end

    test "invalid default defn compiler" do
      assert_raise ArgumentError,
                   ~r"expected @defn_compiler/@default_defn_compiler to be an atom or",
                   fn ->
                     defmodule Sample do
                       @default_defn_compiler "unknown"
                       import Nx.Defn
                       defn add(a, b), do: a + b
                     end
                   end
    end
  end

  @default_defn_compiler Evaluator

  describe "default arguments" do
    defn sum_axis_opts(a, opts \\ []), do: Nx.sum(a, opts)
    defn local_calls_sum_axis_opts(a), do: sum_axis_opts(a)
    defn remote_calls_sum_axis_opts(a), do: __MODULE__.sum_axis_opts(a)

    test "are supported" do
      assert sum_axis_opts(Nx.tensor([[1, 2], [3, 4]])) == Nx.tensor(10)
      assert sum_axis_opts(Nx.tensor([[1, 2], [3, 4]]), axes: [0]) == Nx.tensor([4, 6])
      assert sum_axis_opts(Nx.tensor([[1, 2], [3, 4]]), axes: [1]) == Nx.tensor([3, 7])
    end

    test "can be called within defn" do
      assert local_calls_sum_axis_opts(Nx.tensor([[1, 2], [3, 4]])) == Nx.tensor(10)
      assert remote_calls_sum_axis_opts(Nx.tensor([[1, 2], [3, 4]])) == Nx.tensor(10)
    end

    defn random_opts(opts \\ []), do: Nx.random_uniform({}, 0, 1, opts)

    test "exclusively" do
      assert random_opts([]).type == {:s, 64}
      assert random_opts(type: {:f, 64}).type == {:f, 64}
    end

    @defn_compiler Identity
    defn sum_axis_expr(a, opts \\ []), do: Nx.sum(a, opts)

    test "have their own cache key" do
      sum_axis_expr(Nx.tensor([[1, 2], [3, 4]]), axes: [0])
      key0 = Process.get(Identity)
      assert is_function(key0, 1)

      sum_axis_expr(Nx.tensor([[1, 2], [3, 4]]), axes: [1])
      key1 = Process.get(Identity)
      assert is_function(key1, 1)

      sum_axis_expr(Nx.tensor([[1, 2], [3, 4]]), axes: [0])
      assert Process.get(Identity) == key0
    end
  end

  describe "private definitions" do
    defnp private(a, b), do: a + b
    defn calls_private(a, b), do: private(a, b)

    test "are supported" do
      assert private(1, 2) == Nx.tensor(3)
    end

    test "are not exported" do
      refute function_exported?(__MODULE__, :private, 2)
    end

    test "are callable from defn" do
      assert calls_private(1, 2) == Nx.tensor(3)
    end
  end
end
