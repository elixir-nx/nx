defmodule Nx.DefnTest do
  use ExUnit.Case, async: true

  alias Nx.Tensor, as: T
  alias Nx.Defn.{Expr, Debug, Evaluator}
  alias Nx.DefnTest.Sample
  import Nx.Defn

  defmacrop location(plus) do
    file = Path.relative_to_cwd(__CALLER__.file)
    quote do: "#{unquote(file)}:#{unquote(__CALLER__.line) + unquote(plus)}"
  end

  setup context do
    Nx.Defn.default_options(compiler: context[:compiler] || Debug)
    :ok
  end

  describe "doctest" do
    @describetag compiler: Evaluator
    doctest Nx.Defn
  end

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

    defn nested_tuple_shape_match_signature({a, {b, c}}) do
      a + b + c
    end

    test "allows nested pattern matching on the tuple shape on signature" do
      assert %T{shape: {}, type: {:f, 32}, data: %Expr{op: :add, args: [_, _]}} =
               nested_tuple_shape_match_signature({1, {2.0, 3.0}})
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

    defn verify_maps(map), do: verify_maps_transform(map)

    deftransformp verify_maps_transform(map) do
      for {k, v} <- map do
        assert Elixir.Kernel.==(v.shape, {String.to_integer(k)})
      end

      map
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

    defn calls_binary_funs({funa, funb}, a, b), do: {funa.(a, b), funb.(a, b)}

    test "receives multiple anonymous functions in tuple" do
      assert {fun_left, fun_right} = calls_binary_funs({&Nx.add/2, &Nx.subtract/2}, 1, 2.0)

      %T{shape: {}, type: {:f, 32}, data: %Expr{op: :add, args: [left, right]}} = fun_left
      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:f, 32}} = right

      %T{shape: {}, type: {:f, 32}, data: %Expr{op: :subtract, args: [left, right]}} = fun_right
      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:f, 32}} = right
    end
  end

  describe "unary ops" do
    defn exp(t), do: Nx.exp(t)

    defn unary_plus_minus_guards(opts \\ []) do
      case opts[:value] do
        value when value == -2 ->
          -1

        value when value == +2 ->
          1
      end
    end

    defn unary_plus_minus_match(opts \\ []) do
      case opts[:value] do
        -2 ->
          -1

        +2 ->
          1
      end
    end

    test "to expr" do
      assert %T{shape: {3}, type: {:f, 32}, data: %Expr{op: :exp, args: [_]}} =
               exp(Nx.tensor([1, 2, 3]))

      assert %T{shape: {3}, type: {:f, 32}, data: %Expr{op: :exp, args: [_]}} =
               exp(Nx.tensor([1, 2, 3], type: {:f, 32}))

      assert %T{shape: {3}, type: {:bf, 16}, data: %Expr{op: :exp, args: [_]}} =
               exp(Nx.tensor([1, 2, 3], type: {:bf, 16}))
    end

    test "unary + and - work with guards and hard matches" do
      assert Expr.tensor(-1) == unary_plus_minus_guards(value: -2)
      assert Expr.tensor(1) == unary_plus_minus_guards(value: 2)
      assert Expr.tensor(-1) == unary_plus_minus_match(value: -2)
      assert Expr.tensor(1) == unary_plus_minus_match(value: 2)
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
    defn iota(t), do: Nx.iota(Nx.shape(t))
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

    test "raise an error given a shape tuple with tensor values" do
      assert_raise ArgumentError,
                   ~r"invalid dimension in axis 0 found in shape.*if you are trying to pass a dimension or a shape as an argument to a defn function"s,
                   fn -> iota({10}) end
    end
  end

  describe "tensor ops" do
    defn dot2(t1, t2), do: Nx.dot(t1, t2)
    defn dot6(t1, t2), do: Nx.dot(t1, [-2], [], t2, [-1], [])
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

    test "transpose" do
      assert %T{data: %Expr{op: :transpose, args: [_, [1, 0]]}, shape: {3, 2}} =
               transpose_1(Nx.tensor([[1, 2, 3], [1, 2, 3]]))

      assert %T{data: %Expr{op: :transpose, args: [_, [1, 0]]}, shape: {3, 2}} =
               transpose_2(Nx.tensor([[1, 2, 3], [1, 2, 3]]))
    end

    test "reshape" do
      assert %T{data: %Expr{op: :reshape, args: [_]}, shape: {2, 3}} =
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

    defn broadcast_collapse1(t), do: t |> Nx.broadcast({5, 3}) |> Nx.broadcast({7, 5, 3})

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
      true and a
    end

    test "and" do
      assert %T{data: %Expr{op: :logical_and, args: [_, _]}} = land_two(1, 2)
    end

    defn lor_two(a, b), do: a or b

    defn lor_true(opts \\ []) do
      true or opts[:value]
    end

    test "or" do
      assert %T{data: %Expr{op: :logical_or, args: [_, _]}} = lor_two(1, 2)
    end

    defn lnot(a), do: not a
    defn lnot_boolean(opts \\ []), do: not constant_boolean_transform(opts)

    deftransformp constant_boolean_transform(opts) do
      if opts[:value] == true do
        true
      else
        false
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

  describe "optional" do
    defn cumsum(t), do: Nx.cumulative_sum(t)

    @tag compiler: Evaluator
    test "cummulative" do
      assert cumsum(Nx.tensor([1, 2, 3, 4])) == Nx.tensor([1, 3, 6, 10])
      assert cumsum(Nx.tensor([1, 2, 3, 4], names: [:x])) == Nx.tensor([1, 3, 6, 10], names: [:x])
    end
  end

  describe "lu" do
    defn lu(t), do: Nx.LinAlg.lu(t)

    test "returns tuples" do
      assert {p, l, u} = lu(Nx.iota({3, 3}))

      assert %T{data: %Expr{op: :elem, args: [lu_expr, 0]}, shape: {3, 3}} = p
      assert %T{data: %Expr{op: :elem, args: [^lu_expr, 1]}, shape: {3, 3}} = l
      assert %T{data: %Expr{op: :elem, args: [^lu_expr, 2]}, shape: {3, 3}} = u
    end
  end

  describe "qr" do
    defn qr(t), do: Nx.LinAlg.qr(t)

    test "returns tuples" do
      assert {left, right} = qr(Nx.iota({3, 2}))

      assert %T{data: %Expr{op: :elem, args: [qr_expr, 0]}, shape: {3, 2}} = left
      assert %T{data: %Expr{op: :elem, args: [^qr_expr, 1]}, shape: {2, 2}} = right
    end
  end

  describe "svd" do
    defn svd(t), do: Nx.LinAlg.svd(t)

    test "returns tuples" do
      assert {u, s, vt} = svd(Nx.iota({3, 3}))

      assert %T{data: %Expr{op: :elem, args: [svd_expr, 0]}, shape: {3, 3}} = u
      assert %T{data: %Expr{op: :elem, args: [^svd_expr, 1]}, shape: {3}} = s
      assert %T{data: %Expr{op: :elem, args: [^svd_expr, 2]}, shape: {3, 3}} = vt
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

    defn add_two_io(a, b), do: IO.inspect({a, b})

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

    test "IO remote" do
      assert_raise RuntimeError,
                   "cannot invoke IO.inspect/1 inside defn because it was not defined with defn. " <>
                     "To print the runtime value of a tensor, use print_value/2. " <>
                     "To print the tensor expression, use print_expr/2",
                   fn -> add_two_io(1, 2) end
    end
  end

  describe "if" do
    defn if3(a, b, c), do: if(a, do: b, else: c)
    defn if2(a, b), do: if(a, do: b)
    defn if_map(a, b, c), do: if(a, do: %{foo: b}, else: %{foo: c})
    defn if_scalar_error(a), do: if(a, do: {1, 2}, else: {3})

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

    test "preserves maps" do
      assert %{foo: %T{data: %Expr{op: :cond}, shape: {}, type: {:s, 64}}} =
               if_map(Nx.tensor(0), Nx.tensor(1), Nx.tensor(2))
    end

    defn if_branch_elimination(a) do
      if Nx.rank(a) == 0 do
        11
      else
        13
      end
    end

    test "eliminates branches" do
      assert %T{data: %Expr{op: :constant, args: [11]}} = if_branch_elimination(Nx.tensor(0))

      assert %T{data: %Expr{op: :constant, args: [13]}} =
               if_branch_elimination(Nx.tensor([1, 2, 3]))
    end

    defn if_branch_elimination_complex(a) do
      if elem(Nx.shape(a), Nx.rank(a) - 1) == 1 do
        11
      else
        13
      end
    end

    test "eliminates complex branches" do
      assert %T{data: %Expr{op: :constant, args: [11]}} =
               if_branch_elimination_complex(Nx.iota({1, 1}))

      assert %T{data: %Expr{op: :constant, args: [13]}} =
               if_branch_elimination_complex(Nx.iota({3, 3}))
    end

    defn if_branch_elimination_transform(a) do
      if if_branch_elimination_transform_pred(a) do
        11
      else
        13
      end
    end

    deftransformp if_branch_elimination_transform_pred(a) do
      if Nx.rank(a) == 0, do: 1, else: 0
    end

    test "eliminates branches from transform" do
      assert %T{data: %Expr{op: :constant, args: [11]}} =
               if_branch_elimination_transform(Nx.tensor(0))

      assert %T{data: %Expr{op: :constant, args: [13]}} =
               if_branch_elimination_transform(Nx.tensor([1, 2, 3]))
    end

    defn if_assignment(a, b) do
      d =
        if c = Nx.all(a) do
          b
        else
          -b
        end

      c + d
    end

    @tag compiler: Evaluator
    test "returns assignment from if" do
      assert if_assignment(Nx.tensor([1, 2, 3]), Nx.tensor([1, 2, 3])) == Nx.tensor([2, 3, 4])
      assert if_assignment(Nx.tensor([0, 1, 2]), Nx.tensor([1, 2, 3])) == Nx.tensor([-1, -2, -3])
    end

    defn if_boolean(opts \\ []) do
      if opts[:boolean] do
        1
      else
        -1
      end
    end

    test "evaluates boolean predicates" do
      assert if_boolean(boolean: true) == Expr.tensor(1)
      assert if_boolean(boolean: false) == Expr.tensor(-1)

      assert_raise CompileError,
                   ~r"cond in defn expects the predicate to be true, false, or a scalar tensor",
                   fn -> if_boolean(boolean: nil) end
    end

    defn if_boolean_raise(opts \\ []) do
      if opts[:boolean] do
        1
      else
        raise ArgumentError, "oops"
      end
    end

    test "raises lazily" do
      assert if_boolean_raise(boolean: true) == Expr.tensor(1)
      assert_raise ArgumentError, "oops", fn -> if_boolean_raise(boolean: false) end
    end

    test "raises correct error on incompatible shapes" do
      assert_raise CompileError, ~r/cond\/if expects all branches/, fn ->
        if_scalar_error(Nx.tensor(0))
      end
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
      if Nx.any(a), do: 1, else: -1
    end

    @tag compiler: Evaluator
    test "supports literals" do
      assert cond_lit(Nx.tensor(0)) == Nx.tensor(-1)
      assert cond_lit(Nx.tensor(1)) == Nx.tensor(1)
    end

    defn cond_empty_map(a) do
      if Nx.any(a), do: %{}, else: %{}
    end

    test "handles empty maps in branches" do
      assert cond_empty_map(Nx.tensor(0)) == %{}
      assert cond_empty_map(Nx.tensor(0)) == %{}
    end

    defn cond_branch_elimination(a, b) do
      cond do
        Nx.rank(b) != 0 -> 7
        Nx.rank(a) == 0 -> 11
        true -> 13
      end
    end

    test "eliminates branches" do
      assert %T{data: %Expr{op: :constant, args: [7]}} =
               cond_branch_elimination(Nx.tensor(0), Nx.tensor([1, 2, 3]))

      assert %T{data: %Expr{op: :constant, args: [7]}} =
               cond_branch_elimination(Nx.tensor([1, 2, 3]), Nx.tensor([1, 2, 3]))

      assert %T{data: %Expr{op: :constant, args: [11]}} =
               cond_branch_elimination(Nx.tensor(0), Nx.tensor(0))

      assert %T{data: %Expr{op: :constant, args: [13]}} =
               cond_branch_elimination(Nx.tensor([1, 2, 3]), Nx.tensor(0))
    end

    defn cond_branch_variable_access(a) do
      cond do
        b = Nx.all(a > 0) -> a + b
        b = Nx.all(a < 0) -> a - b
        true -> a
      end
    end

    @tag compiler: Evaluator
    test "access variable from condition" do
      assert cond_branch_variable_access(Nx.tensor([1, 2, 3])) == Nx.tensor([2, 3, 4])
      assert cond_branch_variable_access(Nx.tensor([-1, -2, -3])) == Nx.tensor([-2, -3, -4])
      assert cond_branch_variable_access(Nx.tensor([-1, 0, 1])) == Nx.tensor([-1, 0, 1])
    end

    defn bad_cond(a) do
      cond do
        Nx.any(a) -> +a
        Nx.all(a) -> -a
      end
    end

    test "raises if cond is missing last atom clause" do
      assert_raise CompileError,
                   ~r"cond/if expects at least one branch to always evaluate to true",
                   fn -> bad_cond(1) end
    end

    test "raises if given a non-scalar as condition" do
      assert_raise CompileError, ~r"condition must be a scalar tensor, got:", fn ->
        cond4(Nx.tensor([0, 0]), Nx.tensor(1), Nx.tensor(2), Nx.tensor(3))
      end
    end

    defn non_tensor_cond(a) do
      if Nx.any(a) do
        :foo
      else
        :bar
      end
    end

    test "raises on non-tensor return" do
      assert_raise CompileError,
                   ~r"cond/if expects all branches to return compatible tensor types. Got: :foo and :bar",
                   fn -> non_tensor_cond(1) end
    end
  end

  describe "case/2" do
    defn simple_rank(tensor) do
      case Nx.shape(tensor) do
        {} -> 0
        {_} -> 1
        {_, _} -> 2
      end
    end

    defn tuple_rank(tensorA, tensorB) do
      case {Nx.shape(tensorA), Nx.shape(tensorB)} do
        {{}, {}} -> 0
        {{_}, {}} -> -1
        {{}, {_}} -> 1
      end
    end

    defn guard_rank(tensor) do
      case Nx.shape(tensor) do
        {x, x} -> 0
        {x, y} when x > y -> -1
        {x, y} when x < y -> 1
      end
    end

    defn struct_case(opts \\ []) do
      case opts[:struct] do
        %URI{} -> 0
        %Regex{} -> 1
        %Nx.Tensor{} -> 2
        _ -> 3
      end
    end

    deftransformp uri, do: %URI{}

    defn invalid_case(tensor) do
      case tensor do
        _ -> 0
      end
    end

    test "matches shapes" do
      assert simple_rank(123) == Expr.tensor(0)
      assert simple_rank(Nx.tensor([1, 2, 3])) == Expr.tensor(1)
      assert simple_rank(Nx.tensor([[1, 2, 3]])) == Expr.tensor(2)
    end

    test "matches tuples shapes" do
      assert tuple_rank(0, 0) == Expr.tensor(0)
      assert tuple_rank(Nx.tensor([1, 2, 3]), 0) == Expr.tensor(-1)
      assert tuple_rank(0, Nx.tensor([1, 2, 3])) == Expr.tensor(1)
    end

    test "matches struct names" do
      assert struct_case(struct: %URI{}) == Expr.tensor(0)
      assert struct_case(struct: ~r/foo/) == Expr.tensor(1)
      assert struct_case(struct: Nx.tensor([1, 2, 3])) == Expr.tensor(2)
      assert struct_case(struct: 123) == Expr.tensor(3)
    end

    test "matches using guards" do
      assert guard_rank(Nx.iota({2, 2})) == Expr.tensor(0)
      assert guard_rank(Nx.iota({3, 2})) == Expr.tensor(-1)
      assert guard_rank(Nx.iota({2, 3})) == Expr.tensor(1)
    end

    test "raises if not tuple, atom, integer, or struct" do
      assert_raise ArgumentError,
                   ~r"only tuples, atoms, numbers, and structs are allowed as arguments to case/2 inside defn",
                   fn -> invalid_case(%{}) end
    end

    test "raises on outside variables in guards" do
      assert_raise CompileError,
                   ~r"case/2 in defn allow guards to only access variables defined in patterns",
                   fn ->
                     defmodule Fail do
                       defn invalid_case(y) do
                         case y do
                           x when x > y -> 1
                         end
                       end
                     end
                   end
    end
  end

  describe "raise/2" do
    defn raise_argument_error(tensor) do
      case Nx.shape(tensor) do
        {n, n} -> n
        shape -> raise ArgumentError, "expected a square tensor: #{inspect(shape)}"
      end
    end

    test "raises ArgumentError" do
      assert raise_argument_error(Nx.iota({4, 4})) == Expr.tensor(4)

      assert_raise ArgumentError, "expected a square tensor: {4, 3}", fn ->
        raise_argument_error(Nx.iota({4, 3}))
      end
    end

    defn raise_runtime_error(tensor) do
      case Nx.shape(tensor) do
        {n, n} -> n
        shape -> raise "expected a square tensor: " <> inspect(shape)
      end
    end

    test "raises RuntimeError" do
      assert raise_runtime_error(Nx.iota({4, 4})) == Expr.tensor(4)

      assert_raise RuntimeError, "expected a square tensor: {4, 3}", fn ->
        raise_runtime_error(Nx.iota({4, 3}))
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
      assert %T{data: %Expr{op: :constant, args: [1]}, shape: {}, type: {:s, 64}} = body
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
                   ~r/the do-block in while must return tensors with the same shape, type, and names as the initial arguments. Got body \{f32, f32\} and initial \{s64, f32\}/,
                   fn -> factorial(10.0) end
    end

    defn add_complex(t) do
      Nx.complex(t, 2)
    end

    test "complex numbers" do
      t = add_complex(1)

      assert inspect(t) ==
               """
               #Nx.Tensor<
                 c64\n\s\s
                 Nx.Defn.Expr
                 parameter a:0         s64
                 b = add 0.0+2.0i, a   c64
               >
               """
               |> String.trim()
    end

    defn while_last(t) do
      {_, last, _} =
        while {i = 0, _last = 0, t}, i <= t do
          {i + 1, i, t}
        end

      last
    end

    @tag compiler: Evaluator
    test "underscored vars" do
      assert while_last(10) == Nx.tensor(10)
    end

    defn while_generator_sum(tensor) do
      while acc = tensor[0] * 0, part <- tensor do
        acc + part
      end
    end

    @tag compiler: Evaluator
    test "tensor generator" do
      assert while_generator_sum(Nx.tensor([0, 1, 2, 3, 4])) == Nx.tensor(10)
      assert while_generator_sum(Nx.iota({3, 3})) == Nx.tensor([9, 12, 15])
    end

    defn while_generator_sum_unroll(tensor) do
      while acc = 0, part <- tensor, unroll: true do
        acc + part
      end
    end

    test "tensor generator expression exact" do
      expr = while_generator_sum(Nx.tensor([0, 1, 2, 3, 4]))

      assert inspect(expr) == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               parameter a:0                s64[5]
               b = slice a, [0], [1], [1]   s64[1]
               c = squeeze b, [0]           s64
               d = multiply 0, c            s64
               e = while {0, a, d}          tuple3
               f = elem e, 2                s64
             >\
             """

      %T{
        data: %Expr{
          op: :elem,
          args: [%T{data: %Expr{op: :while, args: [_, arg, _, while_body]}}, 2]
        }
      } = expr

      assert inspect(arg) == """
             {#Nx.Tensor<
                s64
              \s\s
                Nx.Defn.Expr
                parameter a:0   s64
              >, #Nx.Tensor<
                s64[5]
              \s\s
                Nx.Defn.Expr
                parameter a:1   s64[5]
              >, #Nx.Tensor<
                s64
              \s\s
                Nx.Defn.Expr
                parameter a:2   s64
              >}\
             """

      {counter, param, body} = while_body

      assert inspect(counter) == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               parameter a:0   s64
               b = add 1, a    s64
             >\
             """

      assert inspect(param) ==
               """
               #Nx.Tensor<
                 s64[5]
               \s\s
                 Nx.Defn.Expr
                 parameter a:1   s64[5]
               >\
               """

      assert inspect(body) ==
               """
               #Nx.Tensor<
                 s64
               \s\s
                 Nx.Defn.Expr
                 parameter a:2                s64
                 parameter b:1                s64[5]
                 parameter c:0                s64
                 d = slice b, [c], [1], [1]   s64[1]
                 e = squeeze d, [0]           s64
                 f = add a, e                 s64
               >\
               """
    end

    test "tensor generator unrolled" do
      assert while_generator_sum_unroll(Nx.tensor([1, 2, 3])) |> inspect() == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               parameter a:0                s64[3]
               b = slice a, [0], [1], [1]   s64[1]
               c = squeeze b, [0]           s64
               d = slice a, [1], [1], [1]   s64[1]
               e = squeeze d, [0]           s64
               f = add c, e                 s64
               g = slice a, [2], [1], [1]   s64[1]
               h = squeeze g, [0]           s64
               i = add f, h                 s64
             >\
             """
    end

    defn while_generator_sum_unroll_int(tensor) do
      while acc = 0, part <- tensor, unroll: 3 do
        acc + part
      end
    end

    @tag compiler: Evaluator
    test "tensor generator unrolled integer result" do
      assert while_generator_sum_unroll_int(Nx.tensor([1, 2, 3])) == Nx.tensor(6)
      assert while_generator_sum_unroll_int(Nx.tensor([1, 2, 3, 4, 5, 6])) == Nx.tensor(21)
      assert while_generator_sum_unroll_int(Nx.tensor([1, 2, 3, 4, 5, 6, 7, 8])) == Nx.tensor(36)
    end

    test "tensor generator unrolled integer expression exact" do
      expr = while_generator_sum_unroll_int(Nx.tensor([1, 2, 3, 4, 5, 6]))

      assert inspect(expr) == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               parameter a:0         s64[6]
               b = while {0, a, 0}   tuple3
               c = elem b, 2         s64
             >\
             """

      %T{
        data: %Expr{
          op: :elem,
          args: [%T{data: %Expr{op: :while, args: [_, _, _, while_body]}}, 2]
        }
      } = expr

      {counter, param, body} = while_body

      assert inspect(counter) == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               parameter a:0   s64
               b = add 3, a    s64
             >\
             """

      assert inspect(param) == """
             #Nx.Tensor<
               s64[6]
             \s\s
               Nx.Defn.Expr
               parameter a:1   s64[6]
             >\
             """

      assert inspect(body) == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               parameter a:2                s64
               parameter b:1                s64[6]
               parameter c:0                s64
               d = slice b, [c], [1], [1]   s64[1]
               e = squeeze d, [0]           s64
               f = add a, e                 s64
               g = add 1, c                 s64
               h = slice b, [g], [1], [1]   s64[1]
               i = squeeze h, [0]           s64
               j = add f, i                 s64
               k = add 2, c                 s64
               l = slice b, [k], [1], [1]   s64[1]
               m = squeeze l, [0]           s64
               n = add j, m                 s64
             >\
             """
    end

    test "tensor generator unrolled integer expression left-over" do
      assert while_generator_sum_unroll_int(Nx.tensor([1, 2, 3])) |> inspect() == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               parameter a:0                s64[3]
               b = slice a, [0], [1], [1]   s64[1]
               c = squeeze b, [0]           s64
               d = slice a, [1], [1], [1]   s64[1]
               e = squeeze d, [0]           s64
               f = add c, e                 s64
               g = slice a, [2], [1], [1]   s64[1]
               h = squeeze g, [0]           s64
               i = add f, h                 s64
             >\
             """

      assert while_generator_sum_unroll_int(Nx.tensor([1, 2, 3, 4, 5, 6, 7, 8])) |> inspect() ==
               """
               #Nx.Tensor<
                 s64
               \s\s
                 Nx.Defn.Expr
                 parameter a:0                s64[8]
                 b = while {0, a, 0}          tuple3
                 c = elem b, 2                s64
                 d = slice a, [6], [1], [1]   s64[1]
                 e = squeeze d, [0]           s64
                 f = add c, e                 s64
                 g = slice a, [7], [1], [1]   s64[1]
                 h = squeeze g, [0]           s64
                 i = add f, h                 s64
               >\
               """
    end

    defn while_generator_range(acc, opts \\ []) do
      opts = keyword!(opts, [:range, unroll: false])

      while acc, part <- opts[:range], unroll: opts[:unroll] do
        acc + part
      end
    end

    @tag compiler: Evaluator
    test "range generator" do
      # Increasing ranges
      assert while_generator_range(0, range: 0..4) == Nx.tensor(10)
      assert while_generator_range(100, range: 1..10) == Nx.tensor(155)
      assert while_generator_range(0, range: 0..4, unroll: true) == Nx.tensor(10)
      assert while_generator_range(100, range: 1..10, unroll: true) == Nx.tensor(155)
      assert while_generator_range(0, range: 0..4, unroll: 3) == Nx.tensor(10)
      assert while_generator_range(100, range: 1..10, unroll: 3) == Nx.tensor(155)

      # Empty ranges
      assert while_generator_range(0, range: 0..-1//1) == Nx.tensor(0)
      assert while_generator_range(100, range: 0..-1//1) == Nx.tensor(100)
      assert while_generator_range(0, range: 0..-1//1, unroll: true) == Nx.tensor(0)
      assert while_generator_range(100, range: 0..-1//1, unroll: true) == Nx.tensor(100)
      assert while_generator_range(0, range: 0..-1//1, unroll: 3) == Nx.tensor(0)
      assert while_generator_range(100, range: 0..-1//1, unroll: 3) == Nx.tensor(100)

      # Decreasing ranges
      assert while_generator_range(0, range: 4..0//-1) == Nx.tensor(10)
      assert while_generator_range(100, range: 10..1//-1) == Nx.tensor(155)
      assert while_generator_range(0, range: 4..0//-1, unroll: true) == Nx.tensor(10)
      assert while_generator_range(100, range: 10..1//-1, unroll: true) == Nx.tensor(155)
      assert while_generator_range(0, range: 4..0//-1, unroll: 3) == Nx.tensor(10)
      assert while_generator_range(100, range: 10..1//-1, unroll: 3) == Nx.tensor(155)
    end

    test "range generator unrolled" do
      assert while_generator_range(0, range: 1..3, unroll: true) |> inspect() == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               parameter a:0   s64
               b = add 6, a    s64
             >\
             """
    end

    test "range generator unrolled integer expression exact" do
      expr = while_generator_range(0, range: 1..6, unroll: 3)

      assert inspect(expr) == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               parameter a:0      s64
               b = while {1, a}   tuple2
               c = elem b, 1      s64
             >\
             """

      %T{
        data: %Expr{
          op: :elem,
          args: [%T{data: %Expr{op: :while, args: [_, _, _, while_body]}}, 1]
        }
      } = expr

      {counter, body} = while_body

      assert inspect(counter) == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               parameter a:0   s64
               b = add 3, a    s64
             >\
             """

      assert inspect(body) == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               parameter a:0   s64
               parameter b:1   s64
               c = add b, a    s64
               d = add c, a    s64
               e = add a, d    s64
               f = add 1, e    s64
               g = add 2, f    s64
             >\
             """
    end

    test "range generator unrolled integer expression left-over" do
      assert while_generator_range(0, range: 1..3, unroll: 3) |> inspect() == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               parameter a:0   s64
               b = add 6, a    s64
             >\
             """

      assert while_generator_range(0, range: 1..8, unroll: 3) |> inspect() ==
               """
               #Nx.Tensor<
                 s64
               \s\s
                 Nx.Defn.Expr
                 parameter a:0      s64
                 b = while {1, a}   tuple2
                 c = elem b, 1      s64
                 d = add 15, c      s64
               >\
               """

      assert while_generator_range(0, range: 8..1//-1, unroll: 3) |> inspect() ==
               """
               #Nx.Tensor<
                 s64
               \s\s
                 Nx.Defn.Expr
                 parameter a:0      s64
                 b = while {8, a}   tuple2
                 c = elem b, 1      s64
                 d = add 3, c       s64
               >\
               """
    end

    defn while_mixed_return(a, b) do
      while {a, b}, Nx.less(a, 10) do
        %{a: a, b: b}
      end
    end

    test "raises on mixed return" do
      assert_raise CompileError,
                   ~r/the do-block in while must return tensors with the same shape, type, and names as the initial arguments. Got body %\{:a => s64, :b => s64\} and initial \{s64, s64\}/,
                   fn -> while_mixed_return(Nx.tensor(0), Nx.tensor(1)) end
    end

    defn while_mixed_context(a, b) do
      while a, Nx.less(a, 10) do
        a + b
      end
    end

    test "raises on mixed context" do
      assert_raise RuntimeError,
                   ~r"cannot build defn because expressions come from different contexts: :root and {:while, #Reference<[^>]+>}",
                   fn -> while_mixed_context(Nx.tensor(0), Nx.tensor(1)) end
    end

    defn while_condition_opts(a) do
      while a, Nx.less(a, 10), unroll: true do
        a + 1
      end
    end

    test "raises on options given to condition" do
      assert_raise CompileError,
                   ~r"options are not supported on while with conditions, only with generators, got: \[unroll: true\]",
                   fn -> while_condition_opts(Nx.tensor(0)) end
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
                   ~r"expected last argument of \"while\" to be a do-block, got: a \+ 1",
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
      (Nx.tanh(a) + Nx.power(b, 3)) |> print_expr()
    end

    defn transform_inspect_label(a, b) do
      (Nx.tanh(a) + Nx.power(b, 3)) |> print_expr(label: "HELLO")
    end

    test "executes the transformation" do
      assert ExUnit.CaptureIO.capture_io(fn -> transform_inspect(1, 2) end) == """
             #Nx.Tensor<
               f32
             \s\s
               Nx.Defn.Expr
               parameter a:0    s64
               parameter c:1    s64
               b = tanh a       f32
               d = power c, 3   s64
               e = add b, d     f32
             >
             """

      assert ExUnit.CaptureIO.capture_io(fn -> transform_inspect_label(1, 2) end) == """
             HELLO: #Nx.Tensor<
               f32
             \s\s
               Nx.Defn.Expr
               parameter a:0    s64
               parameter c:1    s64
               b = tanh a       f32
               d = power c, 3   s64
               e = add b, d     f32
             >
             """
    end

    defn transform_back_and_forth(a) do
      a |> Nx.negate() |> private_back_and_forth() |> Nx.exp()
    end

    deftransformp private_back_and_forth(a) do
      Evaluator = Nx.Defn.Compiler.current()
      final_back_and_forth(a)
    end

    defn final_back_and_forth(a), do: Nx.tanh(a)

    @tag compiler: Evaluator
    test "back and forth between Elixir and defn" do
      assert transform_back_and_forth(Nx.tensor(1)) ==
               Nx.tensor(1) |> Nx.negate() |> Nx.tanh() |> Nx.exp()
    end
  end

  describe "debug expression" do
    defn defn_debug(a, b), do: a + b

    test "debug_expr/2" do
      assert %Nx.Tensor{data: %Nx.Defn.Expr{op: :add, args: [left, right]}} =
               Nx.Defn.debug_expr(&defn_debug/2).(1, 2)

      assert %Nx.Tensor{data: %Nx.Defn.Expr{op: :parameter, args: [0]}} = left
      assert %Nx.Tensor{data: %Nx.Defn.Expr{op: :parameter, args: [1]}} = right
    end

    test "debug_expr_apply/3" do
      assert %Nx.Tensor{data: %Nx.Defn.Expr{op: :add, args: [left, right]}} =
               Nx.Defn.debug_expr_apply(&defn_debug/2, [1, 2])

      assert %Nx.Tensor{data: %Nx.Defn.Expr{op: :parameter, args: [0]}} = left
      assert %Nx.Tensor{data: %Nx.Defn.Expr{op: :parameter, args: [1]}} = right
    end
  end

  describe "jit" do
    defn defn_jit({a, b}, c), do: a + b - c

    test "compiles defn function" do
      assert %T{data: %Expr{op: :subtract}} =
               Nx.Defn.jit(&defn_jit/2, compiler: Debug).({1, 2}, 3)

      Nx.Defn.default_options(compiler: Evaluator)
      assert Nx.Defn.jit(&defn_jit/2).({4, 5}, 3) == Nx.tensor(6)
      assert Nx.Defn.jit(&defn_jit/2).({4, 5}, Nx.tensor(3)) == Nx.tensor(6)
      assert Nx.Defn.jit(&defn_jit(&1, 3)).({4, 5}) == Nx.tensor(6)
    end

    @tag compiler: Evaluator
    test "applies with options" do
      fun = Nx.Defn.jit(&Nx.sum/2)
      assert fun.(Nx.iota({2, 2}), axes: [1]) == Nx.tensor([1, 5])
    end

    defn defn_jit_or_apply(ab, c),
      do: Nx.Defn.jit_apply(&defn_jit/2, [ab, c], on_conflict: :reuse)

    test "jits or applies" do
      assert %T{data: %Expr{op: :subtract}} =
               Nx.Defn.jit_apply(&defn_jit_or_apply/2, [{1, 2}, 3],
                 compiler: Debug,
                 on_conflict: :reuse
               )

      Nx.Defn.default_options(compiler: Evaluator)
      assert Nx.Defn.jit_apply(&defn_jit/2, [{4, 5}, 3], on_conflict: :reuse) == Nx.tensor(6)
      assert defn_jit_or_apply({4, 5}, 3) == Nx.tensor(6)
    end

    def elixir_jit({a, b}, c) do
      true = Process.get(Nx.Defn.Compiler) in [Evaluator, Debug]
      a |> Nx.add(b) |> Nx.subtract(c)
    end

    test "compiles elixir function" do
      assert %T{data: %Expr{op: :subtract}} =
               Nx.Defn.jit(&elixir_jit/2, compiler: Debug).({4, 5}, 3)

      Nx.Defn.default_options(compiler: Evaluator)
      assert Nx.Defn.jit(&elixir_jit/2).({4, 5}, 3) == Nx.tensor(6)
      assert Nx.Defn.jit(&elixir_jit/2).({4, 5}, Nx.tensor(3)) == Nx.tensor(6)
      assert Nx.Defn.jit(&elixir_jit(&1, 3)).({4, 5}) == Nx.tensor(6)
    end

    defp elixir_constant_jit() do
      %Nx.Tensor{data: %Nx.Defn.Expr{}} = Nx.to_tensor(0)
    end

    test "compiles elixir function with default backend for constants" do
      assert Nx.Defn.jit(&elixir_constant_jit/0, compiler: Evaluator).() == Nx.tensor(0)
    end

    test "raises if it doesn't return an expression" do
      assert_raise Protocol.UndefinedError,
                   fn -> Nx.Defn.jit(fn -> :ok end, compiler: Evaluator).() end
    end

    defn jit_iota(), do: Nx.iota({3, 3})

    @tag :capture_log
    @tag compiler: Evaluator
    test "uses the default backend on iota" do
      Nx.default_backend(UnknownBackend)
      assert_raise UndefinedFunctionError, fn -> Nx.Defn.jit(&jit_iota/0).() end
      assert_raise UndefinedFunctionError, fn -> Nx.Defn.jit(fn -> Nx.iota({3, 3}) end).() end
    end

    defn nested_jit(opts \\ []), do: nested_jit_transform(opts)

    deftransformp nested_jit_transform(opts) do
      eleven = Nx.tensor(11, backend: Nx.BinaryBackend)
      Nx.Defn.jit(&Nx.Defn.Kernel.*/2, opts).(eleven, eleven)
    end

    @tag compiler: Evaluator
    test "raises on nested JIT unless forcing" do
      assert_raise RuntimeError,
                   "cannot invoke JITed function when there is a JIT compilation happening",
                   fn -> nested_jit() end

      assert nested_jit(on_conflict: :force) == Nx.tensor(11 * 11)
    end
  end

  describe "compile" do
    defn defn_compile({a, b}, c), do: a + b - c

    @tag compiler: Evaluator
    test "compiles defn function" do
      fun = Nx.Defn.compile(&defn_compile/2, [{4, 5}, 3])
      assert fun.({4, 5}, 3) == Nx.tensor(6)
      assert fun.({40, 50}, 30) == Nx.tensor(60)

      fun = Nx.Defn.compile(&defn_compile(&1, 3), [{4, 5}])
      assert fun.({4, 5}) == Nx.tensor(6)
      assert fun.({40, 50}) == Nx.tensor(87)
    end

    @tag compiler: Evaluator
    test "compiles defn function with options" do
      fun = Nx.Defn.compile(&Nx.sum/2, [Nx.template({2, 2}, :s64), [axes: [1]]])
      assert fun.(Nx.iota({2, 2})) == Nx.tensor([1, 5])
    end

    @tag compiler: Evaluator
    test "raises on incompatible shape" do
      fun = Nx.Defn.compile(&defn_compile/2, [{4, 5}, 3])

      assert_raise ArgumentError,
                   ~r"argument at position 1 is not compatible with compiled function template",
                   fn -> fun.(3, {4, 5}) end
    end

    defn nested_compile(opts \\ []), do: nested_compile_transform(opts)

    deftransformp nested_compile_transform(opts) do
      eleven = Nx.tensor(11, backend: Nx.BinaryBackend)
      Nx.Defn.compile(&Nx.Defn.Kernel.*/2, [eleven, eleven], opts).(eleven, eleven)
    end

    @tag compiler: Evaluator
    test "raises when nested on JIT" do
      assert_raise RuntimeError,
                   "cannot invoke compiled function when there is a JIT compilation happening",
                   fn -> nested_compile() end
    end
  end

  describe "compilation errors" do
    test "undefined local function" do
      assert_raise CompileError,
                   ~r"#{location(+6)}: undefined function do_add/2 \(there is no such import\)",
                   fn ->
                     defmodule Sample do
                       import Nx.Defn

                       defn add(a, b) do
                         do_add(a, b)
                       end
                     end
                   end
    end

    test "non-defn local function" do
      assert_raise CompileError,
                   ~r"#{location(+6)}: cannot use function do_add/2 inside defn because it was not defined with defn",
                   fn ->
                     defmodule Sample do
                       import Nx.Defn

                       defn add(a, b) do
                         do_add(a, b)
                       end

                       defp do_add(a, b), do: a + b
                     end
                   end
    end

    test "non-variables used as arguments" do
      assert_raise CompileError,
                   ~r"#{location(+4)}: only variables, tuples, maps, and structs are allowed as patterns in defn",
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
  end

  describe "default arguments" do
    defn id_default(empty \\ {}), do: empty
    defn sum_axis_opts(a, opts \\ []), do: Nx.sum(a, opts)
    defn local_calls_sum_axis_opts(a), do: sum_axis_opts(a)
    defn remote_calls_sum_axis_opts(a), do: __MODULE__.sum_axis_opts(a)

    @tag compiler: Evaluator
    test "are supported" do
      assert sum_axis_opts(Nx.tensor([[1, 2], [3, 4]])) == Nx.tensor(10)
      assert sum_axis_opts(Nx.tensor([[1, 2], [3, 4]]), axes: [0]) == Nx.tensor([4, 6])
      assert sum_axis_opts(Nx.tensor([[1, 2], [3, 4]]), axes: [1]) == Nx.tensor([3, 7])
    end

    @tag compiler: Evaluator
    test "accept any valid defn value" do
      assert id_default() == {}
      assert id_default(1) == Nx.tensor(1)
      assert id_default({1}) == {Nx.tensor(1)}
    end

    @tag compiler: Evaluator
    test "can be called within defn" do
      assert local_calls_sum_axis_opts(Nx.tensor([[1, 2], [3, 4]])) == Nx.tensor(10)
      assert remote_calls_sum_axis_opts(Nx.tensor([[1, 2], [3, 4]])) == Nx.tensor(10)
    end

    defn random_opts(opts \\ []), do: Nx.random_uniform({}, 0, 1, opts)

    @tag compiler: Evaluator
    test "exclusively" do
      assert random_opts([]).type == {:s, 64}
      assert random_opts(type: {:f, 64}).type == {:f, 64}
    end

    defn sum_axis_expr(a, opts \\ []), do: Nx.sum(a, opts)

    test "have their own cache key" do
      sum_axis_expr(Nx.tensor([[1, 2], [3, 4]]), axes: [0])
      key0 = Process.get(Debug)
      assert is_function(key0, 1)

      sum_axis_expr(Nx.tensor([[1, 2], [3, 4]]), axes: [1])
      key1 = Process.get(Debug)
      assert is_function(key1, 1)

      sum_axis_expr(Nx.tensor([[1, 2], [3, 4]]), axes: [0])
      assert Process.get(Debug) == key0
    end
  end

  describe "private definitions" do
    defnp private(a, b), do: a + b
    defn calls_private(a, b), do: private(a, b)

    @tag compiler: Evaluator
    test "are supported" do
      assert private(1, 2) == Nx.tensor(3)
    end

    @tag compiler: Evaluator
    test "are not exported" do
      refute function_exported?(__MODULE__, :private, 2)
    end

    @tag compiler: Evaluator
    test "are callable from defn" do
      assert calls_private(1, 2) == Nx.tensor(3)
    end

    test "warns when unused" do
      assert ExUnit.CaptureIO.capture_io(:stderr, fn ->
               defmodule Example do
                 import Nx.Defn
                 defnp add(a, b), do: a + b
               end
             end) =~ "function add/2 is unused"
    end
  end

  describe "deftransform" do
    defn deftransform_test(a, b, opts \\ []) do
      {c, d} = deftransform_test_transformation(opts)

      deftransform_test_send(c)
      deftransform_test_send(d)

      x = a + c
      y = b + d

      x * y
    end

    deftransform deftransform_test_transformation(opts) do
      b = opts[:b] || raise "missing :b"
      c = opts[:c] || raise "missing :c"
      {b, c}
    end

    deftransformp deftransform_test_send(value) do
      send(self(), {:deftransform, value})
    end

    defn default_args1(x) do
      default_args(x)
    end

    defn default_args2(x, y) do
      default_args(x, y)
    end

    defn default_args3(x, y, z) do
      default_args(x, y, z)
    end

    defn public_default_args1(x), do: public_default_args(x)
    defn public_default_args2(x, y), do: public_default_args(x, y)

    # Ensure that defp works with defaults
    deftransformp default_args(arg1 \\ 1, arg2, arg3 \\ 3) do
      {arg1, arg2, arg3}
    end

    # Ensure that def works
    deftransform public_default_args(arg1, arg2 \\ 2), do: {arg1, arg2}

    # Ensure multi-clause and guards work
    deftransform multi_clause_transform(number, fun) when is_function(fun, 1) do
      fun.(number)
    end

    deftransform multi_clause_transform(x, y) when is_number(x) and is_number(y) and y > 0 do
      x ** y
    end

    defn multi_clause_first(x), do: multi_clause_transform(x, &(&1 + &1))
    defn multi_clause_second(opts \\ []), do: multi_clause_transform(opts[:x], opts[:y])

    deftransform(multi_clause_bodiless_tf(x \\ 1, y))
    deftransform multi_clause_bodiless_tf(1, y), do: y
    deftransform multi_clause_bodiless_tf(x, _y), do: x

    defn multi_clause_transform_bodiless1(a), do: multi_clause_bodiless_tf(a)

    defn multi_clause_transform_bodiless2(opts \\ []),
      do: multi_clause_bodiless_tf(opts[:a], opts[:b])

    deftransformp(multi_clause_bodiless_tf_private(x \\ 1, y))
    deftransformp multi_clause_bodiless_tf_private(1, y), do: y
    deftransformp multi_clause_bodiless_tf_private(x, _y), do: x

    defn multi_clause_transform_bodiless3(a), do: multi_clause_bodiless_tf_private(a)

    defn multi_clause_transform_bodiless4(opts \\ []),
      do: multi_clause_bodiless_tf_private(opts[:a], opts[:b])

    test "can call deftransform and deftransformp functions from within defn" do
      result = deftransform_test(Nx.tensor(1), Nx.tensor(2), b: 3, c: 4)

      assert String.trim("""
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               parameter a:0       s64
               parameter c:1       s64
               b = add 3, a        s64
               d = add 4, c        s64
               e = multiply b, d   s64
             >
             """) == inspect(result)

      assert_received {:deftransform, 3}
      assert_received {:deftransform, 4}
    end

    @tag compiler: Evaluator
    test "deftransformp handles default arguments" do
      assert default_args1(20) == {Nx.tensor(1), Nx.tensor(20), Nx.tensor(3)}
      assert default_args2(10, 20) == {Nx.tensor(10), Nx.tensor(20), Nx.tensor(3)}
      assert default_args3(10, 20, 30) == {Nx.tensor(10), Nx.tensor(20), Nx.tensor(30)}
    end

    @tag compiler: Evaluator
    test "deftransform handles default arguments" do
      assert public_default_args1(10) == {Nx.tensor(10), Nx.tensor(2)}
      assert public_default_args2(10, 20) == {Nx.tensor(10), Nx.tensor(20)}
    end

    @tag compiler: Evaluator
    test "multi-clause and guards work for deftransform" do
      assert multi_clause_first(2) == Nx.tensor(4)
      assert multi_clause_second(x: 2, y: 3) == Nx.tensor(8)
    end

    test "multi-clause raises for no clause matching args" do
      assert_raise FunctionClauseError,
                   "no function clause matching in Nx.DefnTest.multi_clause_transform/2",
                   fn ->
                     multi_clause_second(x: 2, y: -3)
                   end
    end

    @tag compiler: Evaluator
    test "multi-clause deftransform and deftransformp with bodiless head" do
      assert Nx.tensor(10) == multi_clause_transform_bodiless1(10)
      assert Nx.tensor(10) == multi_clause_transform_bodiless2(a: 1, b: 10)
      assert Nx.tensor(20) == multi_clause_transform_bodiless2(a: 20, b: 10)

      assert Nx.tensor(10) == multi_clause_transform_bodiless3(10)
      assert Nx.tensor(10) == multi_clause_transform_bodiless4(a: 1, b: 10)
      assert Nx.tensor(20) == multi_clause_transform_bodiless4(a: 20, b: 10)
    end
  end

  describe "boolean values" do
    defn supports_booleans(opts \\ []) do
      l = opts[:l]
      r = opts[:r]

      Nx.stack([l < r, l <= r, l > r, l >= r, l == r, l != r, l and r, l or r, not l, not r])
    end

    @tag compiler: Evaluator
    test "supports booleans in some operations" do
      for l <- [true, false, 1, 0], r <- [true, false, 1, 0] do
        lb = l == 1 or l == true
        rb = r == 1 or r == true

        ln = if lb, do: 1, else: 0
        rn = if rb, do: 1, else: 0

        assert supports_booleans(l: l, r: r) ==
                 Nx.tensor([
                   ln < rn,
                   ln <= rn,
                   ln > rn,
                   ln >= rn,
                   ln == rn,
                   ln != rn,
                   lb and rb,
                   lb or rb,
                   not lb,
                   not rb
                 ])
      end
    end
  end
end
