defmodule Nx.Defn.ExprTest do
  use ExUnit.Case, async: true

  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T
  doctest Nx.Defn.Expr

  import Nx.Defn

  setup do
    Nx.Defn.default_options(compiler: Nx.Defn.Debug)
    :ok
  end

  describe "constant optimizations" do
    test "broadcast" do
      assert %T{data: %Expr{op: :constant, args: [1.0]}, type: {:f, 32}, shape: {1, 2, 3}} =
               Nx.broadcast(Expr.tensor(1.0), {1, 2, 3})

      param = Expr.parameter(nil, {:s, 32}, {2, 2}, 1)

      assert %T{
               data: %Expr{
                 op: :add,
                 args: [
                   %T{shape: {2, 2}, data: %Expr{op: :constant, args: [1.0]}},
                   %T{shape: {2, 2}, data: %Expr{op: :parameter}}
                 ]
               },
               shape: {2, 2},
               type: {:f, 32}
             } = Nx.add(Nx.broadcast(Expr.tensor(1.0), {2, 2}), param)
    end

    test "as_type" do
      assert %T{data: %Expr{op: :constant, args: [1.0]}, type: {:f, 32}, shape: {}} =
               Nx.as_type(Expr.tensor(1), {:f, 32})

      assert %T{data: %Expr{op: :constant, args: [1]}, type: {:s, 32}, shape: {}} =
               Nx.as_type(Expr.tensor(1.0), {:s, 32})
    end

    test "add" do
      assert %T{data: %Expr{op: :constant, args: [3.0]}, type: {:f, 32}} =
               Nx.add(Expr.tensor(1.0), Expr.tensor(2))

      param1 = Expr.parameter(nil, {:s, 32}, {2, 2}, 1)
      param2 = Expr.parameter(nil, {:s, 32}, {2, 2}, 2)

      assert %T{data: %Expr{op: :as_type, args: [^param1]}, type: {:f, 32}} =
               Nx.add(param1, Expr.tensor(0.0))

      assert %T{data: %Expr{op: :as_type, args: [^param1]}, type: {:f, 32}} =
               Nx.add(Expr.tensor(0.0), param1)

      assert %T{data: %Expr{op: :broadcast, args: [^param1, {2, 2, 2}, [1, 2]]}} =
               Nx.add(Nx.broadcast(Expr.tensor(0), {2, 2, 2}), param1)

      assert %T{data: %Expr{op: :add, args: [_, ^param1]}, type: {:f, 32}} =
               Nx.add(param1, Expr.tensor(1.0))

      assert %T{data: %Expr{op: :subtract, args: [^param1, ^param2]}, type: {:f, 32}} =
               Nx.add(param1, Nx.subtract(Expr.tensor(0.0), param2))
    end

    test "subtract" do
      param = Expr.parameter(nil, {:s, 32}, {2, 2}, 2)

      assert ^param = Nx.subtract(param, Expr.tensor(0))

      assert %T{data: %Expr{op: :as_type, args: [^param]}, type: {:f, 32}} =
               Nx.subtract(param, Expr.tensor(0.0))
    end

    test "multiply" do
      assert %T{data: %Expr{op: :constant, args: [4.0]}, type: {:f, 32}} =
               Nx.multiply(Expr.tensor(2.0), Expr.tensor(2))

      param1 = Expr.parameter(nil, {:s, 32}, {2, 2}, 1)
      param2 = Expr.parameter(nil, {:s, 32}, {2, 2}, 2)

      assert %T{data: %Expr{op: :as_type, args: [^param1]}, type: {:f, 32}} =
               Nx.multiply(param1, Expr.tensor(1.0))

      assert %T{data: %Expr{op: :as_type, args: [^param1]}, type: {:f, 32}} =
               Nx.multiply(Expr.tensor(1.0), param1)

      assert %T{data: %Expr{op: :broadcast, args: [^param1, {2, 2, 2}, [1, 2]]}} =
               Nx.multiply(Nx.broadcast(Expr.tensor(1), {2, 2, 2}), param1)

      assert %T{data: %Expr{op: :multiply, args: [_, ^param1]}, type: {:f, 32}} =
               Nx.multiply(param1, Expr.tensor(2.0))

      assert %T{data: %Expr{op: :divide, args: [^param1, ^param2]}, type: {:f, 32}} =
               Nx.multiply(param1, Nx.divide(Expr.tensor(1.0), param2))
    end

    test "divide" do
      param = Expr.parameter(nil, {:s, 32}, {2, 2}, 2)

      assert %T{data: %Expr{op: :as_type, args: [^param]}, type: {:f, 32}} =
               Nx.divide(param, Expr.tensor(1.0))

      assert %T{data: %Expr{op: :as_type, args: [^param]}, type: {:f, 32}} =
               Nx.divide(param, Expr.tensor(1))
    end

    test "pow" do
      param = Expr.parameter(nil, {:s, 32}, {2, 2}, 2)

      assert ^param = Nx.pow(param, Expr.tensor(1))

      assert %T{data: %Expr{op: :as_type, args: [^param]}, type: {:f, 32}} =
               Nx.pow(param, Expr.tensor(1.0))
    end

    test "commute" do
      param1 = Expr.parameter(nil, {:s, 32}, {2, 2}, 1)
      param2 = Expr.parameter(nil, {:s, 32}, {2}, 2)
      param3 = Expr.parameter(nil, {:s, 32}, {2}, 3)

      assert %T{
               data: %Expr{
                 op: :add,
                 args: [
                   %T{data: %Expr{op: :constant, args: [3.0]}, shape: {}, type: {:f, 32}},
                   %T{data: %Expr{op: :add, args: [^param2, ^param1]}, shape: {2, 2}}
                 ]
               },
               type: {:f, 32},
               shape: {2, 2}
             } = param1 |> Nx.add(Expr.tensor(1.0)) |> Nx.add(param2) |> Nx.add(Expr.tensor(2))

      assert %T{
               data: %Expr{
                 op: :add,
                 args: [
                   %T{data: %Expr{op: :constant, args: [3.0]}, shape: {}, type: {:f, 32}},
                   %T{
                     data: %Expr{
                       op: :broadcast,
                       args: [
                         %T{data: %Expr{op: :add, args: [^param3, ^param2]}, shape: {2}},
                         {2, 2},
                         [1]
                       ]
                     },
                     shape: {2, 2}
                   }
                 ]
               },
               type: {:f, 32},
               shape: {2, 2}
             } =
               param2
               |> Nx.add(Nx.broadcast(Expr.tensor(1.0), {2, 2}))
               |> Nx.add(param3)
               |> Nx.add(Expr.tensor(2))

      assert %T{
               data: %Expr{
                 op: :add,
                 args: [
                   %T{data: %Expr{op: :constant, args: [3.0]}, shape: {}, type: {:f, 32}},
                   %T{
                     data: %Expr{
                       op: :broadcast,
                       args: [
                         %T{data: %Expr{op: :add, args: [^param3, ^param2]}, shape: {2}},
                         {2, 2},
                         [1]
                       ]
                     },
                     shape: {2, 2}
                   }
                 ]
               },
               type: {:f, 32},
               shape: {2, 2}
             } =
               param2
               |> Nx.add(Expr.tensor(1.0))
               |> Nx.add(param3)
               |> Nx.add(Nx.broadcast(Expr.tensor(2), {2, 2}))
    end

    test "preserves names" do
      named = Nx.tensor([4], names: [:dim])
      assert %T{type: {:f, 32}, names: [:dim]} = Nx.multiply(Expr.tensor(named), Expr.tensor(1.0))
    end

    test "logsumexp" do
      expr = Nx.logsumexp(Expr.tensor(Nx.tensor([1, 2, 3, 4, 5, 6])))

      assert inspect(expr) =~ """
               tensor a                                       s32[6]
               b = reduce_max a, axes: [0], keep_axes: true   s32[1]
               c = metadata b, :stop_grad                     s32[1]
             """
    end

    test "upcast float constants when operating against higher precision types" do
      t_f32 = Nx.tensor([2, 2], type: :f32) |> Expr.tensor()
      c_f64 = Expr.constant(Nx.tensor(0.7, type: :f64), 0.7, [])

      assert %T{type: {:f, 64}, data: %Expr{op: :multiply, args: [^c_f64, ^t_f32]}} =
               Nx.multiply(t_f32, c_f64)

      t_f64 = Nx.tensor([2, 2], type: :f64) |> Expr.tensor()
      c_f32 = Expr.constant(Nx.tensor(0.7, type: :f32), 0.7, [])

      assert %T{type: {:f, 64}, data: %Expr{op: :multiply, args: [^c_f64, ^t_f64]}} =
               Nx.multiply(t_f64, c_f32)

      c_c64 = Expr.constant(Nx.tensor(0.7, type: :c64), 0.7, [])
      c_c128 = Expr.constant(Nx.tensor(0.7, type: :c128), 0.7, [])

      assert %T{type: {:c, 64}, data: %Expr{op: :multiply, args: [^c_c64, ^t_f32]}} =
               Nx.multiply(t_f32, c_c64)

      assert %T{type: {:c, 128}, data: %Expr{op: :multiply, args: [^c_c128, ^t_f64]}} =
               Nx.multiply(t_f64, c_c64)
    end
  end

  describe "inspect" do
    test "with scalar" do
      assert inspect(Expr.tensor(Nx.tensor(0)), safe: false) == """
             #Nx.Tensor<
               s32
             \s\s
               Nx.Defn.Expr
               0
             >\
             """
    end

    test "with parameters" do
      a = Expr.parameter(nil, {:s, 32}, {2, 2}, 0)
      b = Expr.parameter(nil, {:s, 32}, {2, 2}, 1)

      assert Nx.sum(Nx.add(Nx.add(Nx.dot(a, a), Nx.tanh(b)), 2))
             |> inspect(safe: false) == """
             #Nx.Tensor<
               f32
             \s\s
               Nx.Defn.Expr
               parameter a:0                            s32[2][2]
               parameter c:1                            s32[2][2]
               b = dot a, [1], [], a, [0], []           s32[2][2]
               d = tanh c                               f32[2][2]
               e = add b, d                             f32[2][2]
               f = add 2, e                             f32[2][2]
               g = sum f, axes: nil, keep_axes: false   f32
             >\
             """
    end

    test "with tensors" do
      a = Expr.parameter(nil, {:s, 32}, {2, 2}, 2)
      b = Nx.tensor([[1, 2], [1, 2]])
      c = Nx.iota({2, 2}, backend: Expr)

      assert Nx.argmin(Nx.add(Nx.tanh(Nx.dot(c, b)), a), tie_break: :high)
             |> inspect(safe: false) == """
             #Nx.Tensor<
               s32
             \s\s
               Nx.Defn.Expr
               tensor b                                                      s32[2][2]
               parameter e:2                                                 s32[2][2]
               a = iota nil                                                  s32[2][2]
               c = dot a, [1], [], b, [0], []                                s32[2][2]
               d = tanh c                                                    f32[2][2]
               f = add d, e                                                  f32[2][2]
               g = argmin f, tie_break: :high, axis: nil, keep_axis: false   s32
             >\
             """
    end

    test "with fun" do
      a = Expr.parameter(nil, {:s, 32}, {2, 2}, 0)

      assert Nx.reduce(a, 0, [], &Nx.add/2) |> inspect(safe: false) == """
             #Nx.Tensor<
               s32
             \s\s
               Nx.Defn.Expr
               parameter a:0                                             s32[2][2]
               b = reduce a, 0, axes: nil, keep_axes: false, &Nx.add/2   s32
             >\
             """
    end

    test "with metadata" do
      a = Expr.parameter(nil, {:s, 32}, {}, 0)

      assert Expr.metadata(a, %{}) |> Nx.add(1) |> inspect(safe: false) == """
             #Nx.Tensor<
               s32
             \s\s
               Nx.Defn.Expr
               parameter a:0   s32
               b = add 1, a    s32
             >\
             """

      assert Expr.metadata(a, %{inspect: :foo}) |> Nx.add(1) |> inspect(safe: false) == """
             #Nx.Tensor<
               s32
             \s\s
               Nx.Defn.Expr
               parameter a:0          s32
               b = metadata a, :foo   s32
               c = add 1, b           s32
             >\
             """
    end

    test "with tuple output" do
      a = Expr.parameter(nil, {:s, 32}, {2, 2}, 0)
      {q, r} = Nx.LinAlg.qr(a)

      assert inspect(q, safe: false) == """
             #Nx.Tensor<
               f32[2][2]
             \s\s
               Nx.Defn.Expr
               parameter a:0                            s32[2][2]
               b = qr a, eps: 1.0e-10, mode: :reduced   tuple2
               c = elem b, 0                            f32[2][2]
             >\
             """

      assert inspect(r, safe: false) == """
             #Nx.Tensor<
               f32[2][2]
             \s\s
               Nx.Defn.Expr
               parameter a:0                            s32[2][2]
               b = qr a, eps: 1.0e-10, mode: :reduced   tuple2
               c = elem b, 1                            f32[2][2]
             >\
             """
    end

    test "with tuple and cond" do
      a = Expr.parameter(nil, {:s, 32}, {}, 0)
      b = Expr.parameter(nil, {:s, 32}, {}, 1)
      {left, right} = Expr.cond([{Nx.any(a), {a, b}}], {b, a})

      assert inspect(left, safe: false) == """
             #Nx.Tensor<
               s32
             \s\s
               Nx.Defn.Expr
               parameter a:0                            s32
               parameter c:1                            s32
               b = any a, axes: nil, keep_axes: false   u8
               d = cond b -> {a, c}, true -> {c, a}     tuple2
               e = elem d, 0                            s32
             >\
             """

      assert inspect(right, safe: false) == """
             #Nx.Tensor<
               s32
             \s\s
               Nx.Defn.Expr
               parameter a:0                            s32
               parameter c:1                            s32
               b = any a, axes: nil, keep_axes: false   u8
               d = cond b -> {a, c}, true -> {c, a}     tuple2
               e = elem d, 1                            s32
             >\
             """
    end

    defn factorial(x) do
      {factorial, _} =
        while {factorial = 1.0, x}, Nx.greater(x, 1) do
          {factorial * x, x - 1}
        end

      factorial
    end

    test "with while" do
      assert inspect(factorial(Nx.template({}, {:f, 32})), safe: false) == """
             #Nx.Tensor<
               f32
             \s\s
               Nx.Defn.Expr
               parameter a:0        f32
               b = while {1.0, a}   tuple2
               c = elem b, 0        f32
             >\
             """
    end

    defn sub_add_mult(a, b) do
      token = create_token()
      {token, add} = hook_token(token, a + b, :add, &IO.inspect({:add, &1}))
      {token, mult} = hook_token(token, a * b, :mult, &IO.inspect({:mult, &1}))
      {add, mult} = attach_token(token, {add, mult})
      add - mult
    end

    test "with tokens" do
      result = sub_add_mult(Nx.template({}, {:f, 32}), Nx.template({}, {:f, 32}))

      assert inspect(result, safe: false) == """
             #Nx.Tensor<
               f32
             \s\s
               Nx.Defn.Expr
               parameter a:0               f32
               parameter b:1               f32
               c = multiply a, b           f32
               d = add a, b                f32
               e = token mult: c, add: d   tuple2
               f = attach_token e, d       f32
               g = attach_token e, c       f32
               h = subtract f, g           f32
             >\
             """
    end

    defn add_sub_mult_no_tokens(a, b, c, d) do
      a
      |> Nx.add(b)
      |> Nx.subtract(c)
      |> Nx.multiply(d)
    end

    test "with limit option" do
      t = Nx.template({}, :f32)

      result = add_sub_mult_no_tokens(t, t, t, t)

      full_expr = """
      #Nx.Tensor<
        f32
      \s\s
        Nx.Defn.Expr
        parameter a:0       f32
        parameter b:1       f32
        parameter d:2       f32
        parameter f:3       f32
        c = add a, b        f32
        e = subtract c, d   f32
        g = multiply e, f   f32
      >\
      """

      # infinity
      assert inspect(result, limit: :infinity) == full_expr
      # greater than the number of exprs
      assert inspect(result, limit: 8) == full_expr
      # equal to the number of exprs
      assert inspect(result, limit: 7) == full_expr

      # one less than the number of exprs
      assert inspect(result, limit: 6) == """
             #Nx.Tensor<
               f32
             \s\s
               Nx.Defn.Expr
               parameter a:0       f32
               parameter b:1       f32
               parameter d:2       f32
               ...                \s
               c = add a, b        f32
               e = subtract c, d   f32
               g = multiply e, f   f32
             >\
             """

      assert inspect(result, limit: 3) == """
             #Nx.Tensor<
               f32
             \s\s
               Nx.Defn.Expr
               ...                \s
               c = add a, b        f32
               e = subtract c, d   f32
               g = multiply e, f   f32
             >\
             """

      assert inspect(result, limit: 1) == """
             #Nx.Tensor<
               f32
             \s\s
               Nx.Defn.Expr
               ...                \s
               c = multiply a, b   f32
             >\
             """

      assert inspect(result, limit: 0) == """
             #Nx.Tensor<
               f32
             \s\s
               Nx.Defn.Expr
               ...  \s
             >\
             """
    end
  end
end
