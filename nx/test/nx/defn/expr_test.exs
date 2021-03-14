defmodule Nx.Defn.ExprTest do
  use ExUnit.Case, async: true

  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T

  describe "scalar optimizations" do
    test "broadcast" do
      assert %T{data: %Expr{op: :scalar, args: [1.0]}, type: {:f, 32}, shape: {1, 2, 3}} =
               Nx.broadcast(Expr.tensor(1.0), {1, 2, 3})

      param = Expr.parameter(nil, {:s, 64}, {2, 2}, 1)

      assert %T{
               data: %Expr{
                 op: :add,
                 args: [
                   %T{shape: {}, data: %Expr{op: :scalar, args: [1.0]}},
                   %T{shape: {2, 2}, data: %Expr{op: :parameter}}
                 ]
               },
               shape: {2, 2},
               type: {:f, 32}
             } = Nx.add(Nx.broadcast(Expr.tensor(1.0), {2, 2}), param)
    end

    test "as_type" do
      assert %T{data: %Expr{op: :scalar, args: [1.0]}, type: {:f, 32}, shape: {}} =
               Nx.as_type(Expr.tensor(1), {:f, 32})
    end

    test "add" do
      assert %T{data: %Expr{op: :scalar, args: [3.0]}, type: {:f, 32}} =
               Nx.add(Expr.tensor(1.0), Expr.tensor(2))

      param1 = Expr.parameter(nil, {:s, 64}, {2, 2}, 1)
      param2 = Expr.parameter(nil, {:s, 64}, {2, 2}, 2)

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
      param = Expr.parameter(nil, {:s, 64}, {2, 2}, 2)

      assert ^param = Nx.subtract(param, Expr.tensor(0))

      assert %T{data: %Expr{op: :as_type, args: [^param]}, type: {:f, 32}} =
               Nx.subtract(param, Expr.tensor(0.0))
    end

    test "multiply" do
      assert %T{data: %Expr{op: :scalar, args: [4.0]}, type: {:f, 32}} =
               Nx.multiply(Expr.tensor(2.0), Expr.tensor(2))

      param1 = Expr.parameter(nil, {:s, 64}, {2, 2}, 1)
      param2 = Expr.parameter(nil, {:s, 64}, {2, 2}, 2)

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
      param = Expr.parameter(nil, {:s, 64}, {2, 2}, 2)

      assert %T{data: %Expr{op: :as_type, args: [^param]}, type: {:f, 32}} =
               Nx.divide(param, Expr.tensor(1.0))

      assert %T{data: %Expr{op: :as_type, args: [^param]}, type: {:f, 32}} =
               Nx.divide(param, Expr.tensor(1))
    end

    test "power" do
      param = Expr.parameter(nil, {:s, 64}, {2, 2}, 2)

      assert ^param = Nx.power(param, Expr.tensor(1))

      assert %T{data: %Expr{op: :as_type, args: [^param]}, type: {:f, 32}} =
               Nx.power(param, Expr.tensor(1.0))
    end

    test "commute" do
      param1 = Expr.parameter(nil, {:s, 64}, {2, 2}, 1)
      param2 = Expr.parameter(nil, {:s, 64}, {2}, 2)
      param3 = Expr.parameter(nil, {:s, 64}, {2}, 3)

      assert %T{
               data: %Expr{
                 op: :add,
                 args: [
                   %T{data: %Expr{op: :scalar, args: [3.0]}, shape: {}, type: {:f, 32}},
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
                   %T{data: %Expr{op: :scalar, args: [3.0]}, shape: {}, type: {:f, 32}},
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
                   %T{data: %Expr{op: :scalar, args: [3.0]}, shape: {}, type: {:f, 32}},
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
  end

  describe "inspect" do
    test "with scalar" do
      assert inspect(Expr.tensor(Nx.tensor(0)), safe: false) == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
             >\
             """
    end

    test "with parameters" do
      a = Expr.parameter(nil, {:s, 64}, {2, 2}, 0)
      b = Expr.parameter(nil, {:s, 64}, {2, 2}, 1)

      assert Nx.sum(Nx.add(Nx.add(Nx.dot(a, a), Nx.tanh(b)), 2))
             |> inspect(safe: false) == """
             #Nx.Tensor<
               f32
             \s\s
               Nx.Defn.Expr
               parameter a                                 s64[2][2]
               parameter c                                 s64[2][2]
               b = dot [ a, [1], a, [0] ]                  s64[2][2]
               d = tanh [ c ]                              f32[2][2]
               e = add [ b, d ]                            f32[2][2]
               f = add [ 2, e ]                            f32[2][2]
               g = sum [ f, axes: nil, keep_axes: false ]  f32
             >\
             """
    end

    test "with tensors" do
      a = Expr.parameter(nil, {:s, 64}, {2, 2}, 2)
      b = Nx.tensor([[1, 2], [1, 2]])
      c = Nx.iota({2, 2}, backend: Expr)

      assert Nx.argmin(Nx.add(Nx.tanh(Nx.dot(c, b)), a), tie_break: :high)
             |> inspect(safe: false) == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               tensor b                                       s64[2][2]
               parameter e                                    s64[2][2]
               a = iota [ nil ]                               s64[2][2]
               c = dot [ a, [1], b, [0] ]                     s64[2][2]
               d = tanh [ c ]                                 f32[2][2]
               f = add [ d, e ]                               f32[2][2]
               g = argmin [ f, tie_break: :high, axis: nil ]  s64
             >\
             """
    end

    test "with fun" do
      a = Expr.parameter(nil, {:s, 64}, {2, 2}, 0)

      assert Nx.reduce(a, 0, [], &Nx.add/2) |> inspect(safe: false) == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               parameter a                                                  s64[2][2]
               b = reduce [ a, 0, axes: nil, keep_axes: false, &Nx.add/2 ]  s64
             >\
             """
    end

    test "with metadata" do
      a = Expr.parameter(nil, {:s, 64}, {}, 0)

      assert Expr.metadata(a, %{foo: true}) |> inspect(safe: false) == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               parameter a                 s64
               b = metadata [ a, [:foo] ]  s64
             >\
             """
    end

    test "with tuple and cond" do
      a = Expr.parameter(nil, {:s, 64}, {}, 0)
      b = Expr.parameter(nil, {:s, 64}, {}, 1)
      {left, right} = Expr.cond([{Nx.any?(a), {a, b}}], {b, a})

      assert inspect(left, safe: false) == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               parameter a                                     s64
               parameter c                                     s64
               b = any? [ a, axes: nil, keep_axes: false ]     u8
               d = cond [ b -> {a, c}, :otherwise -> {c, a} ]  tuple2
               e = elem [ d, 0, 2 ]                            s64
             >\
             """

      assert inspect(right, safe: false) == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               parameter a                                     s64
               parameter c                                     s64
               b = any? [ a, axes: nil, keep_axes: false ]     u8
               d = cond [ b -> {a, c}, :otherwise -> {c, a} ]  tuple2
               e = elem [ d, 1, 2 ]                            s64
             >\
             """
    end
  end
end
