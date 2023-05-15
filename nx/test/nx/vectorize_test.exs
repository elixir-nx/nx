defmodule Nx.VectorizeTest do
  use ExUnit.Case, async: true

  import Nx, only: :sigils
  import Nx.Defn

  @base Nx.tensor([
          [[0, 1, 2]],
          [[3, 4, 5]],
          [[6, 7, 8]]
        ])

  @vectorized Nx.vectorize(@base, :rows)

  @base_unary Nx.tensor([
                [[0.1, 0.2, 0.3]],
                [[0.4, 0.5, 0.6]],
                [[0.7, 0.8, 0.9]]
              ])

  @vectorized_unary Nx.vectorize(@base_unary, :rows)

  describe "vectorize" do
    test "adds new vectorization axes to the end of the list" do
      v = Nx.vectorize(@vectorized, :cols)
      assert v.vectorized_axes == [rows: 3, cols: 1]
      assert v.shape == {3}
    end
  end

  describe "unary math ops" do
    for {name, _} <- Nx.Shared.unary_math_funs(), name != :acosh do
      test "Nx.#{name}/1 works on vectorized tensor" do
        result =
          @base_unary
          |> Nx.unquote(name)()
          |> Nx.vectorize(:rows)

        assert result == Nx.unquote(name)(@vectorized_unary)
      end
    end

    test "Nx.acosh/1 works on vectorized tensor" do
      result =
        @base_unary
        |> Nx.add(1)
        |> Nx.acosh()
        |> Nx.vectorize(:rows)

      assert result == Nx.acosh(Nx.add(@vectorized_unary, 1))
    end
  end

  describe "binary operations" do
    test "left addition by scalar" do
      result = Nx.add(2, @vectorized)
      assert result.shape == {1, 3}

      assert result ==
               Nx.vectorize(
                 Nx.tensor([[[2, 3, 4]], [[5, 6, 7]], [[8, 9, 10]]]),
                 :rows
               )
    end

    test "right addition by scalar" do
      result = Nx.add(@vectorized, 2)
      assert result.shape == {1, 3}

      assert result ==
               Nx.vectorize(
                 Nx.tensor([[[2, 3, 4]], [[5, 6, 7]], [[8, 9, 10]]]),
                 :rows
               )
    end

    test "left addition by rank-1" do
      result = Nx.add(Nx.tensor([2]), @vectorized)
      assert result.shape == {1, 3}

      assert result ==
               Nx.vectorize(
                 Nx.tensor([[[2, 3, 4]], [[5, 6, 7]], [[8, 9, 10]]]),
                 :rows
               )
    end

    test "right addition by rank-1" do
      result = Nx.add(@vectorized, Nx.tensor([2]))
      assert result.shape == {1, 3}

      assert result ==
               Nx.vectorize(
                 Nx.tensor([[[2, 3, 4]], [[5, 6, 7]], [[8, 9, 10]]]),
                 :rows
               )
    end

    test "left addition by rank-2" do
      result = Nx.add(Nx.tensor([[1], [2]]), @vectorized)
      assert result.shape == {2, 3}

      assert result ==
               Nx.vectorize(
                 Nx.tensor([
                   [[1, 2, 3], [2, 3, 4]],
                   [[4, 5, 6], [5, 6, 7]],
                   [[7, 8, 9], [8, 9, 10]]
                 ]),
                 :rows
               )
    end

    test "right addition by rank-2" do
      result = Nx.add(@vectorized, Nx.tensor([[1], [2]]))
      assert result.shape == {2, 3}

      assert result ==
               Nx.vectorize(
                 Nx.tensor([
                   [[1, 2, 3], [2, 3, 4]],
                   [[4, 5, 6], [5, 6, 7]],
                   [[7, 8, 9], [8, 9, 10]]
                 ]),
                 :rows
               )
    end

    test "addition by vectorized with same axes" do
      assert Nx.vectorize(Nx.add(@base, @base), :rows) == Nx.add(@vectorized, @vectorized)
    end

    test "addition by vectorized with different axes" do
      v2 = Nx.vectorize(@base, :cols)

      result =
        Nx.stack([
          Nx.add(@base, Nx.tensor([[0, 1, 2]])),
          Nx.add(@base, Nx.tensor([[3, 4, 5]])),
          Nx.add(@base, Nx.tensor([[6, 7, 8]]))
        ])
        |> Nx.vectorize(:rows)
        |> Nx.vectorize(:cols)

      assert result == Nx.add(@vectorized, v2)
    end

    test "addition by vectorized with common axes" do
      base_2 = Nx.iota({1, 2, 3, 1, 3})
      v2 = base_2 |> Nx.vectorize(:x) |> Nx.vectorize(:y) |> Nx.vectorize(:rows)

      base = Nx.concatenate([@base, @base], axis: 1) |> Nx.reshape({1, 2, 3, 1, 3})

      result =
        Nx.add(base, base_2)
        |> Nx.reshape({3, 1, 2, 1, 3})
        |> Nx.vectorize(:rows)
        |> Nx.vectorize(:x)
        |> Nx.vectorize(:y)

      assert result == Nx.add(@vectorized, v2)
    end
  end

  describe "predicate operations" do
    test "left Nx.equal by scalar" do
      result = Nx.equal(2, @vectorized)
      assert result.shape == {1, 3}

      assert result ==
               Nx.vectorize(
                 Nx.tensor([[[0, 0, 1]], [[0, 0, 0]], [[0, 0, 0]]], type: :u8),
                 :rows
               )
    end

    test "right Nx.equal by scalar" do
      result = Nx.equal(@vectorized, 2)
      assert result.shape == {1, 3}

      assert result ==
               Nx.vectorize(
                 Nx.tensor([[[0, 0, 1]], [[0, 0, 0]], [[0, 0, 0]]], type: :u8),
                 :rows
               )
    end

    test "left Nx.equal by rank-1" do
      result = Nx.equal(Nx.tensor([2]), @vectorized)
      assert result.shape == {1, 3}

      assert result ==
               Nx.vectorize(
                 Nx.tensor([[[0, 0, 1]], [[0, 0, 0]], [[0, 0, 0]]], type: :u8),
                 :rows
               )
    end

    test "right Nx.equal by rank-1" do
      result = Nx.equal(@vectorized, Nx.tensor([2]))
      assert result.shape == {1, 3}

      assert result ==
               Nx.vectorize(
                 Nx.tensor([[[0, 0, 1]], [[0, 0, 0]], [[0, 0, 0]]], type: :u8),
                 :rows
               )
    end

    test "left Nx.equal by rank-2" do
      result = Nx.equal(Nx.tensor([[1], [2]]), @vectorized)
      assert result.shape == {2, 3}

      assert result ==
               Nx.vectorize(
                 Nx.tensor(
                   [
                     [[0, 1, 0], [0, 0, 1]],
                     [[0, 0, 0], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0]]
                   ],
                   type: :u8
                 ),
                 :rows
               )
    end

    test "right Nx.equal by rank-2" do
      result = Nx.equal(@vectorized, Nx.tensor([[1], [2]]))
      assert result.shape == {2, 3}

      assert result ==
               Nx.vectorize(
                 Nx.tensor(
                   [
                     [[0, 1, 0], [0, 0, 1]],
                     [[0, 0, 0], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0]]
                   ],
                   type: :u8
                 ),
                 :rows
               )
    end

    test "Nx.equal by vectorized with same axes" do
      assert Nx.vectorize(Nx.broadcast(Nx.tensor(1, type: :u8), @base), :rows) ==
               Nx.equal(@vectorized, @vectorized)
    end

    test "Nx.equal by vectorized with different axes" do
      v2 = Nx.vectorize(@base, :cols)

      result =
        Nx.stack([
          Nx.equal(@base, Nx.tensor([[0, 1, 2]])),
          Nx.equal(@base, Nx.tensor([[3, 4, 5]])),
          Nx.equal(@base, Nx.tensor([[6, 7, 8]]))
        ])
        |> Nx.vectorize(:rows)
        |> Nx.vectorize(:cols)

      assert result == Nx.equal(@vectorized, v2)
    end

    test "Nx.equal by vectorized with common axes" do
      base_2 = Nx.iota({1, 2, 3, 1, 3})
      v2 = base_2 |> Nx.vectorize(:x) |> Nx.vectorize(:y) |> Nx.vectorize(:rows)

      base = Nx.concatenate([@base, @base], axis: 1) |> Nx.reshape({1, 2, 3, 1, 3})

      result =
        Nx.equal(base, base_2)
        |> Nx.reshape({3, 1, 2, 1, 3})
        |> Nx.vectorize(:rows)
        |> Nx.vectorize(:x)
        |> Nx.vectorize(:y)

      assert result == Nx.equal(@vectorized, v2)
    end
  end

  describe "unary operations" do
    for op <- [
          :logical_not,
          :is_nan,
          :is_infinity,
          :negate
        ] do
      test "#{op}" do
        result =
          @base_unary
          |> Nx.unquote(op)()
          |> Nx.vectorize(:rows)

        assert result == Nx.unquote(op)(@vectorized_unary)
      end
    end

    test "sign" do
      input = Nx.tensor([-10, 0, 10]) |> Nx.vectorize(:rows)
      result = Nx.vectorize(Nx.tensor([-1, 0, 1]), :rows)
      assert result == Nx.sign(input)
    end

    test "abs" do
      input = Nx.tensor([-10, 0, 10]) |> Nx.vectorize(:rows)
      result = Nx.vectorize(Nx.tensor([10, 0, 10]), :rows)
      assert result == Nx.abs(input)
    end

    test "conjugate" do
      input = ~V[-1i 1i 10-i] |> Nx.vectorize(:rows)
      result = Nx.vectorize(~V[1i -1i 10+i], :rows)
      assert result == Nx.conjugate(input)
    end

    test "phase" do
      input = ~V[1i 0 10] |> Nx.vectorize(:rows)
      result = Nx.vectorize(Nx.tensor([:math.pi() / 2, 0, 0]), :rows)
      assert result == Nx.phase(input)
    end

    test "real" do
      input = ~V[-1i 0 10] |> Nx.vectorize(:rows)
      result = Nx.vectorize(~V[0 0 10]f32, :rows)
      assert result == Nx.real(input)
    end

    test "imag" do
      input = ~V[-1i 0 10] |> Nx.vectorize(:rows)
      result = Nx.vectorize(~V[-1 0 0]f32, :rows)
      assert result == Nx.imag(input)
    end

    test "bitwise_not" do
      input = ~V[15 240 0]u8 |> Nx.vectorize(:rows)
      result = Nx.vectorize(~V[240 15 255]u8, :rows)
      assert result == Nx.bitwise_not(input)
    end

    test "population_count" do
      input = ~V[15 240 3]u8 |> Nx.vectorize(:rows)
      result = Nx.vectorize(~V[4 4 2]u8, :rows)
      assert result == Nx.population_count(input)
    end

    test "count_leading_zeros" do
      input = ~V[15 240 3]u8 |> Nx.vectorize(:rows)
      result = Nx.vectorize(~V[4 0 6]u8, :rows)
      assert result == Nx.count_leading_zeros(input)
    end

    test "sort" do
      input = ~M[
        1 2 3
        3 2 1
      ] |> Nx.vectorize(:rows)

      result = ~M[
        1 2 3
        1 2 3
      ] |> Nx.vectorize(:rows)

      assert result == Nx.sort(input, axis: 0)
    end

    test "argsort" do
      input = ~M[
        1 2 3
        3 2 1
      ] |> Nx.vectorize(:rows)

      result = ~M[
        0 1 2
        2 1 0
      ] |> Nx.vectorize(:rows)

      assert result == Nx.argsort(input, axis: 0)
    end

    test "top_k" do
      input =
        Nx.tensor([
          [[1, 2, 3]],
          [[5, 4, 3]]
        ])
        |> Nx.vectorize(:rows)

      result_values =
        Nx.tensor([
          [[3, 2]],
          [[5, 4]]
        ])
        |> Nx.vectorize(:rows)

      result_idx =
        Nx.tensor([
          [[2, 1]],
          [[0, 1]]
        ])
        |> Nx.vectorize(:rows)

      assert {result_values, result_idx} == Nx.top_k(input, k: 2)
    end

    test "reflect" do
      input = ~M[
        0 1 2
        5 4 3
      ] |> Nx.vectorize(:rows)

      result = ~M[
        1 2 1 0 1 2 1
        4 3 4 5 4 3 4
      ] |> Nx.vectorize(:rows)

      assert result == Nx.reflect(input, padding_config: [{3, 1}])
    end
  end

  describe "while/3" do
    defn double_n_times(x, n) do
      {_i, v} =
        while {i = n, v = x}, i > 0 do
          {i - 1, v * 2}
        end

      v
    end

    defn double_x_triple_y_n_times(x, y, n) do
      {_i, v, z} =
        while {i = n, v = x, z = y}, i > 0 do
          {i - 1, v * 2, z * 3}
        end

      {v, z}
    end

    defn y_plus_each_x(x, y) do
      while y, t <- x do
        y + t
      end
    end

    test "simple" do
      assert double_n_times(Nx.tensor(3), Nx.tensor(5)) == Nx.tensor(96)

      x = Nx.vectorize(~V[1 2 3], :x)
      n = Nx.vectorize(~V[5 6 3], :x)

      assert double_n_times(x, n) == Nx.vectorize(~V[32 128 24], :x)
    end

    test "different axes" do
      x = Nx.vectorize(~V[1 2 3], :init)
      n = Nx.vectorize(~V[4 5], :pred)

      assert double_n_times(x, n) ==
               Nx.vectorize(
                 ~M[
                  16 32 48
                  32 64 96
                 ],
                 pred: 2,
                 init: 3
               )
    end

    test "mix of common and different axes" do
      x =
        Nx.vectorize(
          ~M[
          1 2
          3 4
          5 6
        ],
          other: 3,
          pred: 2
        )

      y = Nx.vectorize(~V[1 2], :pred)
      n = Nx.vectorize(~V[3 4], :pred)

      assert double_x_triple_y_n_times(x, y, n) == {
               Nx.vectorize(
                 ~M[
                   8 16 24
                   64 80 96
                 ],
                 pred: 2,
                 other: 3
               ),
               Nx.vectorize(~V[27 162], pred: 2)
             }
    end

    test "simple with multiple pred axes" do
      x = Nx.vectorize(~V[1 2 3], :x)
      n = Nx.vectorize(~M[
        0 1 2
        5 6 3
      ], y: 2, x: 3)

      assert double_n_times(x, n) ==
               Nx.vectorize(
                 ~M[
                  1 4 12
                  32 128 24
                ],
                 y: 2,
                 x: 3
               )
    end

    test "vectorized generator" do
      x_devec = Nx.iota({3, 4})
      x = Nx.vectorize(x_devec, x: 3)
      y = Nx.tensor([[10, 20, 30, 40], [-10, -20, -30, -40]]) |> Nx.vectorize(y: 2)

      res1 = y_plus_each_x(x_devec[0], y)
      res2 = y_plus_each_x(x_devec[1], y)
      res3 = y_plus_each_x(x_devec[2], y)

      result = y_plus_each_x(x, y)

      assert result ==
               {res1, res2, res3}
               |> Nx.devectorize()
               |> Nx.stack(axis: 0)
               |> Nx.vectorize(x: 3, y: 2)
    end
  end

  describe "cond" do
    defn vectorized_if(pred, then, other) do
      cond do
        pred -> print_value(then, label: "if")
        true -> print_value(other, label: "else")
      end
    end

    defn vectorized_cond(pred1, clause1, pred2, clause2, clause3) do
      cond do
        pred1 -> print_value(clause1, label: "clause_1")
        pred2 -> print_value(clause2, label: "clause_2")
        true -> print_value(clause3, label: "clause_3")
      end
    end

    test "simple if" do
      # this tests the case where we have a single vectorized predicate
      pred = Nx.vectorize(~V[0 1 0], :pred)

      io =
        ExUnit.CaptureIO.capture_io(fn ->
          assert vectorized_if(pred, 1, 2) == Nx.vectorize(~V[2 1 2], :pred)
        end)

      assert io ==
               IO.iodata_to_binary([
                 "if: ",
                 inspect(Nx.tensor(1)),
                 "\n",
                 "else: ",
                 inspect(Nx.tensor(2)),
                 "\n"
               ])
    end

    test "simple cond" do
      # this tests the case where we have a two vectorized predicates
      pred1 = Nx.vectorize(~V[1 0 0], :pred)
      pred2 = Nx.vectorize(~V[0 0 0], :pred)

      io =
        ExUnit.CaptureIO.capture_io(fn ->
          assert vectorized_cond(pred1, 1, pred2, 2, 3) == Nx.vectorize(~V[1 3 3], :pred)
        end)

      # This assertion ensures that the clause for pred2 is never executed
      assert io ==
               IO.iodata_to_binary([
                 "clause_1: #{inspect(Nx.tensor(1))}\n",
                 "clause_3: #{inspect(Nx.tensor(3))}\n"
               ])
    end

    test "if with container result" do
      pred1 = Nx.vectorize(~V[2 0 0], :pred)

      io =
        ExUnit.CaptureIO.capture_io(fn ->
          result =
            vectorized_if(
              pred1,
              {1, 2, 3},
              {7, 8, Nx.vectorize(~V[9 10 11], :x)}
            )

          assert result == {
                   Nx.vectorize(~V[1 7 7], :pred),
                   Nx.vectorize(~V[2 8 8], :pred),
                   Nx.vectorize(~M[
                  3 3 3
                  9 10 11
                  9 10 11
                ], pred: 3, x: 3)
                 }
        end)

      # This assertion ensures that the clause for pred2 is never executed
      assert String.replace(io, ",\n", ",") ==
               IO.iodata_to_binary([
                 "if: ",
                 inspect({Nx.tensor(1), Nx.tensor(2), Nx.tensor(3)}),
                 "\n",
                 "else: ",
                 inspect({Nx.tensor(7), Nx.tensor(8), Nx.tensor([9, 10, 11], names: [:x])}),
                 "\n"
               ])
    end

    defn cond4(p1, c1, p2, c2, p3, c3, c4) do
      cond do
        p1 -> send_value(c1, clause: "c1")
        p2 -> send_value(c2, clause: "c2")
        p3 -> send_value(c3, clause: "c3")
        true -> send_value(c4, clause: "c4")
      end
    end

    deftransformp send_value(val, opts \\ []) do
      Nx.Defn.Kernel.hook(val, &send(self(), {&1, opts}))
    end

    test "only executes selected branches" do
      t = Nx.vectorize(~V[1], :pred)
      f = Nx.vectorize(~V[0], :pred)

      assert = fn res, val, clause ->
        t = Nx.tensor(val)
        assert Nx.vectorize(Nx.new_axis(t, 0), :pred) == res
        assert_received {^t, clause: ^clause}
        refute_received _
      end

      assert.(cond4(t, 10, 0, 20, 0, 30, 40), 10, "c1")
      assert.(cond4(0, 10, t, 20, 0, 30, 40), 20, "c2")
      assert.(cond4(0, 10, 0, 20, t, 30, 40), 30, "c3")
      assert.(cond4(f, 10, 0, 20, 0, 30, 40), 40, "c4")
    end

    test "1 vectorized pred in the beginning" do
      assert cond4(Nx.vectorize(~V[0 1], :pred), 10, 0, 20, 0, 30, 40) ==
               Nx.vectorize(~V[40 10], :pred)

      assert cond4(Nx.vectorize(~V[0 0], :pred), 10, 1, 20, 0, 30, 40) ==
               Nx.vectorize(~V[20 20], :pred)

      assert cond4(Nx.vectorize(~V[0 0], :pred), 10, 0, 20, 1, 30, 40) ==
               Nx.vectorize(~V[30 30], :pred)

      assert cond4(Nx.vectorize(~V[0 0], :pred), 10, 0, 20, 0, 30, 40) ==
               Nx.vectorize(~V[40 40], :pred)
    end

    test "1 vectorized pred in the second but not last position" do
      assert cond4(0, 10, Nx.vectorize(~V[0 1], :pred), 20, 0, 30, 40) ==
               Nx.vectorize(~V[40 20], :pred)

      assert cond4(1, 10, Nx.vectorize(~V[0 1], :pred), 20, 0, 30, 40) ==
               Nx.vectorize(~V[10 10], :pred)

      assert cond4(0, 10, Nx.vectorize(~V[0 0], :pred), 20, 1, 30, 40) ==
               Nx.vectorize(~V[30 30], :pred)

      assert cond4(0, 10, Nx.vectorize(~V[0 0], :pred), 20, 0, 30, 40) ==
               Nx.vectorize(~V[40 40], :pred)
    end

    test "1 vectorized pred in the last position" do
      assert cond4(0, 10, 0, 20, Nx.vectorize(~V[0 1], :pred), 30, 40) ==
               Nx.vectorize(~V[40 30], :pred)

      assert cond4(1, 10, 0, 20, Nx.vectorize(~V[0 1], :pred), 30, 40) ==
               Nx.vectorize(~V[10 10], :pred)

      assert cond4(0, 10, 1, 20, Nx.vectorize(~V[0 1], :pred), 30, 40) ==
               Nx.vectorize(~V[20 20], :pred)

      assert cond4(0, 10, 0, 20, Nx.vectorize(~V[0 0], :pred), 30, 40) ==
               Nx.vectorize(~V[40 40], :pred)
    end

    test "2 vectorized preds with different axes" do
      assert cond4(
               Nx.vectorize(~V[0 1 0], :pred1),
               10,
               Nx.vectorize(~V[1 0], :pred2),
               20,
               0,
               30,
               40
             ) ==
               Nx.vectorize(~M[
              20 40
              10 10
              20 40
            ], pred1: 3, pred2: 2)
    end

    test "2 vectorized preds with different axes + clauses that match either" do
      assert cond4(
               Nx.vectorize(~V[0 1 0], :pred1),
               Nx.vectorize(~V[10 100], :pred2),
               Nx.vectorize(~V[1 0], :pred2),
               Nx.vectorize(~V[20 200 2000], :pred1),
               0,
               30,
               40
             ) ==
               Nx.vectorize(~M[
              20 40
              10 100
              2000 40
            ], pred1: 3, pred2: 2)
    end
  end

  describe "access" do
    test "get vectorized tensor, unvectorized indices" do
      t = Nx.iota({3}, vectorized_axes: [x: 2])

      assert t[0] == Nx.vectorize(Nx.tensor([0, 0]), x: 2)
      assert t[1] == Nx.vectorize(Nx.tensor([1, 1]), x: 2)
      assert t[2] == Nx.vectorize(Nx.tensor([2, 2]), x: 2)

      t = Nx.iota({3, 4}, vectorized_axes: [x: 2])

      assert t[[0, 1..2]] == Nx.vectorize(Nx.tensor([[1, 2], [1, 2]]), x: 2)
      assert t[[1, 1..1]] == Nx.vectorize(Nx.tensor([[5], [5]]), x: 2)
      assert t[[2, 2..3]] == Nx.vectorize(Nx.tensor([[10, 11], [10, 11]]), x: 2)
    end

    test "get unvectorized tensor, vectorized indices" do
      t = Nx.iota({6})
      i = Nx.tensor([0, 2, 4]) |> Nx.vectorize(:i)

      assert t[i] == Nx.vectorize(Nx.tensor([0, 2, 4]), i: 3)

      t = Nx.iota({6, 2})

      i = Nx.tensor([0, 2, 4]) |> Nx.vectorize(:i)

      assert t[[i, 0..1]] ==
               Nx.vectorize(
                 Nx.tensor([
                   [0, 1],
                   [4, 5],
                   [8, 9]
                 ]),
                 i: 3
               )
    end

    test "get vectorized tensor, vectorized indices" do
      t = Nx.iota({3, 6}) |> Nx.vectorize(i: 3)
      i = Nx.tensor([0, 2, 4]) |> Nx.vectorize(i: 3)

      assert t[i] == Nx.vectorize(Nx.tensor([0, 8, 16]), i: 3)

      t = Nx.iota({2, 6, 2}) |> Nx.vectorize(j: 2)

      assert t[[i, 0..1]] ==
               Nx.vectorize(
                 Nx.tensor([
                   [[0, 1], [4, 5], [8, 9]],
                   [[12, 13], [16, 17], [20, 21]]
                 ]),
                 j: 2,
                 i: 3
               )
    end
  end
end
