defmodule Torchx.DonationTest do
  use Torchx.Case, async: true

  import Nx.Defn

  defp donated(list, opts \\ []), do: list |> Nx.tensor(opts) |> Nx.donatable()

  defp assert_consumed(tensor) do
    assert_raise RuntimeError, ~r"deallocated", fn -> Nx.to_flat_list(tensor) end
  end

  describe "elementwise ops" do
    test "a donated tensor is consumed by a unary op" do
      x = donated([1.0, 2.0, 3.0])
      result = Nx.exp(x)

      assert_all_close(result, Nx.exp(Nx.tensor([1.0, 2.0, 3.0])))
      refute Nx.donatable?(result)
      assert_consumed(x)
    end

    test "a tensor that is not donated is left alone" do
      x = Nx.tensor([1.0, 2.0, 3.0])

      assert_all_close(Nx.exp(x), Nx.exp(Nx.tensor([1.0, 2.0, 3.0])))
      assert_all_close(Nx.add(x, x), Nx.tensor([2.0, 4.0, 6.0]))
      assert_equal(x, Nx.tensor([1.0, 2.0, 3.0]))
    end

    test "a donated left operand is consumed by a binary op" do
      x = donated([1.0, 2.0, 3.0])
      y = Nx.tensor([10.0, 20.0, 30.0])

      assert_equal(Nx.add(x, y), Nx.tensor([11.0, 22.0, 33.0]))
      assert_consumed(x)
      assert_equal(y, Nx.tensor([10.0, 20.0, 30.0]))
    end

    test "a donated operand that needs broadcasting gives the right result" do
      x = donated([1.0, 2.0])
      y = Nx.tensor([[10.0, 20.0], [30.0, 40.0]])

      assert_equal(Nx.add(x, y), Nx.tensor([[11.0, 22.0], [31.0, 42.0]]))
    end

    test "ops that change the type give the right result" do
      x = donated([1, 2, 3], type: :s32)
      assert_all_close(Nx.divide(x, 2), Nx.tensor([0.5, 1.0, 1.5]))

      x = donated([1, 2, 3], type: :s32)
      assert_equal(Nx.add(x, 1.5), Nx.tensor([2.5, 3.5, 4.5]))

      x = donated([1, 2, 3], type: :u8)
      assert_equal(Nx.multiply(x, 100), Nx.tensor([100, 200, 44], type: :u8))

      x = donated([1, 2, 3], type: :s32)
      assert_equal(Nx.greater(x, 1), Nx.tensor([0, 1, 1], type: :u8))
    end

    test "a donated operand that shares its buffer with a view is not overwritten" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      view = x |> Nx.reshape({4}) |> Nx.donatable()

      assert_all_close(Nx.exp(view), Nx.exp(Nx.tensor([1.0, 2.0, 3.0, 4.0])))
      assert_equal(x, Nx.tensor([[1.0, 2.0], [3.0, 4.0]]))
    end

    test "a donated broadcast result gives the right result" do
      x = donated([1.0, 2.0])
      broadcast = x |> Nx.broadcast({3, 2}) |> Nx.donatable()

      assert_equal(Nx.add(broadcast, 1.0), Nx.tensor([[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]]))
    end

    test "a donated complex tensor can be conjugated" do
      x = Nx.tensor([Complex.new(1, 2), Complex.new(3, -4)]) |> Nx.donatable()

      assert_equal(Nx.conjugate(x), Nx.tensor([Complex.new(1, -2), Complex.new(3, 4)]))
    end

    test "real and imag of a donated complex tensor" do
      complex = [Complex.new(1, 2), Complex.new(3, -4)]

      assert_equal(Nx.real(Nx.donatable(Nx.tensor(complex))), Nx.tensor([1.0, 3.0]))
      assert_equal(Nx.imag(Nx.donatable(Nx.tensor(complex))), Nx.tensor([2.0, -4.0]))

      x = Nx.tensor(complex)
      assert_equal(Nx.real(x), Nx.tensor([1.0, 3.0]))
      assert_equal(x, Nx.tensor(complex))
    end
  end

  describe "indexing ops" do
    test "put_slice writes into a donated input" do
      x = donated([1.0, 2.0, 3.0])

      assert_equal(Nx.put_slice(x, [1], Nx.tensor([9.0])), Nx.tensor([1.0, 9.0, 3.0]))
      assert_consumed(x)
    end

    test "put_slice copies an input that is not donated" do
      x = Nx.tensor([1.0, 2.0, 3.0])

      assert_equal(Nx.put_slice(x, [1], Nx.tensor([9.0])), Nx.tensor([1.0, 9.0, 3.0]))
      assert_equal(x, Nx.tensor([1.0, 2.0, 3.0]))
    end

    test "put_slice with a changed type does not consume the input" do
      x = Nx.tensor([1, 2, 3], type: :s32)
      result = Nx.put_slice(x, [1], Nx.tensor([9.5]))

      assert_equal(result, Nx.tensor([1.0, 9.5, 3.0]))
      assert_equal(x, Nx.tensor([1, 2, 3], type: :s32))
    end

    test "indexed_put and indexed_add on a donated input" do
      indices = Nx.tensor([[0], [2]])
      updates = Nx.tensor([10.0, 30.0])

      x = donated([1.0, 2.0, 3.0])
      assert_equal(Nx.indexed_put(x, indices, updates), Nx.tensor([10.0, 2.0, 30.0]))
      assert_consumed(x)

      x = donated([1.0, 2.0, 3.0])
      assert_equal(Nx.indexed_add(x, indices, updates), Nx.tensor([11.0, 2.0, 33.0]))

      x = Nx.tensor([1.0, 2.0, 3.0])
      assert_equal(Nx.indexed_add(x, indices, updates), Nx.tensor([11.0, 2.0, 33.0]))
      assert_equal(x, Nx.tensor([1.0, 2.0, 3.0]))
    end

    test "indexed_put on a donated input along other axes" do
      x = donated([[1.0, 2.0], [3.0, 4.0]])
      indices = Nx.tensor([[0], [1]])
      updates = Nx.tensor([[9.0, 9.0], [8.0, 8.0]])

      result = Nx.indexed_put(x, indices, updates, axes: [1])
      assert_equal(result, Nx.tensor([[9.0, 8.0], [9.0, 8.0]]))
    end

    test "slice of a donated tensor" do
      x = donated([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      slice = Nx.slice(x, [0, 1], [2, 2])

      assert_equal(slice, Nx.tensor([[2.0, 3.0], [5.0, 6.0]]))
      assert_consumed(x)
      assert_equal(Nx.add(Nx.donatable(slice), 1.0), Nx.tensor([[3.0, 4.0], [6.0, 7.0]]))
    end

    test "slice of a tensor that is not donated is a copy" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      slice = Nx.slice(x, [0, 1], [2, 2])

      assert_equal(Nx.exp(Nx.donatable(slice)), Nx.exp(Nx.tensor([[2.0, 3.0], [5.0, 6.0]])))
      assert_equal(x, Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    end
  end

  describe "view ops" do
    test "reshape and transpose of a donated tensor hand over the buffer" do
      x = donated([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      reshaped = Nx.reshape(x, {3, 2})

      assert_consumed(x)
      assert_equal(reshaped, Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))

      transposed = Nx.transpose(Nx.donatable(reshaped))
      assert_consumed(reshaped)

      assert_equal(
        Nx.add(Nx.donatable(transposed), 1.0),
        Nx.tensor([[2.0, 4.0, 6.0], [3.0, 5.0, 7.0]])
      )
    end

    test "reshape of a tensor that is not donated keeps it valid" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      reshaped = Nx.reshape(x, {4})

      assert_equal(Nx.exp(Nx.donatable(reshaped)), Nx.exp(Nx.tensor([1.0, 2.0, 3.0, 4.0])))
      assert_equal(x, Nx.tensor([[1.0, 2.0], [3.0, 4.0]]))
    end

    test "squeeze and as_type of a donated tensor" do
      x = donated([[1.0], [2.0]])
      assert_equal(Nx.squeeze(x, axes: [1]), Nx.tensor([1.0, 2.0]))

      x = donated([1.5, 2.5])
      assert_equal(Nx.as_type(x, :s32), Nx.tensor([1, 2], type: :s32))
    end

    test "bitcast between s64 and u64 does not share a handle with the input" do
      x = Nx.tensor([1, 2, 3], type: :s64)
      y = Nx.bitcast(x, :u64)

      assert_equal(Nx.add(Nx.donatable(y), 1), Nx.tensor([2, 3, 4], type: :u64))
      assert_equal(x, Nx.tensor([1, 2, 3], type: :s64))
    end

    test "eye and iota with batch axes can be written to" do
      eye = Nx.eye({2, 2, 2})

      assert_equal(
        Nx.put_slice(eye, [0, 0, 0], Nx.tensor([[[5.0]]])),
        Nx.broadcast(Nx.tensor([[1.0, 0.0], [0.0, 1.0]]), {2, 2, 2})
        |> Nx.put_slice([0, 0, 0], Nx.tensor([[[5.0]]]))
      )

      iota = Nx.iota({2, 3}, axis: 1)
      assert_equal(Nx.add(Nx.donatable(iota), 1), Nx.tensor([[1, 2, 3], [1, 2, 3]]))
    end
  end

  describe "under Nx.Defn.Evaluator" do
    defn shared(x) do
      y = Nx.add(x, 1)
      Nx.add(Nx.exp(y), Nx.negate(y))
    end

    defn held(x), do: Nx.add(x, Nx.negate(x))

    defn square(x) do
      y = Nx.add(x, 1)
      Nx.multiply(y, y)
    end

    defn with_cond(pred, x) do
      y = Nx.add(x, 1)

      z =
        if pred do
          y
        else
          Nx.exp(y)
        end

      Nx.add(z, Nx.negate(y))
    end

    defn loop(x) do
      {_, acc} =
        while {i = 0, acc = x}, i < 3 do
          {i + 1, acc * 2.0 + 1.0}
        end

      acc
    end

    defn hooked(x) do
      y = print_value(Nx.add(x, 1))
      Nx.add(Nx.exp(y), Nx.negate(y))
    end

    defp expected(fun, args) do
      args = Enum.map(args, &Nx.backend_copy(&1, Nx.BinaryBackend))
      Nx.with_default_backend(Nx.BinaryBackend, fn -> apply(fun, args) end)
    end

    defp run(fun, args) do
      expected = expected(fun, args)

      donated_args =
        Enum.map(args, fn arg ->
          arg
          |> Nx.backend_copy(Nx.BinaryBackend)
          |> Nx.backend_transfer(Torchx.Backend)
          |> Nx.donatable()
        end)

      result = apply(Nx.Defn.jit(fun), donated_args)
      assert_all_close(result, expected)
      result
    end

    test "intermediates used once and donated arguments" do
      run(&once/1, [Nx.tensor([1.0, 2.0, 3.0])])
      run(&shared/1, [Nx.tensor([1.0, 2.0, 3.0])])
      run(&held/1, [Nx.tensor([1.0, 2.0, 3.0])])
      run(&square/1, [Nx.tensor([1.0, 2.0, 3.0])])
    end

    defn once(x), do: Nx.exp(Nx.add(x, 1.0))

    test "values returned by a cond branch are read after the cond" do
      for pred <- [0, 1] do
        run(&with_cond/2, [Nx.tensor(pred), Nx.tensor([1.0, 2.0, 3.0])])
      end
    end

    test "while loops" do
      run(&loop/1, [Nx.tensor([1.0, 2.0])])
    end

    test "hooks see the values that were computed" do
      ExUnit.CaptureIO.capture_io(fn ->
        run(&hooked/1, [Nx.tensor([1.0, 2.0])])
      end)
    end

    test "a donated argument is consumed at its last use" do
      x = Nx.tensor([1.0, 2.0, 3.0]) |> Nx.donatable()

      assert_all_close(Nx.Defn.jit(&once/1).(x), Nx.exp(Nx.tensor([2.0, 3.0, 4.0])))
      assert_consumed(x)
    end

    test "a donated argument that is read again is not consumed early" do
      x = Nx.tensor([1.0, 2.0, 3.0]) |> Nx.donatable()

      assert_all_close(Nx.Defn.jit(&held/1).(x), Nx.tensor([0.0, 0.0, 0.0]))
    end

    test "the caller's tensor is consumed only when it was donated" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      _ = Nx.Defn.jit(&once/1).(x)
      assert_equal(x, Nx.tensor([1.0, 2.0, 3.0]))
    end
  end
end
