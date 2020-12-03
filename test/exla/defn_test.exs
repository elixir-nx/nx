defmodule Exla.DefnTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  @default_defn_compiler Exla

  describe "scalar" do
    defn just_two_int, do: 2
    defn just_two_float, do: 2.0

    test "returns the tensor for the scalar" do
      t = just_two_int()
      assert Nx.to_bitstring(t) == <<2::64-native>>
      assert Nx.type(t) == {:s, 64}
      assert Nx.shape(t) == {}

      t = just_two_float()
      assert Nx.to_bitstring(t) == <<2.0::float-64-native>>
      assert Nx.type(t) == {:f, 64}
      assert Nx.shape(t) == {}
    end
  end

  describe "tensor constants" do
    @two 2
    defn add_two_attribute(t), do: t + @two

    @two_per_two Nx.tensor([[1, 2], [3, 4]])
    defn add_2x2_attribute(t), do: t + @two_per_two

    test "expands module attributes to scalars" do
      assert add_two_attribute(1) == Nx.tensor(3)
      assert add_two_attribute(Nx.tensor([1, 2, 3])) == Nx.tensor([3, 4, 5])
    end

    test "expands module attributes to tensors" do
      assert add_2x2_attribute(1) == Nx.tensor([[2, 3], [4, 5]])
      assert add_2x2_attribute(Nx.tensor([1, 2])) == Nx.tensor([[2, 4], [4, 6]])
    end
  end

  describe "+/2" do
    defn add_two(a, b), do: a + b
    @defn_compiler Nx.Defn
    defn add_two_nx(a, b), do: a + b

    test "same shape and type" do
      tensor = add_two(1.0, 2.0)
      assert Nx.to_bitstring(tensor) == <<3.0::float-64-native>>
      assert Nx.type(tensor) == {:f, 64}
      assert Nx.shape(tensor) == {}

      tensor = add_two(1, 2)
      assert Nx.to_bitstring(tensor) == <<3::64-native>>
      assert Nx.type(tensor) == {:s, 64}
      assert Nx.shape(tensor) == {}

      tensor = add_two(Nx.tensor([1, 2]), Nx.tensor([3, 4]))
      assert Nx.to_bitstring(tensor) == <<4::64-native, 6::64-native>>
      assert Nx.type(tensor) == {:s, 64}
      assert Nx.shape(tensor) == {2}

      tensor = add_two(Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0]))
      assert Nx.to_bitstring(tensor) == <<4.0::float-64-native, 6.0::float-64-native>>
      assert Nx.type(tensor) == {:f, 64}
      assert Nx.shape(tensor) == {2}
    end

    test "different types" do
      tensors = [
        {1, 2},
        {1.0, 2},
        {1.0, 3.0},
        {Nx.tensor([1, 2], type: {:u, 8}), 3},
        {Nx.tensor([1, 2], type: {:u, 8}), -3},
        {Nx.tensor([1, 2], type: {:u, 8}), 3.0},
        {Nx.tensor([1, 2], type: {:s, 8}), 3},
        {Nx.tensor([1, 2], type: {:s, 8}), 3.0},
        {Nx.tensor([1, 2], type: {:f, 32}), 3},
        {Nx.tensor([1, 2], type: {:f, 32}), 3.0},
        {Nx.tensor([1, 2], type: {:u, 8}), Nx.tensor(3, type: {:u, 16})},
        {Nx.tensor([1, 2], type: {:u, 8}), Nx.tensor(-3, type: {:s, 16})},
        {Nx.tensor([1, 2], type: {:u, 8}), Nx.tensor(3.0, type: {:f, 32})},
        {Nx.tensor([1, 2], type: {:s, 8}), Nx.tensor(3, type: {:s, 16})},
        {Nx.tensor([1, 2], type: {:s, 8}), Nx.tensor(3.0, type: {:f, 32})},
        {Nx.tensor([1, 2], type: {:f, 32}), Nx.tensor(3, type: {:u, 16})},
        {Nx.tensor([1, 2], type: {:f, 32}), Nx.tensor(3, type: {:s, 16})},
        {Nx.tensor([1, 2], type: {:f, 32}), Nx.tensor(3.0, type: {:f, 64})}
      ]

      for {left, right} <- tensors do
        assert add_two(left, right) == add_two_nx(left, right)
        assert add_two(right, left) == add_two_nx(right, left)
      end
    end

    defn add_two_int_int, do: 1 + 2
    @defn_compiler Nx.Defn
    defn add_two_int_int_nx, do: 1 + 2

    defn add_two_int_float, do: 1 + 2.0
    @defn_compiler Nx.Defn
    defn add_two_int_float_nx, do: 1 + 2.0

    defn add_two_float_int, do: 1.0 + 2
    @defn_compiler Nx.Defn
    defn add_two_float_int_nx, do: 1.0 + 2

    defn add_two_float_float, do: 1.0 + 2
    @defn_compiler Nx.Defn
    defn add_two_float_float_nx, do: 1.0 + 2

    defn add_two_int(t), do: t + 2
    @defn_compiler Nx.Defn
    defn add_two_int_nx(t), do: t + 2

    defn add_two_float(t), do: t + 2.0
    @defn_compiler Nx.Defn
    defn add_two_float_nx(t), do: t + 2.0

    test "constants" do
      assert add_two_int_int() == add_two_int_int_nx()
      assert add_two_int_float() == add_two_int_float_nx()
      assert add_two_float_int() == add_two_float_int_nx()
      assert add_two_float_float() == add_two_float_float_nx()

      tensors = [
        Nx.tensor([1, 2], type: {:u, 8}),
        Nx.tensor([1, 2], type: {:u, 8}),
        Nx.tensor([1, 2], type: {:u, 8}),
        Nx.tensor([1, 2], type: {:s, 8}),
        Nx.tensor([1, 2], type: {:s, 8}),
        Nx.tensor([1, 2], type: {:f, 32}),
        Nx.tensor([1, 2], type: {:f, 32})
      ]

      for t <- tensors do
        assert add_two_int(t) == add_two_int_nx(t)
        assert add_two_float(t) == add_two_float_nx(t)
      end
    end

    test "broadcast" do
      tensors = [
        {Nx.tensor([1, 2]), Nx.tensor([[1, 2], [3, 4]])},
        {Nx.tensor([1, 2]), Nx.tensor([[[1, 2], [3, 4]], [[4, 5], [6, 7]]])},
        {Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]])},
        {Nx.tensor([[10, 20]]), Nx.tensor([[1], [2]])},
        {Nx.tensor([[[10], [20]]]), Nx.tensor([[[1, 2]], [[3, 4]]])},
        {Nx.tensor([[[100], [200], [300]]]),
         Nx.tensor([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]])},
        {Nx.tensor([[[[1]]]]), Nx.tensor([[1, 2], [3, 4]])},
        {Nx.tensor([[[[1]]]]), Nx.tensor([1, 2])},
        {Nx.tensor([[[10], [20]], [[30], [40]]]), Nx.tensor([[1, 2]])},
        {Nx.tensor([[[[10], [20]], [[30], [40]]]]), Nx.tensor([[[1, 2]], [[3, 4]]])},
        {Nx.tensor([[[[10], [20]], [[30], [40]]]]), Nx.tensor([[[[1, 2]]], [[[3, 4]]]])},
        {Nx.tensor([[[10], [20]], [[30], [40]]]), Nx.tensor([[[1, 2]], [[3, 4]]])}
      ]

      for {left, right} <- tensors do
        assert add_two(left, right) == add_two_nx(left, right)
        assert add_two(right, left) == add_two_nx(right, left)
      end
    end

    test "broadcast error" do
      assert_raise RuntimeError, ~r"Binary op add with incompatible shapes", fn ->
        add_two(Nx.tensor([1, 2]), Nx.tensor([1, 2, 3]))
      end
    end
  end

  describe "//2" do
    defn divide_two(a, b), do: a / b
    @defn_compiler Nx.Defn
    defn divide_two_nx(a, b), do: a / b

    test "parameters" do
      tensors = [
        {1, 2},
        {1, Nx.tensor([1.0, 2.0, 3.0])},
        {Nx.tensor([1, 2, 3]), 1},
        {Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]])},
        {Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8})},
        {Nx.tensor([[1], [2]], type: {:f, 32}), Nx.tensor([[10, 20]], type: {:f, 32})}
      ]

      for {left, right} <- tensors do
        assert divide_two(left, right) == divide_two_nx(left, right)
        assert divide_two(right, left) == divide_two_nx(right, left)
      end
    end

    defn divide_two_int(t), do: t / 2
    @defn_compiler Nx.Defn
    defn divide_two_int_nx(t), do: t / 2

    defn divide_two_float(t), do: t / 2.0
    @defn_compiler Nx.Defn
    defn divide_two_float_nx(t), do: t / 2.0

    test "constants" do
      tensors = [
        Nx.tensor([1, 2], type: {:u, 8}),
        Nx.tensor([1, 2], type: {:u, 8}),
        Nx.tensor([1, 2], type: {:u, 8}),
        Nx.tensor([1, 2], type: {:s, 8}),
        Nx.tensor([1, 2], type: {:s, 8}),
        Nx.tensor([1, 2], type: {:f, 32}),
        Nx.tensor([1, 2], type: {:f, 32})
      ]

      for t <- tensors do
        assert divide_two_int(t) == divide_two_int_nx(t)
        assert divide_two_float(t) == divide_two_float_nx(t)
      end
    end
  end

  describe "remainder" do
    defn remainder(a, b), do: Nx.remainder(a, b)
    @defn_compiler Nx.Defn
    defn remainder_nx(a, b), do: Nx.remainder(a, b)

    test "integers" do
      left = Nx.tensor([-1023, 1023])
      right = Nx.tensor([[-4], [4]])
      assert Nx.shape(remainder(left, right)) == {2, 2}
      assert remainder(left, right) == remainder_nx(left, right)
    end

    test "floats" do
      left = Nx.tensor([-8.3, -8.4, -8.5, 8.3, 8.4, 8.5])
      right = Nx.tensor([[-4.2], [-4.1], [-4.0], [4.0], [4.1], [4.2]])
      assert Nx.shape(remainder(left, right)) == {6, 6}
      assert remainder(left, right) == remainder_nx(left, right)
    end
  end

  describe "element-wise arith operators" do
    @tensors [
      {1, 2},
      {1, Nx.tensor([1.0, 2.0, 3.0])},
      {Nx.tensor([1, 2, 3]), 1},
      {Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]])},
      {Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8})},
      {Nx.tensor([[1], [2]], type: {:f, 32}), Nx.tensor([[10, 20]], type: {:f, 32})}
    ]

    defn subtract_two(a, b), do: a - b
    @defn_compiler Nx.Defn
    defn subtract_two_nx(a, b), do: a - b

    test "-" do
      for {left, right} <- @tensors do
        assert subtract_two(left, right) == subtract_two_nx(left, right)
        assert subtract_two(right, left) == subtract_two_nx(right, left)
      end
    end

    defn multiply_two(a, b), do: a * b
    @defn_compiler Nx.Defn
    defn multiply_two_nx(a, b), do: a * b

    test "*" do
      for {left, right} <- @tensors do
        assert multiply_two(left, right) == multiply_two_nx(left, right)
        assert multiply_two(right, left) == multiply_two_nx(right, left)
      end
    end

    defn max_two(a, b), do: max(a, b)
    @defn_compiler Nx.Defn
    defn max_two_nx(a, b), do: max(a, b)

    test "max" do
      for {left, right} <- @tensors do
        assert max_two(left, right) == max_two_nx(left, right)
        assert max_two(right, left) == max_two_nx(right, left)
      end
    end

    defn min_two(a, b), do: min(a, b)
    @defn_compiler Nx.Defn
    defn min_two_nx(a, b), do: min(a, b)

    test "min" do
      for {left, right} <- @tensors do
        assert min_two(left, right) == min_two_nx(left, right)
        assert min_two(right, left) == min_two_nx(right, left)
      end
    end

    defn arctan2_two(a, b), do: Nx.arctan2(a, b)
    @defn_compiler Nx.Defn
    defn arctan2_two_nx(a, b), do: Nx.arctan2(a, b)

    test "arctan2" do
      <<neg_zero::float>> = <<0x8000000000000000::64>>
      left = Nx.tensor([-1.0, neg_zero, 0.0, 1.0])
      right = Nx.tensor([[-1.0], [neg_zero], [0.0], [1.0]])

      assert arctan2_two(left, right) == arctan2_two_nx(left, right)
      assert arctan2_two(right, left) == arctan2_two_nx(right, left)
    end
  end

  describe "element-wise bitwise operators" do
    @left Nx.tensor([-2, -1, 0, 1, 2])
    @right Nx.tensor([[-2], [-1], [0], [1], [2]])

    defn bitwise_and(a, b), do: a &&& b
    @defn_compiler Nx.Defn
    defn bitwise_and_nx(a, b), do: a &&& b

    test "bitwise_and" do
      assert Nx.shape(bitwise_and(@left, @right)) == {5, 5}
      assert bitwise_and(@left, @right) == bitwise_and_nx(@left, @right)
    end

    defn bitwise_or(a, b), do: a ||| b
    @defn_compiler Nx.Defn
    defn bitwise_or_nx(a, b), do: a ||| b

    test "bitwise_or" do
      assert Nx.shape(bitwise_or(@left, @right)) == {5, 5}
      assert bitwise_or(@left, @right) == bitwise_or_nx(@left, @right)
    end

    defn bitwise_xor(a, b), do: a ^^^ b
    @defn_compiler Nx.Defn
    defn bitwise_xor_nx(a, b), do: a ^^^ b

    test "bitwise_xor" do
      assert Nx.shape(bitwise_xor(@left, @right)) == {5, 5}
      assert bitwise_xor(@left, @right) == bitwise_xor_nx(@left, @right)
    end

    @left Nx.tensor([-2, -1, 0, 1, 2])
    @right Nx.tensor([[0], [1], [2], [3], [4]])

    defn left_shift(a, b), do: a <<< b
    @defn_compiler Nx.Defn
    defn left_shift_nx(a, b), do: a <<< b

    test "left_shift" do
      assert Nx.shape(left_shift(@left, @right)) == {5, 5}
      assert left_shift(@left, @right) == left_shift_nx(@left, @right)
    end

    @left_signed Nx.tensor([-128, -127, -2, -1, 0, 1, 2, 126, 127], type: {:s, 8})
    @right_signed Nx.tensor([[0], [1], [2], [3], [4], [5], [6], [7], [8]], type: {:s, 8})

    @left_unsigned Nx.tensor([0, 1, 2, 253, 254, 255], type: {:u, 8})
    @right_unsigned Nx.tensor([[0], [1], [2], [3], [4], [5]], type: {:u, 8})

    defn right_shift(a, b), do: a >>> b
    @defn_compiler Nx.Defn
    defn right_shift_nx(a, b), do: a >>> b

    test "right_shift" do
      assert Nx.shape(right_shift(@left_signed, @right_signed)) == {9, 9}

      assert right_shift(@left_signed, @right_signed) ==
               right_shift_nx(@left_signed, @right_signed)

      assert Nx.shape(right_shift(@left_unsigned, @right_unsigned)) == {6, 6}

      assert right_shift(@left_unsigned, @right_unsigned) ==
               right_shift_nx(@left_unsigned, @right_unsigned)
    end
  end

  describe "exp" do
    defn exp(t), do: Nx.exp(t)

    test "computes the exp across types" do
      assert Nx.tensor([1, 2, 3]) |> exp() |> Nx.to_bitstring() ==
               <<2.718281828459045::float-64-native, 7.38905609893065::float-64-native,
                 20.085536923187668::float-64-native>>

      assert Nx.tensor([1, 2, 3], type: {:s, 8}) |> exp() |> Nx.to_bitstring() ==
               <<2.718281828459045::float-32-native, 7.38905609893065::float-32-native,
                 20.085536923187668::float-32-native>>

      assert Nx.tensor([1, 2, 3], type: {:u, 8}) |> exp() |> Nx.to_bitstring() ==
               <<2.718281828459045::float-32-native, 7.38905609893065::float-32-native,
                 20.085536923187668::float-32-native>>

      assert Nx.tensor([1.0, 2.0, 3.0]) |> exp() |> Nx.to_bitstring() ==
               <<2.718281828459045::float-64-native, 7.38905609893065::float-64-native,
                 20.085536923187668::float-64-native>>

      assert Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}) |> exp() |> Nx.to_bitstring() ==
               <<2.718281828459045::float-32-native, 7.38905609893065::float-32-native,
                 20.085536923187668::float-32-native>>
    end

    defn exp_int(), do: Nx.exp(1)
    defn exp_float(), do: Nx.exp(1.0)

    test "constants" do
      assert exp_int() == Nx.tensor(2.718281828459045)
      assert exp_float() == Nx.tensor(2.718281828459045)
    end
  end

  describe "sum" do
    defn sum(t), do: Nx.sum(t)

    test "computes the sum across types" do
      assert Nx.tensor([1, 2, 3]) |> sum() |> Nx.to_bitstring() ==
               <<6::64-native>>

      assert Nx.tensor([1, 2, 3], type: {:s, 8}) |> sum() |> Nx.to_bitstring() ==
               <<6::8-native>>

      assert Nx.tensor([1, 2, 3], type: {:u, 8}) |> sum() |> Nx.to_bitstring() ==
               <<6::8-native>>

      assert Nx.tensor([1.0, 2.0, 3.0]) |> sum() |> Nx.to_bitstring() ==
               <<6::64-float-native>>

      assert Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}) |> sum() |> Nx.to_bitstring() ==
               <<6::32-float-native>>
    end
  end

  describe "softmax" do
    defn softmax(t), do: Nx.exp(t) / Nx.sum(Nx.exp(t))

    test "computes softmax" do
      tensor = softmax(Nx.tensor([1.0, 2.0, 3.0, 4.0]))

      assert Nx.type(tensor) == {:f, 64}
      assert Nx.shape(tensor) == {4}

      assert Nx.to_bitstring(tensor) ==
               <<0.03205860328008499::float-64-native, 0.08714431874203257::float-64-native,
                 0.23688281808991013::float-64-native, 0.6439142598879722::float-64-native>>
    end
  end

  describe "options" do
    @defn_compiler {Exla, keep_on_device: true}
    defn add_two_keep_on_device(a, b), do: a + b

    test "keeps data on device" do
      tensor = add_two_keep_on_device(1, 2)
      assert {Exla.NxDevice, {ref, :default}} = tensor.data
      assert is_reference(ref)
      assert tensor |> Nx.device_read() |> Nx.to_bitstring() == <<3::64-native>>

      tensor = add_two_keep_on_device(Nx.tensor([[1, 2], [3, 4]]), tensor)
      assert {Exla.NxDevice, {ref, :default}} = tensor.data
      assert is_reference(ref)

      assert tensor |> Nx.device_read() |> Nx.to_bitstring() ==
               <<4::64-native, 5::64-native, 6::64-native, 7::64-native>>

      assert tensor |> Nx.device_transfer() |> Nx.to_bitstring() ==
               <<4::64-native, 5::64-native, 6::64-native, 7::64-native>>

      assert_raise RuntimeError,
                   "Attempt to read from deallocated buffer.",
                   fn -> Nx.device_read(tensor) end
    end
  end
end
