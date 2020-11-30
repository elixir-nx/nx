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
        exla = add_two(left, right)
        nx = Nx.add(left, right)
        assert Nx.type(exla) == Nx.type(nx)
        assert Nx.shape(exla) == Nx.shape(nx)
        assert Nx.to_bitstring(exla) == Nx.to_bitstring(nx)

        exla = add_two(right, left)
        nx = Nx.add(right, left)
        assert Nx.type(exla) == Nx.type(nx)
        assert Nx.shape(exla) == Nx.shape(nx)
        assert Nx.to_bitstring(exla) == Nx.to_bitstring(nx)
      end
    end

    test "broadcast error" do
      assert_raise RuntimeError, ~r"Binary op add with incompatible shapes", fn ->
        add_two(Nx.tensor([1, 2]), Nx.tensor([1, 2, 3]))
      end
    end
  end

  describe "sum" do
    defn sum(t), do: Nx.sum(t)

    test "compures the sum across types" do
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

      assert Nx.to_bitstring(tensor) ==
               <<0.03205860328008499::float-64-native, 0.08714431874203257::float-64-native,
                 0.23688281808991013::float-64-native, 0.6439142598879722::float-64-native>>

      assert Nx.type(tensor) == {:f, 64}
      assert Nx.shape(tensor) == {4}
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
