defmodule Nx.TensorTest do
  use ExUnit.Case, async: true

  describe "inspect" do
    test "scalar" do
      assert inspect(Nx.tensor(123)) == """
             #Nx.Tensor<
               s64
               123
             >\
             """
    end

    test "n-dimensional" do
      assert inspect(Nx.tensor([[1, 2, 3], [4, 5, 6]])) == """
             #Nx.Tensor<
               s64[2][3]
               [
                 [1, 2, 3],
                 [4, 5, 6]
               ]
             >\
             """

      assert inspect(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])) == """
             #Nx.Tensor<
               f64[2][3]
               [
                 [1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0]
               ]
             >\
             """
    end

    test "custom device" do
      t = Nx.tensor([1, 2, 3, 4])
      t = Nx.device_transfer(t, Nx.ProcessDevice, key: :example)

      assert inspect(t) == """
             #Nx.Tensor<
               s64[4]
               Nx.ProcessDevice
             >\
             """
    end

    test "limit" do
      assert inspect(Nx.tensor([1, 2]), limit: :infinity) == """
             #Nx.Tensor<
               s64[2]
               [1, 2]
             >\
             """

      assert inspect(Nx.tensor([[1, 2], [3, 4]]), limit: 3) == """
             #Nx.Tensor<
               s64[2][2]
               [
                 [1, 2],
                 [3, ...]
               ]
             >\
             """

      assert inspect(Nx.tensor([[1, 2], [3, 4], [5, 6]]), limit: 3) == """
             #Nx.Tensor<
               s64[3][2]
               [
                 [1, 2],
                 [3, ...],
                 ...
               ]
             >\
             """
    end

    test "infinity and nan for bf16" do
      bin = <<0xFF80::16-native, 0x7F80::16-native, 0xFFC1::16-native, 0xFF81::16-native>>

      assert inspect(Nx.from_binary(bin, {:bf, 16})) == """
             #Nx.Tensor<
               bf16[4]
               [-Inf, Inf, NaN, NaN]
             >\
             """
    end

    test "infinity and nan for f32" do
      bin =
        <<0xFF800000::32-native, 0x7F800000::32-native, 0xFF800001::32-native,
          0xFFC00001::32-native>>

      # Assert that none of them are indeed valid
      assert for(<<x::float-32-native <- bin>>, do: x) == []

      assert inspect(Nx.from_binary(bin, {:f, 32})) == """
             #Nx.Tensor<
               f32[4]
               [-Inf, Inf, NaN, NaN]
             >\
             """
    end

    test "infinity and nan for f64" do
      bin =
        <<0xFFF0000000000000::64-native, 0x7FF0000000000000::64-native,
          0x7FF0000000000001::64-native, 0x7FF8000000000001::64-native>>

      # Assert that none of them are indeed valid
      assert for(<<x::float-64-native <- bin>>, do: x) == []

      assert inspect(Nx.from_binary(bin, {:f, 64})) == """
             #Nx.Tensor<
               f64[4]
               [-Inf, Inf, NaN, NaN]
             >\
             """
    end
  end
end
