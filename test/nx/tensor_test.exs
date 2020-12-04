defmodule TensorTest do
  use ExUnit.Case, async: true

  test "inspect" do
    t1 = Nx.tensor(5)
    t2 = Nx.tensor([1, 2, 3])
    t3 = Nx.tensor([4.0, 5.0, 6.0])

    <<neg_inf::64-float-native>> = <<0xFF800000::64-float-native>>
    <<inf::64-float-native>> = <<0x7F800000::64-float-native>>
    <<nan::64-float-native>> = <<0x7FC00000::64-float-native>>

    t4 = Nx.tensor([[nan, neg_inf, 4.0, 5.0], [inf, neg_inf, nan, 2.0]])

    assert inspect(t1) == "#Nx.Tensor<\ns64\n5\n>"
    assert inspect(t2) == "#Nx.Tensor<\ns64[3]\n[1, 2, 3]\n>"
    assert inspect(t3) == "#Nx.Tensor<\nf64[3]\n[4.0, 5.0, 6.0]\n>"
    assert inspect(t4) == "#Nx.Tensor<\nf64[2][4]\n[[NaN, -Infinity, 4.0, 5.0], [Infinity, -Infinity, NaN, 2.0]]\n>"
  end
end