defmodule TorchxTest do
  use ExUnit.Case, async: true

  describe "creation" do
    test "arange" do
      {:cpu, ref} = tensor = Torchx.arange(0, 26, 2, :short, :cpu)

      assert is_reference(ref)
      assert Torchx.scalar_type(tensor) == :short
      assert Torchx.shape(tensor) == {13}
    end
  end

  describe "operations" do
    test "dot" do
      a = Torchx.arange(0, 3, 1, :float, :cpu)
      b = Torchx.arange(4, 7, 1, :float, :cpu)

      {:cpu, ref} = Torchx.tensordot(a, b, [0], [0])
      assert is_reference(ref)
    end
  end

  describe "torchx<->nx" do
    test "to_nx" do
      assert Torchx.arange(0, 26, 1, :short, :cpu)
             |> Torchx.to_nx()
             |> Nx.backend_transfer() == Nx.iota({26}, type: {:s, 16}, backend: Nx.BinaryBackend)
    end

    test "from_nx" do
      tensor = Nx.iota({26}, type: {:s, 16})
      assert Nx.to_binary(tensor) == tensor |> Torchx.from_nx() |> Torchx.to_blob()
    end
  end

  describe "slice/4" do
    test "out of bound indices" do
      expected =
        Nx.tensor([
          [
            [0, 1, 2, 3],
            [12, 13, 14, 15]
          ],
          [
            [40, 41, 42, 43],
            [52, 53, 54, 55]
          ],
          [
            [80, 81, 82, 83],
            [92, 93, 94, 95]
          ]
        ])

      result =
        {6, 5, 4}
        |> Nx.iota()
        |> Nx.slice([1, 1, 1], [6, 5, 4], strides: [2, 3, 1])

      assert expected |> Nx.equal(result) |> Nx.all?() |> Nx.to_number() == 1
    end
  end
end
