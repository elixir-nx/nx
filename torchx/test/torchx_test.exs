defmodule TorchxTest do
  use Torchx.Case, async: true

  doctest Torchx

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
      tensor = Nx.iota({6, 5, 4})

      slice = fn t -> Nx.slice(t, [1, 1, 1], [6, 5, 4], strides: [2, 3, 1]) end

      expected = tensor |> Nx.backend_copy(Nx.BinaryBackend) |> then(slice)

      result = slice.(tensor)

      assert expected |> Nx.equal(result) |> Nx.all() |> Nx.to_number() == 1
    end
  end

  describe "indexed_add" do
    test "clamps when out of bounds" do
      t = Nx.tensor([[1, 2], [3, 4]])

      result = Nx.indexed_add(t, Nx.tensor([[3, -10]]), Nx.tensor([50]))

      assert_all_close(result, Nx.tensor([[1, 2], [53, 4]]))
    end
  end

  describe "gather" do
    test "raises when out of bounds" do
      t = Nx.tensor([[1, 2], [3, 4]])

      result = Nx.gather(t, Nx.tensor([[3, -10]]))
      assert_all_close(result, Nx.tensor([3]))
    end
  end

  describe "concatenate" do
    test "works with mixed backends" do
      backends = [Nx.BinaryBackend, Torchx.Backend]

      for b1 <- backends, b2 <- backends do
        t1 = Nx.tensor([1, 2], backend: b1)
        t2 = Nx.tensor([3, 4], backend: b2)

        assert_equal(Nx.tensor([1, 2, 3, 4]), Nx.concatenate([t1, t2]))
      end
    end
  end
end
