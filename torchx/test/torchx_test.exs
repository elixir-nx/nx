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

  describe "indexed_put" do
    test "clamps when out of bounds" do
      t = Nx.tensor([[1, 2], [3, 4]])

      result = Nx.indexed_put(t, Nx.tensor([[3, -10]]), Nx.tensor([50]))

      assert_all_close(result, Nx.tensor([[1, 2], [50, 4]]))
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

  describe "bool type" do
    test "adds correctly" do
      t_tx =
        {2, 3}
        |> Nx.iota(type: {:u, 8})
        |> Torchx.from_nx()
        |> Torchx.to_type(:bool)

      t = Torchx.to_nx(t_tx)

      # Show that bools don't add as expected for u8
      # This is why we add the cast to byte in Torchx.to_nx
      assert_equal(Torchx.to_nx(t_tx), t_tx |> Torchx.add(t_tx) |> Torchx.to_nx())

      assert_equal(Nx.add(t, t), Nx.tensor([[0, 2, 2], [2, 2, 2]]))
    end

    test "works with argmax and argmin" do
      t =
        {2, 3}
        |> Nx.iota(type: {:u, 8})
        |> Torchx.from_nx()
        |> Torchx.to_type(:bool)
        |> Torchx.to_nx()

      assert_equal(0, Nx.argmin(t))
      assert_equal(1, Nx.argmax(t))
    end
  end

  describe "pad" do
    test "works with inner and outer padding" do
      t = Nx.iota({2, 4, 3})
      padding_config = [{1, 1, 1}, {1, 1, 1}, {1, 1, 1}]

      all_zeros = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
      ]

      assert_equal(
        Nx.pad(t, 0, padding_config),
        Nx.tensor([
          all_zeros,
          [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 2, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 4, 0, 5, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 6, 0, 7, 0, 8, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 9, 0, 10, 0, 11, 0],
            [0, 0, 0, 0, 0, 0, 0]
          ],
          all_zeros,
          [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 12, 0, 13, 0, 14, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 15, 0, 16, 0, 17, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 18, 0, 19, 0, 20, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 21, 0, 22, 0, 23, 0],
            [0, 0, 0, 0, 0, 0, 0]
          ],
          all_zeros
        ])
      )
    end

    # [
    #     [0, 1, 2]
    #     [3, 4, 5],
    #     [6, 7, 8],
    #     [9, 10, 11]
    #   ],
    #   [
    #     [12, 13, 14],
    #     [15, 16, 17],
    #     [18, 19, 20],
    #     [21, 22, 23]
    #   ]

    test "works with inner padding and left negative pading" do
      t = Nx.iota({2, 4, 3})
      padding_config = [{-1, 1, 1}, {-2, 1, 1}, {-2, 1, 1}]

      assert_equal(
        Nx.pad(t, 0, padding_config),
        Nx.tensor([
          [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
          ],
          [
            [16, 0, 17, 0],
            [0, 0, 0, 0],
            [19, 0, 20, 0],
            [0, 0, 0, 0],
            [22, 0, 23, 0],
            [0, 0, 0, 0]
          ],
          [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
          ]
        ])
      )
    end

    test "works with inner padding and right negative pading" do
      t = Nx.iota({2, 4, 3})
      padding_config = [{1, -1, 1}, {1, -2, 1}, {1, -2, 1}]

      assert_equal(
        Nx.pad(t, 0, padding_config),
        Nx.tensor([
          [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
          ],
          [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 3, 0, 4],
            [0, 0, 0, 0],
            [0, 6, 0, 7]
          ],
          [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
          ]
        ])
      )
    end

    test "works with inner padding and left and right negative pading" do
      t = Nx.iota({2, 4, 3})
      padding_config = [{1, -2, 1}, {-1, -2, 1}, {-1, -2, 1}]

      assert_equal(
        Nx.pad(t, 0, padding_config),
        Nx.tensor([
          [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
          ],
          [
            [0, 0],
            [0, 4],
            [0, 0],
            [0, 7]
          ]
        ])
      )
    end
  end
end
