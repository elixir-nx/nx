defmodule Nx.RandomTest do
  use ExUnit.Case, async: true

  describe "threefry_seed/1" do
    test "transforms given integer into PRNG key" do
      key = Nx.Random.threefry_seed(44)

      assert key |> Nx.type() |> Nx.Type.integer?()
      assert Nx.shape(key) == {2}
    end
  end

  describe "threefry_split/2" do
    test "splits key into multiple keys" do
      key = Nx.Random.threefry_seed(33)

      two_keys = Nx.Random.threefry_split(key)
      multiple_keys = Nx.Random.threefry_split(key, 12)

      assert Nx.shape(two_keys) == {2, 2}
      assert Nx.shape(multiple_keys) == {12, 2}
    end
  end

  describe "fold_in/2" do
    test "incorporates integer data into PRNG key" do
      key = Nx.Random.threefry_seed(22)

      keys = Enum.map(1..10, &Nx.Random.fold_in(key, &1))
      assert keys |> Enum.uniq() |> length() == 10
    end

    test "bigger data" do
      key = Nx.Random.threefry_seed(23)

      data = [2 ** 32 - 2, 2 ** 32 - 1]
      keys = Enum.map(data, &Nx.Random.fold_in(key, &1))
      assert keys |> Enum.uniq() |> length() == 2
    end
  end

  describe "random_bits/2" do
    test "generates random 32 bit numbers from a key" do
      key = Nx.Random.threefry_seed(1701)

      bits = Nx.Random.random_bits(key)
      expected = Nx.tensor([741_045_208], type: :u32)
      assert bits == expected
    end

    test "accepts custom shape" do
      key = Nx.Random.threefry_seed(1701)

      bits = Nx.Random.random_bits(key, {3})
      expected = Nx.tensor([56_197_195, 4_200_222_568, 961_309_823], type: :u32)

      assert bits == expected

      bits = Nx.Random.random_bits(key, {3, 2})

      expected =
        Nx.tensor(
          [
            [927_208_350, 3_916_705_582],
            [1_835_323_421, 676_898_860],
            [3_164_047_411, 4_010_691_890]
          ],
          type: :u32
        )

      assert bits == expected
    end
  end

  defp to_hex(tensor) do
    tensor
    |> Nx.to_flat_list()
    |> Enum.map(&Integer.to_string(&1, 16))
  end

  describe "threefry2x32/2" do
    test "matches known results from reference implementation" do
      # values from https://github.com/DEShawResearch/Random123-Boost/blob/65e3d874b67aa7b3e02d5ad8306462f52d2079c0/libs/random/test/test_threefry.cpp#L30-L32

      expected_results = [
        ["6B200159", "99BA4EFE"],
        ["1CB996FC", "BB002BE7"],
        ["C4923A9C", "483DF7A0"]
      ]

      inputs = [
        [
          Nx.tensor([[0], [0]], type: :u32),
          Nx.tensor([[0], [0]], type: :u32)
        ],
        [
          Nx.tensor([[-1], [-1]], type: :u32),
          Nx.tensor([[-1], [-1]], type: :u32)
        ],
        [
          Nx.tensor([[0x13198A2E], [0x03707344]], type: :u32),
          Nx.tensor([[0x243F6A88], [0x85A308D3]], type: :u32)
        ]
      ]

      for {expected, {key, count}} <- Enum.zip(expected_results, inputs) do
        result = Nx.Random.threefry2x32(key, count)
        assert to_hex(result) == expected
      end
    end
  end
end
