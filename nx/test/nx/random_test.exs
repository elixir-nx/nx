defmodule Nx.RandomTest do
  use ExUnit.Case, async: true

  import Nx.Helpers

  doctest Nx.Random

  describe "key/1" do
    test "transforms given integer into PRNG key" do
      key = Nx.Random.key(44)

      assert key |> Nx.type() |> Nx.Type.integer?()
      assert Nx.shape(key) == {2}
    end
  end

  describe "split/2" do
    test "splits key into multiple keys" do
      key = Nx.Random.key(33)

      two_keys = Nx.Random.split(key)
      multiple_keys = Nx.Random.split(key, 12)

      assert Nx.shape(two_keys) == {2, 2}
      assert Nx.shape(multiple_keys) == {12, 2}
    end
  end

  # describe "fold_in/2" do
  #   test "incorporates integer data into PRNG key" do
  #     key = Nx.Random.key(22)

  #     keys = Enum.map(1..10, &Nx.Random.fold_in(key, &1))
  #     assert keys |> Enum.uniq() |> length() == 10
  #   end

  #   test "bigger data" do
  #     key = Nx.Random.key(23)

  #     data = [2 ** 32 - 2, 2 ** 32 - 1]
  #     keys = Enum.map(data, &Nx.Random.fold_in(key, &1))
  #     assert keys |> Enum.uniq() |> length() == 2
  #   end
  # end

  # describe "random_bits/2" do
  #   test "generates random 32 bit numbers from a key" do
  #     key = Nx.Random.key(1701)

  #     bits = Nx.Random.random_bits(key)
  #     expected = Nx.tensor([741_045_208], type: :u32)
  #     assert bits == expected
  #   end

  #   test "accepts custom shape" do
  #     key = Nx.Random.key(1701)

  #     bits = Nx.Random.random_bits(key, shape: {3})
  #     expected = Nx.tensor([56_197_195, 4_200_222_568, 961_309_823], type: :u32)

  #     assert bits == expected

  #     bits = Nx.Random.random_bits(key, shape: {3, 2})

  #     expected =
  #       Nx.tensor(
  #         [
  #           [927_208_350, 3_916_705_582],
  #           [1_835_323_421, 676_898_860],
  #           [3_164_047_411, 4_010_691_890]
  #         ],
  #         type: :u32
  #       )

  #     assert bits == expected
  #   end
  # end

  # describe "threefry2x32/2" do
  #   test "matches known results from reference implementation" do
  #     # values from https://github.com/DEShawResearch/Random123-Boost/blob/65e3d874b67aa7b3e02d5ad8306462f52d2079c0/libs/random/test/test_threefry.cpp#L30-L32

  #     expected_results = [
  #       Nx.tensor([0x6B200159, 0x99BA4EFE], type: :u32),
  #       Nx.tensor([0x1CB996FC, 0xBB002BE7], type: :u32),
  #       Nx.tensor([0xC4923A9C, 0x483DF7A0], type: :u32)
  #     ]

  #     inputs = [
  #       [
  #         Nx.tensor([[0], [0]], type: :u32),
  #         Nx.tensor([[0], [0]], type: :u32)
  #       ],
  #       [
  #         Nx.tensor([[-1], [-1]], type: :u32),
  #         Nx.tensor([[-1], [-1]], type: :u32)
  #       ],
  #       [
  #         Nx.tensor([[0x13198A2E], [0x03707344]], type: :u32),
  #         Nx.tensor([[0x243F6A88], [0x85A308D3]], type: :u32)
  #       ]
  #     ]

  #     for {expected, {key, count}} <- Enum.zip(expected_results, inputs) do
  #       result = Nx.Random.threefry2x32(key, count)
  #       assert result == expected
  #     end
  #   end
  # end

  describe "distributions" do
    defp distribution_case(name, args: args, expected: expected) do
      seed = :erlang.adler32("#{name}threefry2x32")
      key = Nx.Random.key(seed)
      actual = apply(Nx.Random, name, [key | args])

      assert_all_close(actual, expected)
    end

    test "randint" do
      distribution_case(:randint,
        args: [0, 10, [shape: {5}]],
        expected: Nx.tensor([9, 9, 5, 8, 7], type: :s64)
      )
    end

    test "uniform" do
      distribution_case(:uniform,
        args: [[shape: {5}]],
        expected: Nx.tensor([0.298671, 0.073213, 0.873356, 0.260549, 0.412797], type: :f32)
      )
    end
  end

  describe "properties" do
    defp continuous_uniform_variance(a, b) do
      (b - a) ** 2 / 12
    end

    # about the mean
    defp discrete_uniform_second_moment(count) do
      (count - 1) * (count + 1) / 12
    end

    defp property_case(name,
           args: args,
           moment: moment,
           expected_func: expected_func,
           expected_args: expected_args
         ) do
      seed = :erlang.adler32("uniformthreefry2x32")
      key = Nx.Random.key(Nx.tensor(seed, type: :s64))
      t = apply(Nx.Random, name, [key | args])

      apply(Nx, moment, [t])
      |> assert_all_close(apply(expected_func, expected_args), rtol: 0.1)

      seed = :erlang.adler32("uniformthreefry2x32")
      key = Nx.Random.key(Nx.tensor(seed, type: :u64))
      t = apply(Nx.Random, name, [key | args])

      apply(Nx, moment, [t])
      |> assert_all_close(apply(expected_func, expected_args), rtol: 0.1)
    end

    test "uniform mean property" do
      property_case(:uniform,
        args: [[min_val: 10, max_val: 15, shape: {10000}]],
        moment: :mean,
        expected_func: fn x -> Nx.tensor(x) end,
        expected_args: [12.5]
      )
    end

    test "randint mean property" do
      property_case(:randint,
        args: [10, 95, [shape: {10000}]],
        moment: :mean,
        expected_func: fn x -> Nx.tensor(x) end,
        expected_args: [52.5]
      )
    end

    test "uniform variance property" do
      property_case(:uniform,
        args: [[min_val: 10, max_val: 15, shape: {10000}]],
        moment: :variance,
        expected_func: fn a, b -> continuous_uniform_variance(a, b) end,
        expected_args: [10, 15]
      )
    end

    test "randint variance property" do
      property_case(:randint,
        args: [10, 95, [shape: {10000}]],
        moment: :variance,
        expected_func: fn x -> discrete_uniform_second_moment(x) end,
        expected_args: [85]
      )
    end
  end
end
