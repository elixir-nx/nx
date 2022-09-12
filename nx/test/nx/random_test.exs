defmodule Nx.RandomTest do
  use ExUnit.Case, async: true

  import Nx.Helpers

  doctest Nx.Random

  describe "key/1" do
    test "transforms given integer into PRNG key" do
      key = Nx.Random.key(44)
      assert key == Nx.tensor([0, 44], type: :u32)
    end
  end

  describe "split/2" do
    test "splits key into multiple keys" do
      key = Nx.Random.key(33)

      two_keys = Nx.Random.split(key)
      multiple_keys = Nx.Random.split(key, 12)

      assert two_keys == Nx.tensor([
        [671281441, 790285293],
        [234515160, 3878582434]
      ], type: :u32)
      assert multiple_keys == Nx.tensor([
        [966561810, 334783285],
        [1262072629, 1899563600],
        [3750833143, 3406870597],
        [2539864401, 3552854032],
        [201687315, 590048257],
        [3348546826, 4091268549],
        [1610907819, 3073378539],
        [3054273782, 2286163366],
        [4120769120, 1468859077],
        [2405343452, 1650615538],
        [4063810472, 2490879298],
        [259087434, 3260250733]
      ], type: :u32)
    end
  end

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
