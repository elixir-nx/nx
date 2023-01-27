defmodule Torchx.Nx.RandomTest do
  use Torchx.Case, async: true

  doctest Nx.Random, except: [normal: 2, normal: 4]

  describe "key/1" do
    test "transforms given integer into PRNG key" do
      key = Nx.Random.key(44)
      assert_equal(key, Nx.tensor([0, 44], type: :u32))
    end
  end

  describe "split/2" do
    test "splits key into multiple keys" do
      key = Nx.Random.key(33)

      two_keys = Nx.Random.split(key)

      assert_equal(
        two_keys,
        Nx.tensor(
          [
            [671_281_441, 790_285_293],
            [234_515_160, 3_878_582_434]
          ],
          type: :u32
        )
      )

      multiple_keys = Nx.Random.split(key, parts: 12)

      assert_equal(
        multiple_keys,
        Nx.tensor(
          [
            [966_561_810, 334_783_285],
            [1_262_072_629, 1_899_563_600],
            [3_750_833_143, 3_406_870_597],
            [2_539_864_401, 3_552_854_032],
            [201_687_315, 590_048_257],
            [3_348_546_826, 4_091_268_549],
            [1_610_907_819, 3_073_378_539],
            [3_054_273_782, 2_286_163_366],
            [4_120_769_120, 1_468_859_077],
            [2_405_343_452, 1_650_615_538],
            [4_063_810_472, 2_490_879_298],
            [259_087_434, 3_260_250_733]
          ],
          type: :u32
        )
      )
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
      # Output does not match Nx because of the sign of the remainder.
      distribution_case(:randint_split,
        args: [0, 10, [shape: {5}]],
        expected: Nx.tensor([3, 2, 6, 0, 0], type: :s64)
      )
    end

    test "uniform" do
      distribution_case(:uniform_split,
        args: [0.0, 1.0, [shape: {5}]],
        expected:
          Nx.tensor([0.40235483, 0.86897706, 0.87899947, 0.98497891, 0.83098864], type: :f32)
      )
    end
  end

  describe "types" do
    test "randint" do
      key = Nx.Random.key(44)

      # Default
      assert_equal(
        Nx.Random.randint_split(key, Nx.tensor(0, type: :u8), Nx.tensor(100, type: :u8)),
        Nx.tensor(12, type: :u8)
      )

      # Explicit
      assert_equal(Nx.Random.randint_split(key, 0, 100, type: :u8), Nx.tensor(12, type: :u8))
      assert_equal(Nx.Random.randint_split(key, 0, 100, type: :u64), Nx.tensor(10, type: :u64))
      assert_equal(Nx.Random.randint_split(key, 0, 100, type: :s64), Nx.tensor(10, type: :s64))
    end

    test "uniform" do
      key = Nx.Random.key(44)

      # default
      assert_equal(
        Nx.Random.uniform_split(key, 0, 100),
        Nx.tensor(0.9405970573425293, type: :f32)
      )

      # inference
      assert_equal(
        Nx.Random.uniform_split(key, Nx.tensor(0, type: :bf16), Nx.tensor(100, type: :bf16)),
        Nx.tensor(43.0, type: :bf16)
      )

      # int to float cast
      assert_equal(
        Nx.Random.uniform_split(key, 0, 100, type: :bf16),
        Nx.tensor(43.0, type: :bf16)
      )

      # f32 to bf16 downcast
      assert_equal(
        Nx.Random.uniform_split(key, 0.0, 100.0, type: :bf16),
        Nx.tensor(43.0, type: :bf16)
      )

      # upcast
      assert_equal(
        Nx.Random.uniform_split(key, 0.0, 100.0, type: :f64),
        Nx.tensor(49.70372348385783, type: :f64)
      )
    end

    @tag :skip_apple_arm64
    test "normal" do
      key = Nx.Random.key(44)

      # default
      assert_equal(Nx.Random.normal_split(key, 0, 100), Nx.tensor(-234.9235382080078, type: :f32))

      # inference
      assert_equal(
        Nx.Random.normal_split(key, Nx.tensor(0, type: :bf16), Nx.tensor(100, type: :bf16)),
        Nx.tensor(-17.25, type: :bf16)
      )

      # int to float cast
      assert_equal(
        Nx.Random.normal_split(key, 0, 100, type: :bf16),
        Nx.tensor(-17.25, type: :bf16)
      )

      # f32 to bf16 downcast
      assert_equal(
        Nx.Random.normal_split(key, 0.0, 100.0, type: :bf16),
        Nx.tensor(-17.25, type: :bf16)
      )

      # f32 to f16 downcast
      assert_equal(
        Nx.Random.normal_split(key, 0.0, 100.0, type: :f16),
        Nx.tensor(-17.15625, type: :f16)
      )

      # upcast
      assert_equal(
        Nx.Random.normal_split(key, 0, 100, type: :f64),
        Nx.tensor(-0.7426619192938216, type: :f64)
      )

      # complex
      assert_all_close(
        Nx.Random.normal_split(key, 0.0, 100.0, type: :c64),
        Nx.complex(-0.74267750, 6.5133848)
      )
    end
  end

  describe "properties" do
    defp continuous_uniform_variance(a, b) do
      (b - a) ** 2 / 12
    end

    defp discrete_uniform_second_moment(count) do
      (count - 1) * (count + 1) / 12
    end

    defp property_case(name,
           args: args,
           moment: moment,
           expected_func: expected_func,
           expected_args: expected_args
         ) do
      seed = :erlang.adler32("#{name}threefry2x32")
      key = Nx.Random.key(Nx.tensor(seed, type: :s64))
      t = apply(Nx.Random, name, [key | args])

      apply(Nx, moment, [t])
      |> assert_all_close(apply(expected_func, expected_args), rtol: 0.1)

      seed = :erlang.adler32("#{name}threefry2x32")
      key = Nx.Random.key(Nx.tensor(seed, type: :u64))
      t = apply(Nx.Random, name, [key | args])

      apply(Nx, moment, [t])
      |> assert_all_close(apply(expected_func, expected_args), rtol: 0.1)
    end

    test "uniform mean property" do
      property_case(:uniform_split,
        args: [10, 15, [shape: {10000}]],
        moment: :mean,
        expected_func: fn x -> Nx.tensor(x) end,
        expected_args: [12.5]
      )
    end

    test "randint mean property" do
      property_case(:randint_split,
        args: [10, 95, [shape: {10000}]],
        moment: :mean,
        expected_func: fn x -> Nx.tensor(x) end,
        expected_args: [52.5]
      )
    end

    test "uniform variance property" do
      property_case(:uniform_split,
        args: [Nx.tensor(10), Nx.tensor(15), [shape: {10000}]],
        moment: :variance,
        expected_func: fn a, b -> continuous_uniform_variance(a, b) end,
        expected_args: [10, 15]
      )
    end

    test "randint variance property" do
      property_case(:randint_split,
        args: [Nx.tensor(10), Nx.tensor(95), [shape: {10000}]],
        moment: :variance,
        expected_func: fn x -> discrete_uniform_second_moment(x) end,
        expected_args: [85]
      )
    end

    test "normal properties" do
      key = Nx.Random.key(:rand.uniform(10_000))
      normal = Nx.Random.normal_split(key, 10, 5, shape: {1_000})

      assert_all_close(Nx.mean(normal), 10, rtol: 0.1)
      assert_all_close(Nx.standard_deviation(normal), 5, rtol: 0.1)
    end
  end
end
