defmodule EXLA.NxRandomTest do
  use EXLA.Case, async: true

  setup do
    Nx.default_backend(EXLA.Backend)
    :ok
  end

  describe "range" do
    test "randint" do
      key = Nx.Random.key(127)

      assert_equal(
        Nx.Random.randint_split(key, 0, 100, shape: {10})
        |> Nx.less(0)
        |> Nx.any(),
        Nx.tensor(0, type: :u8)
      )

      assert_equal(
        Nx.Random.randint_split(key, -100, 0, shape: {10})
        |> Nx.less(-100)
        |> Nx.any(),
        Nx.tensor(0, type: :u8)
      )

      assert_equal(
        Nx.Random.randint_split(key, 0, Nx.Constants.max(:s64), shape: {10})
        |> Nx.less(0)
        |> Nx.any(),
        Nx.tensor(0, type: :u8)
      )

      assert_equal(
        Nx.Random.randint_split(key, Nx.Constants.min(:s64), 0, shape: {10})
        |> Nx.greater(0)
        |> Nx.any(),
        Nx.tensor(0, type: :u8)
      )
    end
  end

  @tag :cuda_required
  test "regression on single-dimensional and multi-dimensional Random.shuffle" do
    # these are put in the process dictionary, so it's thread-safe to do this
    Nx.default_backend({EXLA.Backend, client: :cuda})
    Nx.Defn.default_options(compiler: EXLA, client: :cuda)
    key = Nx.Random.key(127)

    t1 = Nx.iota({2, 100})
    t2 = Nx.iota({100})

    {t1_shuffled_0, key} = Nx.Random.shuffle(key, t1, axis: 0)
    {t1_shuffled_1, key} = Nx.Random.shuffle(key, t1, axis: 1)
    {t2_shuffled, _key} = Nx.Random.shuffle(key, t2)

    assert_equal(Nx.sort(t1_shuffled_0, axis: 0), t1)
    assert_equal(Nx.sort(t1_shuffled_1, axis: 1), t1)
    assert_equal(Nx.sort(t2_shuffled), t2)
  end
end
