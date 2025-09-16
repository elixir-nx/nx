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
end
