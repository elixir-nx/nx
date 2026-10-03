defmodule Nx.ShapeTest do
  use ExUnit.Case, async: true

  doctest Nx.Shape

  test "to_padding_config raises on an invalid padding mode" do
    assert_raise ArgumentError, ~r/invalid padding mode specified.*got: :foo/, fn ->
      apply(Nx.Shape, :to_padding_config, [{2, 3, 2}, {2, 3, 2}, :foo])
    end
  end

  test "gather raises on a scalar indices shape" do
    assert_raise ArgumentError, "expected indices rank to be at least 1, got: 0", fn ->
      apply(Nx.Shape, :gather, [{2, 3}, {}, []])
    end
  end

  test "conv with empty dimensions raises" do
    assert_raise ArgumentError, ~r/conv would result/, fn ->
      names = [nil, nil, nil]

      Nx.Shape.conv(
        {1, 1, 1},
        names,
        {1, 1, 2},
        names,
        [1],
        [{0, 0}],
        1,
        1,
        [1],
        [1],
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2]
      )
    end
  end
end
