defmodule Nx.ShapeTest do
  use ExUnit.Case, async: true

  doctest Nx.Shape

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
