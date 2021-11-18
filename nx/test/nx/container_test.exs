defmodule Nx.ContainerTest do
  use ExUnit.Case, async: true

  test "to_template" do
    assert Nx.to_template(%Container{a: 1, b: 2, c: 3}) ==
             %Container{a: Nx.template({}, {:s, 64}), b: Nx.template({}, {:s, 64}), c: 3}

    assert Nx.to_template(%Container{a: 1, b: {2, 3.0}, c: 4}) ==
             %Container{
               a: Nx.template({}, {:s, 64}),
               b: {Nx.template({}, {:s, 64}), Nx.template({}, {:f, 32})},
               c: 4
             }
  end

  test "compatible?" do
    assert Nx.compatible?(%Container{a: 1, b: 2, c: 3}, %Container{a: 4, b: 5, c: 6})
    assert Nx.compatible?(%Container{a: 1, b: 2, c: 3}, %Container{a: 4, b: 5, c: 6.0})
    refute Nx.compatible?(%Container{a: 1, b: 2, c: 3}, %Container{a: 4.0, b: 5.0, c: 6.0})
    refute Nx.compatible?(%Container{a: 1, b: 2, c: 3}, %URI{})
  end
end
