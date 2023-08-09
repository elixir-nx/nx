defmodule CandlexTest do
  use ExUnit.Case
  doctest Candlex

  test "greets the world" do
    assert Candlex.hello() == :world
  end
end
