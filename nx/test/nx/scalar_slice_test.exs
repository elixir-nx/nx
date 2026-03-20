defmodule Nx.ScalarSliceTest do
  use ExUnit.Case, async: true

  test "slice of scalar tensor returns scalar" do
    t = Nx.tensor(42)
    result = Nx.slice(t, [], [])
    assert Nx.to_number(result) == 42
  end

  test "slice of scalar f64 tensor" do
    t = Nx.tensor(3.14, type: :f64)
    result = Nx.slice(t, [], [])
    assert_in_delta Nx.to_number(result), 3.14, 1.0e-10
  end
end
