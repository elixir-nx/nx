defmodule Nx.BinaryBackend.ReducersTest do
  use ExUnit.Case, async: true

  alias Nx.BinaryBackend.BinReducer
  alias Nx.BinaryBackend.TraverserReducer
  alias Nx.BinaryBackend.Bits

  test "reducers match" do
    t1 = Nx.iota({3, 2}, type: {:u, 8})
    t2 = Nx.iota({2, 3}, type: {:u, 8})
    out = Nx.iota({3, 3}, type: {:u, 8})

    reducer = fn lhs, rhs, acc ->
      res = Bits.to_number(lhs, {:u, 8}) * Bits.to_number(rhs, {:u, 8}) + acc
      {res, res}
    end

    bin_out = BinReducer.bin_zip_reduce(out, t1, [1], t2, [0], 0, reducer)
    trav_out = TraverserReducer.bin_zip_reduce(out, t1, [1], t2, [0], 0, reducer)
    bin_out = IO.iodata_to_binary(bin_out)
    trav_out = IO.iodata_to_binary(trav_out)
    assert bin_out == trav_out, """
    bin_out:  #{inspect(bin_out, binaries: :as_binaries)}
    trav_out: #{inspect(trav_out, binaries: :as_binaries)}
    """
  end
end