defmodule Nx.BinaryBackendTest do
  use ExUnit.Case, async: true

  alias Nx.BinaryBackend

  describe "dot/6" do
    test "works for batching {2, 1, 1}" do
      t1 = Nx.tensor([10, 20])
      t1 = Nx.reshape(t1, {2, 1, 1})
      t2 = Nx.tensor([30, 40])
      t2 = Nx.reshape(t2, {2, 1, 1})

      out1 = Nx.iota({2, 1, 1})
      out2 = BinaryBackend.dot(out1, t1, [2], [0], t2, [1], [0])
      assert Nx.shape(out2) == {2, 1, 1}
      expected = Nx.tensor([[[300]], [[800]]])
      assert Nx.shape(expected) == {2, 1, 1}
      assert out2 == expected
    end

    test "works for batching {2, 2, 2}" do
      t1 = Nx.tensor([
        [
          [1, 2],
          [3, 4]
        ],
        [
          [5, 6],
          [7, 8]
        ]
      ])
      t2 = Nx.tensor([
        [
          [0, 2],
          [4, 6]
        ],
        [
          [8, 10],
          [12, 14]
        ]
      ])

      IO.inspect([t1: t1, t2: t2], label: :t1_and_t2)
      out1 = Nx.iota({2, 2, 2})
      out2 = BinaryBackend.dot(out1, t1, [2], [0], t2, [1], [0])

      expected = Nx.tensor([
        [
          [8, 14],
          [16, 30]
        ],
        [
          [112, 134],
          [152, 182]
        ]
      ])
      assert Nx.shape(out2) == {2, 2, 2}
      assert Nx.shape(expected) == {2, 2, 2}
      assert out2 == expected
    end
  end
end