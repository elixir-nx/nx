defmodule Nx.LinAlgTest do
  use ExUnit.Case, async: true

  doctest Nx.LinAlg

  describe "norm/2" do
    test "raises for rank 3 or greater tensors" do
      t = Nx.iota({2, 2, 2})

      assert_raise(
        ArgumentError,
        "expected 1-D or 2-D tensor, got tensor with shape {2, 2, 2}",
        fn ->
          Nx.LinAlg.norm(t)
        end
      )
    end

    test "raises for unknown :ord value" do
      t = Nx.iota({2, 2})

      assert_raise(ArgumentError, "unknown ord :blep", fn ->
        Nx.LinAlg.norm(t, ord: :blep)
      end)
    end

    test "raises for invalid :ord integer value" do
      t = Nx.iota({2, 2})

      assert_raise(ArgumentError, "invalid :ord for 2-D tensor, got: -3", fn ->
        Nx.LinAlg.norm(t, ord: -3)
      end)
    end
  end

  describe "qr" do
    test "correctly factors a square matrix" do
      t = Nx.tensor([[2, -2, 18], [2, 1, 0], [1, 2, 0]])
      assert {q, %{type: output_type} = r} = Nx.LinAlg.qr(t)
      assert t |> Nx.round() |> Nx.as_type(output_type) == q |> Nx.dot(r) |> Nx.round()

      assert round(q, 1) ==
               Nx.tensor([
                 [2 / 3, 2 / 3, 1 / 3],
                 [2 / 3, -1 / 3, -2 / 3],
                 [1 / 3, -2 / 3, 2 / 3]
               ])
               |> round(1)

      assert round(r, 1) ==
               Nx.tensor([
                 [3.0, 0.0, 12.0],
                 [0.0, -3.0, 12.0],
                 [0.0, 0.0, 6.0]
               ])
               |> round(1)
    end

    test "factors rectangular matrix" do
      t = Nx.tensor([[1.0, -1.0, 4.0], [1.0, 4.0, -2.0], [1.0, 4.0, 2.0], [1.0, -1.0, 0.0]])
      {q, r} = Nx.LinAlg.qr(t, mode: :reduced)

      assert round(q, 1) ==
               Nx.tensor([
                 [0.5774, -0.8165, 0.0],
                 [0.5774, 0.4082, -0.7071],
                 [0.5774, 0.4082, 0.7071],
                 [0.0, 0.0, 0.0]
               ])
               |> round(1)

      assert round(r, 1) ==
               Nx.tensor([
                 [1.7321, 4.0415, 2.3094],
                 [0.0, 4.0825, -3.266],
                 [0.0, 0.0, 2.8284]
               ])
               |> round(1)

      assert Nx.tensor([
               [1.0, -1.0, 4.0],
               [1.0, 4.0, -2.0],
               [1.0, 4.0, 2.0],
               [0.0, 0.0, 0.0]
             ]) == q |> Nx.dot(r) |> round(1)
    end
  end

  describe "svd" do
    test "correctly finds the singular values of full matrices" do
      t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

      assert {%{type: output_type} = u, %{type: output_type} = s, %{type: output_type} = v} =
               Nx.LinAlg.svd(t, max_iter: 1000)

      zero_row = List.duplicate(0, 3)

      # turn s into a {4, 3} tensor
      s_matrix =
        s
        |> Nx.to_flat_list()
        |> Enum.with_index()
        |> Enum.map(fn {x, idx} -> List.replace_at(zero_row, idx, x) end)
        |> Enum.concat([zero_row])
        |> Nx.tensor()

      assert round(t, 2) == u |> Nx.dot(s_matrix) |> Nx.dot(v) |> round(2)

      assert round(u, 3) ==
               Nx.tensor([
                 [-0.141, -0.825, -0.547, 0.019],
                 [-0.344, -0.426, 0.744, 0.382],
                 [-0.547, -0.028, 0.153, -0.822],
                 [-0.75, 0.371, -0.35, 0.421]
               ])
               |> round(3)

      assert Nx.tensor([25.462, 1.291, 0.0]) |> round(3) == round(s, 3)

      assert Nx.tensor([
               [-0.505, -0.575, -0.644],
               [0.761, 0.057, -0.646],
               [-0.408, 0.816, -0.408]
             ])
             |> round(3) == round(v, 3)
    end

    test "correctly finds the singular values triangular matrices" do
      t = Nx.tensor([[1.0, 2.0, 3.0], [0.0, 4.0, 0.0], [0.0, 0.0, 9.0]])

      assert {%{type: output_type} = u, %{type: output_type} = s, %{type: output_type} = v} =
               Nx.LinAlg.svd(t)

      zero_row = List.duplicate(0, 3)

      # turn s into a {4, 3} tensor
      s_matrix =
        s
        |> Nx.to_flat_list()
        |> Enum.with_index()
        |> Enum.map(fn {x, idx} -> List.replace_at(zero_row, idx, x) end)
        |> Nx.tensor()

      assert round(t, 2) == u |> Nx.dot(s_matrix) |> Nx.dot(v) |> round(2)

      assert round(u, 3) ==
               Nx.tensor([
                 [-0.335, 0.408, -0.849],
                 [-0.036, 0.895, 0.445],
                 [-0.941, -0.18, 0.286]
               ])
               |> round(3)

      # The expected value is ~ [9, 5, 1] since the eigenvalues of
      # a triangular matrix are the diagonal elements. Close enough!
      assert Nx.tensor([9.52, 4.433, 0.853]) |> round(3) == round(s, 3)

      assert Nx.tensor([
               [-0.035, -0.086, -0.996],
               [0.092, 0.992, -0.089],
               [-0.995, 0.095, 0.027]
             ])
             |> round(3) == round(v, 3)
    end
  end

  defp round(tensor, places) do
    Nx.map(tensor, fn x ->
      Float.round(x, places)
    end)
  end
end
