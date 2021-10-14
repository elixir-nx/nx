defmodule Nx.LinAlgTest do
  use ExUnit.Case, async: true

  doctest Nx.LinAlg

  describe "triangular_solve" do
    test "property" do
      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [0, 1, 1]])
      b = Nx.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 4.0], [2.0, 0.0, 1.0]])
      assert Nx.dot(a, Nx.LinAlg.triangular_solve(a, b)) == b

      upper = Nx.transpose(a)
      assert Nx.dot(upper, Nx.LinAlg.triangular_solve(upper, b, lower: false)) == b

      assert Nx.dot(
               Nx.LinAlg.triangular_solve(upper, b, left_side: false, lower: false),
               upper
             ) == b

      assert Nx.LinAlg.triangular_solve(a, b, transform_a: :transpose) ==
               Nx.LinAlg.triangular_solve(upper, b, lower: false)

      assert Nx.dot(
               Nx.transpose(a),
               Nx.LinAlg.triangular_solve(a, b, transform_a: :transpose)
             ) == b
    end
  end

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

    test "property" do
      for _ <- 1..10 do
        square = Nx.random_uniform({4, 4})
        tall = Nx.random_uniform({4, 3})
        # Wide-matrix QR is not yet implemented

        assert {q, r} = Nx.LinAlg.qr(square)
        assert q |> Nx.dot(r) |> Nx.subtract(square) |> Nx.all_close?(1.0e-5)

        assert {q, r} = Nx.LinAlg.qr(tall)
        assert q |> Nx.dot(r) |> Nx.subtract(tall) |> Nx.all_close?(1.0e-5)
      end
    end
  end

  describe "eigh" do
    test "correctly a eigenvalues and eigenvectors" do
      t =
        Nx.tensor([
          [5, -1, 0, 1, 2],
          [-1, 5, 0, 5, 3],
          [0, 0, 4, 7, 2],
          [1, 5, 7, 0, 9],
          [2, 3, 2, 9, 2]
        ])

      assert {eigenvals, eigenvecs} = Nx.LinAlg.eigh(t)

      # Eigenvalues
      assert round(eigenvals, 3) ==
               Nx.tensor([16.394, -9.739, 5.901, 4.334, -0.892])

      # Eigenvectors
      assert round(eigenvecs, 3) ==
               Nx.tensor([
                 [0.112, -0.004, -0.828, 0.440, 0.328],
                 [0.395, 0.163, 0.533, 0.534, 0.497],
                 [0.427, 0.326, -0.137, -0.699, 0.452],
                 [0.603, -0.783, -0.008, -0.079, -0.130],
                 [0.534, 0.504, -0.103, 0.160, -0.651]
               ])
    end

    test "property" do
      for _ <- 1..10 do
        # Random symmetric matrix
        rm = Nx.random_uniform({3, 3})

        t =
          rm
          |> Nx.transpose()
          |> Nx.add(rm)

        # Eigenvalues and eigenvectors
        assert {eigenvals, eigenvecs} = Nx.LinAlg.eigh(t)

        # Eigenvalue equation
        evecs_evals = Nx.multiply(eigenvecs, eigenvals)
        t_evecs = Nx.dot(t, eigenvecs)
        al = Nx.all_close?(evecs_evals, t_evecs, atol: 1.0e-2)
        assert al == Nx.tensor(1, type: {:u, 8})
      end
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

    test "property" do
      for _ <- 1..10 do
        square = Nx.random_uniform({4, 4})

        assert {u, d, vt} = Nx.LinAlg.svd(square)
        m = u |> Nx.shape() |> elem(1)
        n = vt |> Nx.shape() |> elem(0)

        assert u
               |> Nx.dot(diag(d, m, n))
               |> Nx.dot(vt)
               |> Nx.subtract(square)
               |> Nx.all_close?(1.0e-5)

        tall = Nx.random_uniform({4, 3})

        assert {u, d, vt} = Nx.LinAlg.svd(tall)
        m = u |> Nx.shape() |> elem(1)
        n = vt |> Nx.shape() |> elem(0)

        assert u
               |> Nx.dot(diag(d, m, n))
               |> Nx.dot(vt)
               |> Nx.subtract(tall)
               |> Nx.all_close?(1.0e-5)

        # TODO: SVD does not work for wide matrices and
        # raises a non-semantic error

        #  wide = Nx.random_uniform({3, 4})

        # assert {u, d, vt} = Nx.LinAlg.svd(wide)
        # m = u |> Nx.shape() |> elem(1)
        # n = vt |> Nx.shape() |> elem(0)

        # assert u
        #        |> Nx.dot(diag(d, m, n))
        #        |> Nx.dot(vt)
        #        |> Nx.subtract(wide)
        #        |> Nx.all_close?(1.0e-5)
      end
    end
  end

  describe "lu" do
    test "property" do
      for _ <- 1..10 do
        # Generate random L and U matrices so we can construct
        # a factorizable A matrix:
        shape = {4, 4}
        lower_selector = Nx.iota(shape, axis: 0) |> Nx.greater_equal(Nx.iota(shape, axis: 1))
        upper_selector = Nx.transpose(lower_selector)

        l_prime =
          shape
          |> Nx.random_uniform()
          |> Nx.multiply(lower_selector)

        u_prime = shape |> Nx.random_uniform() |> Nx.multiply(upper_selector)

        a = Nx.dot(l_prime, u_prime)

        assert {p, l, u} = Nx.LinAlg.lu(a)
        assert p |> Nx.dot(l) |> Nx.dot(u) |> Nx.subtract(a) |> Nx.all_close?(1.0e-5)
      end
    end
  end

  describe "cholesky" do
    test "property" do
      for _ <- 1..10 do
        # Generate random L matrix so we can construct
        # a factorizable A matrix:
        shape = {4, 4}
        lower_selector = Nx.iota(shape, axis: 0) |> Nx.greater_equal(Nx.iota(shape, axis: 1))

        l_prime =
          shape
          |> Nx.random_uniform()
          |> Nx.multiply(lower_selector)

        a = Nx.dot(l_prime, Nx.transpose(l_prime))

        assert l = Nx.LinAlg.cholesky(a)
        assert l |> Nx.dot(Nx.transpose(l)) |> Nx.subtract(a) |> Nx.all_close?(1.0e-5)
      end
    end
  end

  defp round(tensor, places) do
    Nx.map(tensor, fn x ->
      Float.round(Nx.to_scalar(x), places)
    end)
  end

  defp diag(%Nx.Tensor{shape: {r}} = t, m, n) do
    base_result =
      t
      |> Nx.reshape({r, 1})
      |> Nx.tile([1, n])
      |> Nx.multiply(Nx.eye(n))

    if m > r do
      Nx.concatenate([base_result, Nx.broadcast(0, {m - r, n})])
    else
      base_result
    end
  end
end
