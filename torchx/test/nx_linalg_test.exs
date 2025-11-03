defmodule Torchx.NxLinAlgTest do
  use Torchx.Case, async: true

  describe "eig (Torchx default backend)" do
    test "computes eigenvalues and eigenvectors for diagonal matrix" do
      t = Nx.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
      assert {eigenvals, _eigenvecs} = Nx.LinAlg.eig(t)
      expected = Nx.tensor([3.0, 2.0, 1.0]) |> Nx.as_type({:c, 64})
      assert_all_close(Nx.abs(eigenvals), Nx.abs(expected), atol: 1.0e-2)
    end

    test "computes eigenvalues and eigenvectors for upper triangular matrix" do
      t = Nx.tensor([[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]])
      assert {eigenvals, eigenvecs} = Nx.LinAlg.eig(t)
      expected = Nx.tensor([6.0, 4.0, 1.0])
      assert_all_close(Nx.abs(eigenvals), Nx.abs(expected), atol: 1.0e-2)

      assert_all_close(Nx.dot(t, eigenvecs), Nx.dot(eigenvecs, Nx.make_diagonal(eigenvals)),
        atol: 1.0e-2
      )
    end

    test "computes eigenvalues and eigenvectors for lower triangular matrix" do
      t = Nx.tensor([[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]])
      assert {eigenvals, eigenvecs} = Nx.LinAlg.eig(t)
      expected = Nx.tensor([6.0, 3.0, 1.0])
      assert_all_close(Nx.abs(eigenvals), Nx.abs(expected), atol: 1.0e-2)

      assert_all_close(Nx.dot(t, eigenvecs), Nx.dot(eigenvecs, Nx.make_diagonal(eigenvals)),
        atol: 1.0e-2
      )
    end

    test "computes complex eigenvalues for rotation matrix" do
      t = Nx.tensor([[0.0, -1.0], [1.0, 0.0]])
      assert {eigenvals, _eigenvecs} = Nx.LinAlg.eig(t)
      assert_all_close(Nx.abs(eigenvals), Nx.tensor([1.0, 1.0]), atol: 1.0e-3)
      assert_all_close(Nx.sum(Nx.imag(eigenvals)), Nx.tensor(0.0), atol: 1.0e-3)
    end

    test "works with batched matrices" do
      t = Nx.tensor([[[1.0, 0.0], [0.0, 2.0]], [[3.0, 0.0], [0.0, 4.0]]])
      assert {eigenvals, _eigenvecs} = Nx.LinAlg.eig(t)
      expected = Nx.tensor([[2.0, 1.0], [4.0, 3.0]])
      assert_all_close(Nx.abs(eigenvals), expected, atol: 1.0e-3)
    end

    test "works with vectorized matrices" do
      t =
        Nx.tensor([
          [[[1.0, 0.0], [0.0, 2.0]]],
          [[[3.0, 0.0], [0.0, 4.0]]]
        ])
        |> Nx.vectorize(x: 2, y: 1)

      assert {eigenvals, eigenvecs} = Nx.LinAlg.eig(t)
      assert eigenvals.vectorized_axes == [x: 2, y: 1]
      assert eigenvecs.vectorized_axes == [x: 2, y: 1]

      eigenvals = Nx.devectorize(eigenvals)
      assert_all_close(Nx.abs(eigenvals[0][0]), Nx.tensor([2.0, 1.0]), atol: 1.0e-3)
      assert_all_close(Nx.abs(eigenvals[1][0]), Nx.tensor([4.0, 3.0]), atol: 1.0e-3)

      eigenvecs_dev = Nx.devectorize(eigenvecs)

      for batch <- 0..1, col <- 0..1 do
        v = eigenvecs_dev[batch][0][[.., col]]
        norm = Nx.LinAlg.norm(v) |> Nx.to_number()
        assert_in_delta(norm, 1.0, 0.1)
      end
    end

  @tag :skip
  test "property: eigenvalue equation A*v = λ*v" do
      key = Nx.Random.key(System.unique_integer())

      for _ <- 1..3, type <- [{:f, 32}, {:f, 64}], reduce: key do
        key ->
          {base_q, key} = Nx.Random.uniform(key, -2, 2, shape: {2, 3, 3}, type: type)
          {q, _} = Nx.LinAlg.qr(base_q)

          evals_test =
            [10, 1, 0.1]
            |> Enum.map(fn magnitude ->
              sign = if :rand.uniform() - 0.5 > 0, do: 1, else: -1
              rand = :rand.uniform() * magnitude * 0.1 + magnitude
              rand * sign
            end)
            |> Nx.tensor(type: type)

          evals_test_diag =
            evals_test
            |> Nx.make_diagonal()
            |> Nx.reshape({1, 3, 3})
            |> Nx.tile([2, 1, 1])

          q_adj = Nx.LinAlg.adjoint(q)

          a =
            q
            |> Nx.dot([2], [0], evals_test_diag, [1], [0])
            |> Nx.dot([2], [0], q_adj, [1], [0])

          assert {eigenvals, eigenvecs} = Nx.LinAlg.eig(a, balance: 0)

          evals =
            eigenvals
            |> Nx.vectorize(x: 2)
            |> Nx.make_diagonal()
            |> Nx.devectorize(keep_names: false)

          assert_all_close(
            Nx.dot(eigenvecs, [-1], [0], evals, [-2], [0]),
            Nx.dot(a, [-1], [0], eigenvecs, [-2], [0]),
            atol: 1.0e-3
          )

          key
      end
    end
  end
end
