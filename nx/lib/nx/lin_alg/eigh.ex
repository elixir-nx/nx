defmodule Nx.LinAlg.Eigh do
  import Nx.Defn

  alias Nx.LinAlg.QR

  defn eigh(a, opts \\ []) do
    opts =
      keyword!(opts, eps: 1.0e-10, max_iter: 10_000)

    a
    |> Nx.revectorize([collapsed_axes: :auto],
      target_shape: {Nx.axis_size(a, -2), Nx.axis_size(a, -1)}
    )
    |> eigh_matrix(opts)
    |> revectorize_result(a)
  end

  deftransformp revectorize_result({eigenvals, eigenvecs}, a) do
    shape = Nx.shape(a)

    {
      Nx.revectorize(eigenvals, a.vectorized_axes,
        target_shape: Tuple.delete_at(shape, tuple_size(shape) - 1)
      ),
      Nx.revectorize(eigenvecs, a.vectorized_axes, target_shape: shape)
    }
  end

  defnp eigh_matrix(a, opts \\ []) do
    case Nx.shape(a) do
      {1, 1} ->
        {a, Nx.fill(a, 1)}

      {_, _} ->
        eigh_2d(a, opts)
    end
  end

  defnp eigh_2d(a, opts \\ []) do
    # The input Hermitian matrix A reduced to Hessenberg matrix H by Householder transform.
    # Then, by using QR iteration it converges to AQ = QΛ,
    # where Λ is the diagonal matrix of eigenvalues and the columns of Q are the eigenvectors.

    eps = opts[:eps]
    max_iter = opts[:max_iter]

    {h, q_h} = hessenberg_decomposition(a, eps)

    [_, zero] = Nx.broadcast_vectors([a, Nx.u8(0)])

    {{eigenvals_diag, eigenvecs}, _} =
      while {{a = h, q = q_h}, {has_converged = zero, iter = 0}},
            iter < max_iter and not has_converged do
        {q_next, r} = QR.qr(a, eps: eps)

        a_next = Nx.dot(r, q_next)
        q_next = Nx.dot(q, q_next)

        has_converged = Nx.all_close(q, q_next, atol: eps)
        {{a_next, q_next}, {has_converged, iter + 1}}
      end

    eigenvals = eigenvals_diag |> Nx.take_diagonal() |> Nx.real() |> approximate_zeros(eps)
    eigenvecs = approximate_zeros(eigenvecs, eps)

    {eigenvals, eigenvecs}
  end

  defnp hessenberg_decomposition(matrix, eps) do
    # The input Hermitian matrix A reduced to Hessenberg matrix H by Householder transform.
    # Then, by using QR iteration it converges to AQ = QΛ,
    # where Λ is the diagonal matrix of eigenvalues and the columns of Q are the eigenvectors.

    {n, _} = Nx.shape(matrix)
    column_iota = Nx.iota({Nx.axis_size(matrix, 0)}, vectorized_axes: matrix.vectorized_axes)

    out_type = Nx.Type.to_floating(Nx.type(matrix))

    eye = Nx.eye({n, n}, vectorized_axes: matrix.vectorized_axes, type: out_type)

    {{hess, q}, _} =
      while {{hess = Nx.as_type(matrix, out_type), q = eye}, {eps, column_iota}},
            i <- 0..(n - 2)//1 do
        x = hess[[.., i]]
        x = Nx.select(column_iota <= i, 0, x)
        h = QR.householder_reflector(x, i, eps)

        q = Nx.dot(q, h)

        h_adj = Nx.LinAlg.adjoint(h)

        hess = h |> Nx.dot(hess) |> Nx.dot(h_adj)

        {{hess, q}, {eps, column_iota}}
      end

    {approximate_zeros(hess, eps), approximate_zeros(q, eps)}
  end

  defnp approximate_zeros(matrix, eps), do: Nx.select(Nx.abs(matrix) <= eps, 0, matrix)
end
