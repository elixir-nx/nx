defmodule Nx.LinAlg.Cholesky do
  import Nx.Defn

  defn cholesky(a, opts \\ []) do
    opts =
      keyword!(opts, eps: 1.0e-10)

    vectorized_axes = a.vectorized_axes

    result =
      a
      |> Nx.revectorize([collapsed_axes: :auto],
        target_shape: {Nx.axis_size(a, -2), Nx.axis_size(a, -1)}
      )
      |> cholesky_matrix(opts)
      |> Nx.revectorize(vectorized_axes, target_shape: a.shape)

    custom_grad(result, [a], fn g ->
      cholesky_grad(result, a, g)
    end)
  end

  defn cholesky_matrix(a, opts \\ []) do
    # From wikipedia (https://en.wikipedia.org/wiki/Cholesky_decomposition#The_Cholesky%E2%80%93Banachiewicz_and_Cholesky%E2%80%93Crout_algorithms)
    # adapted to 0-indexing
    # Ljj := sqrt(Ajj - sum(k=0..j-1)[Complex.abs_squared(Ljk)])
    # Lij := 1/Ljj * (Aij - sum(k=0..j-1)[Lik * Complex.conjugate(Ljk)])

    eps = opts[:eps]
    n = Nx.axis_size(a, 0)

    {l, _} =
      while {l = Nx.multiply(0.0, a), {a}}, i <- 0..(n - 1) do
        {l, _} =
          while {l, {a, i, j = 0}}, j <= i do
            value =
              if i == j do
                row = l[i]

                sum = dot_with_dynamic_slice(row, 0, j, conjugate_if_complex(row), 0, j)
                Nx.sqrt(a[[i, i]] - sum)
              else
                sum = dot_with_dynamic_slice(l[i], 0, j, conjugate_if_complex(l[j]), 0, j)

                (a[[i, j]] - sum) / (l[[j, j]] + eps)
              end

            l = Nx.indexed_put(l, Nx.stack([i, j]), value)
            {l, {a, i, j + 1}}
          end

        {l, {a}}
      end

    approximate_zeros(l, eps)
  end

  defnp approximate_zeros(matrix, eps), do: Nx.select(Nx.abs(matrix) <= eps, 0, matrix)

  defnp dot_with_dynamic_slice(left, left_start, left_end, right, right_start, right_end) do
    lhs = zero_out_dynamic_slice(left, left_start, left_end)
    rhs = zero_out_dynamic_slice(right, right_start, right_end)

    Nx.dot(lhs, rhs)
  end

  defnp zero_out_dynamic_slice(t, start_idx, end_idx) do
    # assumes t has rank 1
    zero_out_selector = Nx.logical_or(Nx.iota(t.shape) < start_idx, Nx.iota(t.shape) >= end_idx)

    Nx.take(
      Nx.select(zero_out_selector, 0, t),
      Nx.argsort(zero_out_selector, direction: :asc, stable: true)
    )
  end

  defn cholesky_grad(l, _input, g) do
    num = g |> Nx.tril() |> Nx.dot([0], l, [0]) |> Nx.transpose()
    den = l |> Nx.shape() |> Nx.eye() |> Nx.add(1)
    phi_tril = num |> Nx.divide(den) |> Nx.tril()

    bm = Nx.LinAlg.triangular_solve(l, phi_tril, transform_a: :transpose)

    dl =
      l
      |> conjugate_if_complex()
      |> Nx.LinAlg.triangular_solve(bm, left_side: false)

    # If we end up supporting the "make_symmetric" option for Nx.LinAlg.cholesky
    # we need to apply: dl := (adjoint(dl) + dl)/2 when the option is true.
    # If the option is applied as Nx.add(tensor, adjoint(tensor)) |> Nx.divide(2)
    # on the expression, no modifications are needed here, because
    # the grad for the transformation is actually the same transformation
    # applied on the grad

    [dl]
  end

  defnp conjugate_if_complex(x) do
    case Nx.type(x) do
      {:c, _} ->
        Nx.conjugate(x)

      _ ->
        x
    end
  end
end
