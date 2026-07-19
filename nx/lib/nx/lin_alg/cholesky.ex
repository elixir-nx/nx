defmodule Nx.LinAlg.Cholesky do
  @moduledoc false
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
    n = Nx.axis_size(left, 0)
    idx = Nx.iota({n})
    left_mask = Nx.logical_and(idx >= left_start, idx < left_end)
    right_mask = Nx.logical_and(idx >= right_start, idx < right_end)

    Nx.dot(Nx.select(left_mask, left, 0), Nx.select(right_mask, right, 0))
  end

  defn cholesky_grad(l, _input, g) do
    matrix_shape = {Nx.axis_size(l, -2), Nx.axis_size(l, -1)}

    num =
      case Nx.rank(l) do
        n when n <= 2 ->
          g |> Nx.tril() |> Nx.dot([-2], l, [-2]) |> batch_transpose()

        _ ->
          ba = batch_axes(l)
          g |> Nx.tril() |> Nx.dot([-2], ba, l, [-2], ba) |> batch_transpose()
      end

    den =
      case Nx.rank(l) do
        n when n <= 2 ->
          Nx.eye(matrix_shape) |> Nx.add(1)

        _ ->
          Nx.eye(matrix_shape) |> Nx.add(1) |> Nx.broadcast(Nx.shape(l))
      end

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

  deftransformp batch_transpose(t) do
    rank = tuple_size(t.shape)

    if rank <= 2 do
      Nx.transpose(t)
    else
      axes = Enum.to_list(0..(rank - 3)) ++ [rank - 1, rank - 2]
      Nx.transpose(t, axes: axes)
    end
  end

  deftransformp batch_axes(t) do
    rank = tuple_size(t.shape)

    if rank <= 2 do
      []
    else
      Enum.to_list(0..(rank - 3)//1)
    end
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
