defmodule Nx.LinAlg.LU do
  @moduledoc false
  import Nx.Defn

  defn lu(a, _opts \\ []) do
    vectorized_axes = a.vectorized_axes

    result =
      a
      |> Nx.revectorize([collapsed_axes: :auto],
        target_shape: {Nx.axis_size(a, -2), Nx.axis_size(a, -1)}
      )
      |> lu_matrix()
      |> revectorize_result(a.shape, vectorized_axes)

    custom_grad(result, [a], fn g ->
      lu_grad(result, g)
    end)
  end

  defnp lu_matrix(a) do
    {m, n} = Nx.shape(a)
    input_type = a.type
    type = Nx.Type.to_floating(a.type)

    # Initialize state
    pivot = Nx.iota({m}, vectorized_axes: a.vectorized_axes)
    perm = Nx.iota({m}, vectorized_axes: a.vectorized_axes)
    k = Nx.tensor(0)

    # Create index arrays once
    m_idx = Nx.iota({m}, vectorized_axes: a.vectorized_axes)
    n_idx = Nx.iota({n}, vectorized_axes: a.vectorized_axes)

    # Main decomposition loop - using the EXACT same logic as the working debug
    a = Nx.as_type(a, type)

    {_k, _pivot, perm, a, _m_idx, _n_idx} =
      while {k, pivot, perm, a, m_idx, n_idx}, Nx.less(k, m) do
        # STEP 1: Find pivot
        col_k = a[[.., k]]
        masked_magnitude = Nx.select(m_idx >= k, Nx.abs(col_k), :neg_infinity)
        pivot_row = Nx.argmax(masked_magnitude)

        # STEP 2: Record pivot and perform row swaps
        pivot = Nx.indexed_put(pivot, Nx.reshape(k, {1}), Nx.reshape(pivot_row, {}))

        # Update matrix and permutation
        {a, perm} =
          if k != pivot_row do
            {swap_rows(a, k, pivot_row), swap_elements(perm, k, pivot_row)}
          else
            {a, perm}
          end

        # STEP 3: Scale column below diagonal
        diagonal = a[[k, k]]
        col_k_new = a[[.., k]]
        scale_mask = m_idx > k

        scaled_col =
          if diagonal == 0 do
            col_k_new
          else
            Nx.select(scale_mask, col_k_new / diagonal, col_k_new)
          end

        a = Nx.put_slice(a, [0, k], Nx.reshape(scaled_col, {m, 1}))

        has_trailing = k < m - 1 and k < n - 1

        a =
          cond do
            has_trailing ->
              # Get L column and U row after scaling
              l_col = a[[.., k]]
              u_row = a[k]

              # Create outer product
              outer = Nx.outer(l_col, Nx.LinAlg.adjoint(u_row))

              # Create mask for trailing submatrix
              trailing_mask = Nx.reshape(m_idx > k, {m, 1}) and Nx.reshape(n_idx > k, {1, n})

              # Apply masked update
              masked_update = Nx.select(trailing_mask, outer, 0)
              a - masked_update

            true ->
              a
          end

        # Increment counter
        k = Nx.add(k, 1)
        {k, pivot, perm, a, m_idx, n_idx}
      end

    l = Nx.tril(a, k: -1) + Nx.eye(m)
    u = Nx.triu(a)

    p = Nx.equal(Nx.iota({m, 1}), Nx.reshape(perm, {1, m})) |> Nx.as_type(input_type)

    {p, l, u}
  end

  defnp swap_rows(matrix, i, j) do
    vectorized_axes = matrix.vectorized_axes
    [i, j, _] = Nx.broadcast_vectors([i, j, matrix])

    row_i = matrix[i] |> Nx.devectorize()
    row_j = matrix[j] |> Nx.devectorize()
    i = Nx.devectorize(i)
    j = Nx.devectorize(j)

    matrix = Nx.devectorize(matrix)

    # matrix is {k, m, n}
    # row_i is {k, n}
    # row_j is {k, n}

    {max_k, m, n} = Nx.shape(matrix)

    {matrix, _} =
      while {matrix, {row_i, row_j, i, j, k = 0}}, k < max_k do
        matrix =
          matrix
          |> Nx.put_slice([k, i[k], 0], Nx.reshape(row_j[k], {1, 1, n}))
          |> Nx.put_slice([k, j[k], 0], Nx.reshape(row_i[k], {1, 1, n}))

        {matrix, {row_i, row_j, i, j, k + 1}}
      end

    Nx.revectorize(matrix, vectorized_axes, target_shape: {m, n})
  end

  defnp swap_elements(vector, i, j) do
    elem_i = vector[i]
    elem_j = vector[j]

    vector
    |> Nx.indexed_put(Nx.reshape(i, {1}), Nx.reshape(elem_j, {}))
    |> Nx.indexed_put(Nx.reshape(j, {1}), Nx.reshape(elem_i, {}))
  end

  deftransformp revectorize_result({p, l, u}, shape, vectorized_axes) do
    {p_shape, l_shape, u_shape} = Nx.Shape.lu(shape)

    {
      Nx.revectorize(p, vectorized_axes, target_shape: p_shape),
      Nx.revectorize(l, vectorized_axes, target_shape: l_shape),
      Nx.revectorize(u, vectorized_axes, target_shape: u_shape)
    }
  end

  defn lu_grad({p, l, u}, {_dp, dl, du}) do
    # Definition taken from https://arxiv.org/pdf/2009.10071.pdf
    # Equation (3)

    u_h = Nx.LinAlg.adjoint(u)
    l_h = Nx.LinAlg.adjoint(l)
    p_t = Nx.LinAlg.adjoint(p)

    lh_dl = Nx.dot(l_h, dl)
    du_uh = Nx.dot(du, u_h)

    lt_inv = Nx.LinAlg.invert(l_h)
    ut_inv = Nx.LinAlg.invert(u_h)

    df = lh_dl |> Nx.tril(k: -1) |> Nx.add(Nx.triu(du_uh))
    da = p_t |> Nx.dot(lt_inv) |> Nx.dot(df) |> Nx.dot(ut_inv)

    [da]
  end
end
