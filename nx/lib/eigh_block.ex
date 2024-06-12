defmodule Nx.LinAlg.BlockEigh do
  @moduledoc """
  Parallel Jacobi symmetric eigendecomposition.

  Reference implementation taking from XLA's eigh_expander
  which is built on the approach in:
  Brent, R. P., & Luk, F. T. (1985). The solution of singular-value
  and symmetric eigenvalue problems on multiprocessor arrays.
  SIAM Journal on Computing, 6(1), 69-84. https://doi.org/10.1137/0906007
  """
  require Nx

  import Nx.Defn

  defn calc_rot(tl, tr, br) do
    a = Nx.take_diagonal(br)
    b = Nx.take_diagonal(tr)
    c = Nx.take_diagonal(tl)

    tau = (a - c) / (2 * b)
    t = Nx.sqrt(1 + Nx.pow(tau, 2))
    t = Nx.select(Nx.greater_equal(tau, 0), 1 / (tau + t), 1 / (tau - t))

    pred = Nx.less_equal(Nx.abs(b), 0.1 * 1.0e-4 * Nx.min(Nx.abs(a), Nx.abs(c)))
    t = Nx.select(pred, 0.0, t)

    c = 1.0 / Nx.sqrt(1.0 + Nx.pow(t, 2))
    s = t * c

    rt1 = tl - t * tr
    rt2 = br + t * tr

    {rt1, rt2, c, s}
  end

  defn sq_norm(tl, tr, bl, br) do
    Nx.sum(Nx.pow(tl, 2) + Nx.pow(tr, 2) + Nx.pow(bl, 2) + Nx.pow(br, 2))
  end

  defn off_norm(tl, tr, bl, br) do
    {n, _} = Nx.shape(tl)
    diag = Nx.broadcast(0, {n})
    o_tl = Nx.put_diagonal(tl, diag)
    o_br = Nx.put_diagonal(br, diag)

    Nx.sum(Nx.pow(o_tl, 2) + Nx.pow(tr, 2) + Nx.pow(bl, 2) + Nx.pow(o_br, 2))
  end

  @doc """
  Calculates the Frobenius norm and the norm of the off-diagonals from
  the submatrices. Used to calculate convergeance.
  """
  defn norms(tl, tr, bl, br) do
    frob = sq_norm(tl, tr, bl, br)
    off = off_norm(tl, tr, bl, br)
    {frob, off}
  end

  defn eigh(matrix) do
    matrix
    |> Nx.revectorize([collapsed_axes: :auto],
      target_shape: {Nx.axis_size(matrix, -2), Nx.axis_size(matrix, -1)}
    )
    |> decompose()
    |> then(fn {w, v} ->
      revectorize_result({w, v}, matrix)
    end)
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

  defn decompose(matrix) do
    {n, _} = Nx.shape(matrix)

    if n > 1 do
      m_decompose(matrix)
    else
      {Nx.tensor([1], type: matrix.type), Nx.take_diagonal(matrix)}
    end
  end

  defn m_decompose(matrix) do
    {n, _} = Nx.shape(matrix)
    i_n = n - 1
    {mid, _} = Nx.shape(matrix[[0..i_n//2, 0..i_n//2]])
    i_mid = mid - 1

    {tl, tr, bl, br} =
      {matrix[[0..i_mid, 0..i_mid]], matrix[[0..i_mid, mid..i_n]], matrix[[mid..i_n, 0..i_mid]],
       matrix[[mid..i_n, mid..i_n]]}

    # Pad if not even
    {tl, tr, bl, br} =
      if Nx.remainder(n, 2) == 1 do
        tr = Nx.pad(tr, 0, [{0, 0, 0}, {0, 1, 0}])
        bl = Nx.pad(bl, 0, [{0, 1, 0}, {0, 0, 0}])
        br = Nx.pad(br, 0, [{0, 1, 0}, {0, 1, 0}])
        {tl, tr, bl, br}
      else
        {tl, tr, bl, br}
      end

    # Initialze tensors to hold eigenvectors
    v_tl = Nx.eye(mid, type: :f32)
    v_tr = Nx.broadcast(0.0, {mid, mid})
    v_bl = Nx.broadcast(0.0, {mid, mid})
    v_br = Nx.eye(mid, type: :f32)

    {frob_norm, off_norm} = norms(tl, tr, bl, br)

    # Nested loop
    # Outside loop performs the "sweep" operation until the norms converge
    # or max iterations are hit. The Brent/Luk paper states that Log2(n) is
    # a good estimate for convergence, but XLA chose a static number which wouldn't
    # be reached until a matrix roughly greater than 20kx20k.
    #
    # The inner loop performs "sweep" rounds of n - 1, which is enough permutations to allow
    # all sub matrices to share the needed values.
    {_, _, tl, _tr, _bl, br, v_tl, v_tr, v_bl, v_br, _} =
      while {frob_norm, off_norm, tl, tr, bl, br, v_tl, v_tr, v_bl, v_br, i = 0},
            off_norm > Nx.pow(1.0e-10, 2) * frob_norm and i < 15 do
        {tl, tr, bl, br, v_tl, v_tr, v_bl, v_br} =
          while {tl, tr, bl, br, v_tl, v_tr, v_bl, v_br}, _n <- 0..i_n do
            {rt1, rt2, c, s} = calc_rot(tl, tr, br)
            # build row and column vectors for parrelelized rotations
            c_v = Nx.reshape(c, {mid, 1})
            s_v = Nx.reshape(s, {mid, 1})
            c_h = Nx.reshape(c, {1, mid})
            s_h = Nx.reshape(s, {1, mid})

            # Rotate rows
            {tl, tr, bl, br} = {
              tl * c_v - bl * s_v,
              tr * c_v - br * s_v,
              tl * s_v + bl * c_v,
              tr * s_v + br * c_v
            }

            # Rotate cols
            {tl, tr, bl, br} = {
              tl * c_h - tr * s_h,
              tl * s_h + tr * c_h,
              bl * c_h - br * s_h,
              bl * s_h + br * c_h
            }

            # Store results and permute values across sub matrices
            tl = Nx.put_diagonal(tl, Nx.take_diagonal(rt1))
            tr = Nx.put_diagonal(tr, Nx.broadcast(0, {mid}))
            bl = Nx.put_diagonal(bl, Nx.broadcast(0, {mid}))
            br = Nx.put_diagonal(br, Nx.take_diagonal(rt2))

            {tl, tr} = permute_cols_in_row(tl, tr)
            {bl, br} = permute_cols_in_row(bl, br)
            {tl, bl} = permute_rows_in_col(tl, bl)
            {tr, br} = permute_rows_in_col(tr, br)

            # Rotate to calc vectors
            {v_tl, v_tr, v_bl, v_br} = {
              v_tl * c_v - v_bl * s_v,
              v_tr * c_v - v_br * s_v,
              v_tl * s_v + v_bl * c_v,
              v_tr * s_v + v_br * c_v
            }

            # permute for vectors
            {v_tl, v_bl} = permute_rows_in_col(v_tl, v_bl)
            {v_tr, v_br} = permute_rows_in_col(v_tr, v_br)

            {tl, tr, bl, br, v_tl, v_tr, v_bl, v_br}
          end

        {frob_norm, off_norm} = norms(tl, tr, bl, br)

        {frob_norm, off_norm, tl, tr, bl, br, v_tl, v_tr, v_bl, v_br, i + 1}
      end

    w = Nx.concatenate([Nx.take_diagonal(tl), Nx.take_diagonal(br)])

    v =
      Nx.concatenate([
        Nx.concatenate([v_tl, v_tr], axis: 1),
        Nx.concatenate([v_bl, v_br], axis: 1)
      ])

    # trim padding
    if Nx.remainder(n, 2) == 1 do
      {w[0..i_n], Nx.transpose(v[[0..i_n, 0..i_n]])}
    else
      {w, v}
    end
  end

  defn permute_rows_in_col(top, bottom) do
    {k, _} = Nx.shape(top)

    {top_out, bottom_out} =
      cond do
        k == 2 ->
          {Nx.concatenate([top[0..0], bottom[0..0]], axis: 0),
           Nx.concatenate(
             [
               bottom[1..-1//1],
               top[(k - 1)..(k - 1)]
             ],
             axis: 0
           )}

        k == 1 ->
          {top, bottom}

        true ->
          {Nx.concatenate([top[0..0], bottom[0..0], top[1..(k - 2)]], axis: 0),
           Nx.concatenate(
             [
               bottom[1..-1],
               top[(k - 1)..(k - 1)]
             ],
             axis: 0
           )}
      end

    {top_out, bottom_out}
  end

  defn permute_cols_in_row(left, right) do
    {k, _} = Nx.shape(left)

    {left_out, right_out} =
      cond do
        k == 2 ->
          {Nx.concatenate([left[[.., 0..0]], right[[.., 0..0]]], axis: 1),
           Nx.concatenate([right[[.., 1..(k - 1)]], left[[.., (k - 1)..(k - 1)]]], axis: 1)}

        k == 1 ->
          {left, right}

        true ->
          {Nx.concatenate([left[[.., 0..0]], right[[.., 0..0]], left[[.., 1..(k - 2)]]], axis: 1),
           Nx.concatenate([right[[.., 1..(k - 1)]], left[[.., (k - 1)..(k - 1)]]], axis: 1)}
      end

    {left_out, right_out}
  end
end
