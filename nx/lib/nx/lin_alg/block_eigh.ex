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

  defn eigh(matrix, opts \\ []) do
    opts = keyword!(opts, eps: 1.0e-6, max_iter: 100)

    matrix
    |> Nx.revectorize([collapsed_axes: :auto],
      target_shape: {Nx.axis_size(matrix, -2), Nx.axis_size(matrix, -1)}
    )
    |> decompose(opts)
    |> revectorize_result(matrix)
  end

  defnp decompose(matrix, opts) do
    {n, _} = Nx.shape(matrix)

    if n > 1 do
      m_decompose(matrix, opts)
    else
      {Nx.take_diagonal(Nx.real(matrix)), Nx.tensor([1], type: matrix.type)}
    end
  end

  defnp m_decompose(matrix, opts) do
    eps = opts[:eps]
    max_iter = opts[:max_iter]

    type = Nx.Type.to_floating(Nx.type(matrix))
    matrix = Nx.as_type(matrix, type)
    {n, _} = Nx.shape(matrix)
    i_n = n - 1
    mid = calculate_mid(i_n)
    i_mid = mid - 1

    tl = matrix[[0..i_mid, 0..i_mid]]
    tr = matrix[[0..i_mid, mid..i_n]]
    bl = matrix[[mid..i_n, 0..i_mid]]
    br = matrix[[mid..i_n, mid..i_n]]

    # Pad if not even
    {tr, bl, br} =
      if Nx.remainder(n, 2) == 1 do
        tr = Nx.pad(tr, 0, [{0, 0, 0}, {0, 1, 0}])
        bl = Nx.pad(bl, 0, [{0, 1, 0}, {0, 0, 0}])
        br = Nx.pad(br, 0, [{0, 1, 0}, {0, 1, 0}])
        {tr, bl, br}
      else
        {tr, bl, br}
      end

    # Initialze tensors to hold eigenvectors
    v_tl = v_br = Nx.eye(mid, type: type)
    v_tr = v_bl = Nx.broadcast(Nx.tensor(0, type: type), {mid, mid})

    {frob_norm, off_norm} = norms(tl, tr, bl, br)

    # Nested loop
    # Outside loop performs the "sweep" operation until the norms converge
    # or max iterations are hit. The Brent/Luk paper states that Log2(n) is
    # a good estimate for convergence, but XLA chose a static number which wouldn't
    # be reached until a matrix roughly greater than 20kx20k.
    #
    # The inner loop performs "sweep" rounds of n - 1, which is enough permutations to allow
    # all sub matrices to share the needed values.
    {{tl, br, v_tl, v_tr, v_bl, v_br}, _} =
      while {{tl, br, v_tl, v_tr, v_bl, v_br}, {frob_norm, off_norm, tr, bl, i = 0}},
            off_norm > eps ** 2 * frob_norm and i < max_iter do
        {tl, tr, bl, br, v_tl, v_tr, v_bl, v_br} =
          perform_sweeps(tl, tr, bl, br, v_tl, v_tr, v_bl, v_br, mid, i_n)

        {frob_norm, off_norm} = norms(tl, tr, bl, br)

        {{tl, br, v_tl, v_tr, v_bl, v_br}, {frob_norm, off_norm, tr, bl, i + 1}}
      end

    # Recombine
    w = Nx.concatenate([Nx.take_diagonal(tl), Nx.take_diagonal(br)])

    v =
      Nx.concatenate([
        Nx.concatenate([v_tl, v_tr], axis: 1),
        Nx.concatenate([v_bl, v_br], axis: 1)
      ])
      |> Nx.LinAlg.adjoint()

    # trim padding
    {w, v} =
      if Nx.remainder(n, 2) == 1 do
        {w[0..i_n], v[[0..i_n, 0..i_n]]}
      else
        {w, v}
      end

    sort_ind = Nx.argsort(Nx.abs(w), direction: :desc)

    w = Nx.take(w, sort_ind) |> approximate_zeros(eps)
    v = Nx.take(v, sort_ind, axis: 1) |> approximate_zeros(eps)

    {w, v}
  end

  deftransformp calculate_mid(i_n) do
    Range.size(0..i_n//2)
  end

  defnp calc_rot(tl, tr, br) do
    complex? = tl |> Nx.type() |> Nx.Type.complex?()
    br = Nx.take_diagonal(br) |> Nx.real()
    tr = Nx.take_diagonal(tr)
    tl = Nx.take_diagonal(tl) |> Nx.real()

    {tr, w} =
      if complex? do
        abs_tr = Nx.abs(tr)
        {abs_tr, Nx.select(abs_tr == 0, 1, Nx.conjugate(tr) / abs_tr)}
      else
        {tr, 1}
      end

    z_tr = Nx.equal(tr, 0)
    s_tr = Nx.select(z_tr, 1, tr)
    tau = Nx.select(z_tr, 0, (br - tl) / (2 * s_tr))

    t = Nx.sqrt(1 + tau ** 2)

    t = 1 / (tau + Nx.select(tau >= 0, t, -t))

    pred = Nx.abs(tr) <= 1.0e-5 * Nx.min(Nx.abs(br), Nx.abs(tl))
    t = Nx.select(pred, Nx.tensor(0, type: tl.type), t)

    c = 1.0 / Nx.sqrt(1.0 + t ** 2)
    s = if complex?, do: Nx.complex(t * c, 0) * w, else: t * c

    rt1 = tl - t * tr
    rt2 = br + t * tr
    {rt1, rt2, c, s}
  end

  defnp sq_norm(tl, tr, bl, br) do
    Nx.sum(Nx.abs(tl) ** 2 + Nx.abs(tr) ** 2 + Nx.abs(bl) ** 2 + Nx.abs(br) ** 2)
  end

  defnp off_norm(tl, tr, bl, br) do
    {n, _} = Nx.shape(tl)
    diag = Nx.broadcast(0, {n})
    o_tl = Nx.put_diagonal(tl, diag)
    o_br = Nx.put_diagonal(br, diag)

    sq_norm(o_tl, tr, bl, o_br)
  end

  # Calculates the Frobenius norm and the norm of the off-diagonals from
  # the submatrices. Used to calculate convergeance.
  defnp norms(tl, tr, bl, br) do
    frob = sq_norm(tl, tr, bl, br)
    off = off_norm(tl, tr, bl, br)

    {frob, off}
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

  defnp perform_sweeps(tl, tr, bl, br, v_tl, v_tr, v_bl, v_br, mid, i_n) do
    while {tl, tr, bl, br, v_tl, v_tr, v_bl, v_br}, _n <- 0..i_n do
      {rt1, rt2, c, s} = calc_rot(tl, tr, br)
      # build row and column vectors for parrelelized rotations
      c_v = Nx.new_axis(c, 1)
      s_v = Nx.new_axis(s, 1)
      c_h = Nx.new_axis(c, 0)
      s_h = Nx.new_axis(s, 0)

      s_v_conj =
        if Nx.type(s) |> Nx.Type.complex?() do
          Nx.conjugate(s_v)
        else
          s_v
        end

      s_h_conj = Nx.transpose(s_v_conj)

      # Each rotation group below is performed based on the same
      # tl, bl, tr, br values, so we must do single-expr
      # assignments (i.e. {tl, tr, bl, br} = ...)

      # Rotate rows
      {tl, tr, bl, br} = {
        tl * c_v - bl * s_v_conj,
        tr * c_v - br * s_v_conj,
        tl * s_v + bl * c_v,
        tr * s_v + br * c_v
      }

      # Rotate cols
      {tl, tr, bl, br} = {
        tl * c_h - tr * s_h,
        tl * s_h_conj + tr * c_h,
        bl * c_h - br * s_h,
        bl * s_h_conj + br * c_h
      }

      # Store results and permute values across sub matrices
      zero_diag = Nx.broadcast(0, {mid})
      tl = Nx.put_diagonal(tl, rt1)
      tr = Nx.put_diagonal(tr, zero_diag)
      bl = Nx.put_diagonal(bl, zero_diag)
      br = Nx.put_diagonal(br, rt2)

      {tl, tr} = permute_cols_in_row(tl, tr)
      {bl, br} = permute_cols_in_row(bl, br)
      {tl, bl} = permute_rows_in_col(tl, bl)
      {tr, br} = permute_rows_in_col(tr, br)

      # Rotate to calc vectors
      {v_tl, v_tr, v_bl, v_br} = {
        v_tl * c_v - v_bl * s_v_conj,
        v_tr * c_v - v_br * s_v_conj,
        v_tl * s_v + v_bl * c_v,
        v_tr * s_v + v_br * c_v
      }

      # permute for vectors
      {v_tl, v_bl} = permute_rows_in_col(v_tl, v_bl)
      {v_tr, v_br} = permute_rows_in_col(v_tr, v_br)

      {tl, tr, bl, br, v_tl, v_tr, v_bl, v_br}
    end
  end

  defnp approximate_zeros(matrix, eps), do: Nx.select(Nx.abs(matrix) <= eps, 0, matrix)

  # https://github.com/openxla/xla/blob/main/xla/hlo/transforms/expanders/eigh_expander.cc#L200-L239
  defnp permute_rows_in_col(top, bottom) do
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
               bottom[1..-1//1],
               top[(k - 1)..(k - 1)]
             ],
             axis: 0
           )}
      end

    {top_out, bottom_out}
  end

  defnp permute_cols_in_row(left, right) do
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
