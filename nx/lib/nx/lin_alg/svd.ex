defmodule Nx.LinAlg.SVD do
  @moduledoc false

  # Implementation of Singular Value Decomposition
  # Inspired by the Jax code in https://github.com/google/jax/blob/ba557d5e1beb480851117a003ebf76c0ed2249e0/jax/_src/lax/svd.py
  # QDWH is QR-based Dynamically Weighted Halley iteration
  #
  # References (same as Jax, above):
  #
  # * Nakatsukasa, Yuji, and Nicholas J. Higham.
  #   "Stable and efficient spectral divide and conquer algorithms for the symmetric
  #   eigenvalue decomposition and the SVD." SIAM Journal on Scientific Computing 35,
  #   no. 3 (2013): A1325-A1349.
  #   https://epubs.siam.org/doi/abs/10.1137/120876605
  #
  # * Nakatsukasa, Yuji, Zhaojun Bai, and François Gygi.
  #   "Optimizing Halley's iteration for computing the matrix polar decomposition."
  #   SIAM Journal on Matrix Analysis and Applications 31, no. 5 (2010): 2700-2720.
  #   https://epubs.siam.org/doi/abs/10.1137/090774999

  import Nx.Defn
  @eps 1.1920929e-07

  defn svd(input_tensor, opts \\ []) do
    validate_opts(opts)

    {target_shape, u_shape, s_shape, vt_shape} = calculate_shapes(input_tensor)

    tensor = Nx.revectorize(input_tensor, [vector: :auto], target_shape: target_shape)

    {u, s, vt} =
      if Nx.all(tensor == 0) do
        svd_all_zeros(tensor, opts)
      else
        svd_non_zero(tensor, opts)
      end

    # we can force [] as the vectorized axes because we are guaranteed that the input is devectorized
    result = {
      Nx.revectorize(u, [], target_shape: u_shape),
      Nx.revectorize(s, [], target_shape: s_shape),
      Nx.revectorize(vt, [], target_shape: vt_shape)
    }

    custom_grad(result, [input_tensor], fn g ->
      svd_grad(result, input_tensor, g)
    end)
  end

  deftransformp validate_opts(opts \\ []) do
    opts[:max_iter] || raise ArgumentError, "missing option :max_iter"
  end

  deftransformp calculate_shapes(t) do
    shape = Nx.shape(t)
    rank = tuple_size(shape)
    m = elem(shape, rank - 2)
    n = elem(shape, rank - 1)

    collapsed_axes = shape |> Tuple.delete_at(rank - 2) |> Tuple.delete_at(rank - 2)

    u_shape = collapsed_axes |> Nx.Shared.tuple_append(m) |> Nx.Shared.tuple_append(:auto)
    s_shape = Nx.Shared.tuple_append(collapsed_axes, :auto)
    vt_shape = Nx.Shared.tuple_append(s_shape, n)

    {{m, n}, u_shape, s_shape, vt_shape}
  end

  defnp svd_all_zeros(a, opts) do
    {m, n} = Nx.shape(a)

    k =
      case {m, n} do
        {m, n} when m > n -> n
        _ -> m
      end

    min_shape = Kernel.min(m, n)
    {u_cols, v_rows} = if opts[:full_matrices?], do: {m, n}, else: {min_shape, min_shape}

    s = Nx.broadcast(Nx.tensor(0, type: Nx.type(a)), {k})

    [s, _] = Nx.broadcast_vectors([s, a])

    u = Nx.eye({m, u_cols}, vectorized_axes: a.vectorized_axes, type: Nx.type(a))
    v = Nx.eye({v_rows, n}, vectorized_axes: a.vectorized_axes, type: Nx.type(a))

    {u, s, v}
  end

  defn svd_full(tensor, opts \\ []) do
    {reduce_to_square, q, u_null, a} =
      case Nx.shape(tensor) do
        {m, n} when m > n ->
          {q_full, a_full} = Nx.LinAlg.qr(tensor, mode: :complete)
          q = q_full[[0..-1//1, 0..(n - 1)//1]]
          u_null = q_full[[0..-1//1, n..-1//1]]
          a = a_full[0..(n - 1)//1]
          {true, q, u_null, a}

        {n, n} ->
          {false, tensor, tensor, tensor}
      end

    {u, s, v} = svd_tall_and_square(a, opts)

    u =
      if reduce_to_square do
        u = Nx.dot(q, u)
        Nx.concatenate([u, u_null], axis: -1)
      else
        u
      end

    {u, s, v}
  end

  defn svd_non_full(tensor, opts \\ []) do
    {m, n} = Nx.shape(tensor)

    # The constant `1.15` comes from Yuji Nakatsukasa's implementation
    # https://www.mathworks.com/matlabcentral/fileexchange/36830-symmetric-eigenvalue-decomposition-and-the-svd?s_tid=FX_rc3_behav
    {reduce_to_square, q, a} =
      if m > 1.15 * n do
        {q, a} = Nx.LinAlg.qr(tensor, mode: :reduced)
        {true, q, a}
      else
        {false, tensor, tensor}
      end

    {u, s, v} = svd_tall_and_square(a, opts)

    u =
      if reduce_to_square do
        Nx.dot(q, u)
      else
        u
      end

    {u, s, v}
  end

  defn svd_non_zero(input_tensor, opts \\ []) do
    {is_flipped, a} =
      case Nx.shape(input_tensor) do
        {m, n} when m < n ->
          {true, Nx.LinAlg.adjoint(input_tensor)}

        _ ->
          {false, input_tensor}
      end

    {u, s, v} =
      if opts[:full_matrices?] do
        svd_full(a, opts)
      else
        svd_non_full(a, opts)
      end

    if is_flipped do
      {v, s, Nx.LinAlg.adjoint(u)}
    else
      {u, s, Nx.LinAlg.adjoint(v)}
    end
  end

  defnp svd_tall_and_square(a, opts \\ []) do
    {_m, n} = Nx.shape(a)
    {u, h} = qdwh(a, opts)
    # ensure H is hermitian
    h = (h + Nx.LinAlg.adjoint(h)) / 2
    {s, v} = Nx.LinAlg.eigh(h, max_iter: opts[:max_iter])

    sign = Nx.select(s < 0, -1, 1)

    v = sign * v
    s = sign * s

    # sort s and v according to
    sort_idx = Nx.argsort(s, direction: :desc)

    s_out = Nx.take(s, sort_idx)
    v_out = Nx.take(v, sort_idx, axis: 1)

    u_out = Nx.dot(u, v_out)

    u_out = Nx.select(s[0] < n * @eps * s_out[0], correct_rank_deficiency(u_out), u_out)
    {u_out, s_out, v_out}
  end

  defn qdwh(x, opts \\ []) do
    # reference implementation taken from Jax
    alpha = Nx.sqrt(Nx.LinAlg.norm(x, ord: 1)) * Nx.sqrt(Nx.LinAlg.norm(x, ord: :inf))
    l = @eps

    u = x / alpha
    tol_l = 5 * @eps
    tol_norm = Nx.cbrt(tol_l)

    one_u8 = Nx.iota({}, type: :u8, vectorized_axes: u.vectorized_axes) + 1
    original_type = Nx.type(u)
    u = Nx.as_type(u, min_precision_type(original_type))

    {u, _l, _num_iters, _is_unconverged, _is_not_max_iteration, _max_iter} =
      while {u, l, iter_idx = 1, is_unconverged = one_u8, is_not_max_iteration = one_u8,
             max_iter = opts[:max_iter]},
            is_unconverged and is_not_max_iteration do
        u_prev = u

        l2 = l ** 2
        # if l2 is too small, dd will tend to infinity.
        # keeping it at the `eps` noise floor helps
        # avoid this problem.
        l2 = Nx.select(l2 < @eps, @eps, l2)
        dd = Nx.cbrt(4.0 * (1.0 / l2 - 1.0) / l2)
        sqd = Nx.sqrt(1.0 + dd)
        a = sqd + Nx.sqrt(8.0 - 4.0 * dd + 8.0 * (2.0 - l2) / (l2 * sqd)) / 2
        a = Nx.real(a)
        b = (a - 1.0) ** 2 / 4.0
        c = a + b - 1.0

        l = l * (a + b * l2) / (1.0 + c * l2)

        u =
          if c > 100 do
            qdwh_use_qr(u, x, a, b, c)
          else
            qdwh_use_cholesky(u, x, a, b, c)
          end

        iterating_l = Nx.abs(1.0 - l) > tol_l
        iterating_u = Nx.LinAlg.norm(u - u_prev) > tol_norm
        is_unconverged = iterating_l or iterating_u
        is_not_max_iteration = iter_idx < max_iter
        {u, l, iter_idx + 1, is_unconverged, is_not_max_iteration, max_iter}
      end

    u = Nx.as_type(u, original_type)
    u = 1.5 * u - 0.5 * Nx.dot(u, Nx.dot(Nx.LinAlg.adjoint(u), u))
    h = u |> Nx.LinAlg.adjoint() |> Nx.dot(x)
    h = (h + Nx.LinAlg.adjoint(h)) / 2
    {u, h}
  end

  # f16 is not enough precision to compute SVD
  deftransformp(min_precision_type({:f, 64}), do: {:f, 64})
  deftransformp(min_precision_type(_), do: {:f, 32})

  defn qdwh_use_qr(u, x, a, b, c) do
    {m, _n} = Nx.shape(x)
    {_u_m, u_n} = Nx.shape(u)

    y = Nx.concatenate([Nx.sqrt(c) * u, Nx.eye(u_n)], axis: 0)
    {q, _r} = Nx.LinAlg.qr(y)
    q1 = q[0..(m - 1)//1]
    q2 = Nx.LinAlg.adjoint(q[m..-1//1])
    e = b / c

    e * u + (a - e) / Nx.sqrt(c) * Nx.dot(q1, q2)
  end

  defn qdwh_use_cholesky(u, _x, a, b, c) do
    {_, u_n} = Nx.shape(u)
    uh = Nx.LinAlg.adjoint(u)
    x = c * Nx.dot(uh, u) + Nx.eye(u_n)
    y = Nx.LinAlg.cholesky(x)

    ybar =
      case Nx.type(y) do
        {:c, _} -> Nx.conjugate(y)
        _ -> y
      end

    zbar =
      Nx.LinAlg.triangular_solve(
        ybar,
        Nx.transpose(u),
        left_side: true,
        lower: true
      )

    z =
      case Nx.type(zbar) do
        {:c, _} -> Nx.conjugate(zbar)
        _ -> zbar
      end

    z =
      y
      |> Nx.LinAlg.adjoint()
      |> Nx.LinAlg.triangular_solve(z, left_side: true, lower: false)
      |> Nx.LinAlg.adjoint()

    e = b / c

    e * u + (a - e) * z
  end

  defnp correct_rank_deficiency(u_out) do
    {u_out, r} = Nx.LinAlg.qr(u_out)

    diagonal = Nx.take_diagonal(r)
    diag_r = diagonal < 0
    sign_r = Nx.select(diag_r, -1, 1)
    sign_r = Nx.select(diagonal == 0, 0, sign_r)

    # same as Nx.dot(u_out, Nx.make_diagonal(sign_r))
    u_out * sign_r
  end

  defnp svd_grad({u, s_input, vt}, input, {du, ds, dvt}) do
    {k} = Nx.shape(s_input)
    {m, n} = Nx.shape(input)

    if m < n do
      raise "grad for Nx.LinAlg.svd/2 not implemented for the wide matrix case"
    end

    u =
      if m == n do
        u
      else
        u[[0..(m - 1), 0..(k - 1)]]
      end

    du =
      if m == n do
        du
      else
        du[[0..(m - 1), 0..(k - 1)]]
      end

    # https://j-towns.github.io/papers/svd-derivative.pdf

    eye_k = Nx.eye(k)
    eye_m = Nx.eye(m)
    eye_n = Nx.eye(n)

    s_sq = s_input ** 2
    sub = -(Nx.new_axis(s_sq, 1) - s_sq) + eye_k
    f = Nx.select(eye_k, 0, 1 / sub)

    s = Nx.make_diagonal(s_input)
    s_inv = Nx.make_diagonal(1 / s_input)

    ut_du = Nx.dot(Nx.LinAlg.adjoint(u), du) - Nx.dot(Nx.LinAlg.adjoint(du), u)

    first_component_du = u |> Nx.dot(f * ut_du) |> Nx.dot(s)

    second_component_du = (eye_m - Nx.dot(u, Nx.LinAlg.adjoint(u))) |> Nx.dot(du) |> Nx.dot(s_inv)

    du_component = Nx.dot(first_component_du + second_component_du, vt)

    ds_component = u |> Nx.dot(eye_k * ds) |> Nx.dot(vt)

    first_dvt_component =
      (Nx.dot(vt, Nx.LinAlg.adjoint(dvt)) - Nx.dot(dvt, Nx.LinAlg.adjoint(vt))) * f

    first_dvt_component = s |> Nx.dot(first_dvt_component) |> Nx.dot(vt)

    second_dvt_component =
      s_inv |> Nx.dot(dvt) |> Nx.dot(eye_n - Nx.dot(Nx.LinAlg.adjoint(vt), vt))

    dvt_component = Nx.dot(u, first_dvt_component + second_dvt_component)

    [du_component + ds_component + dvt_component]
  end
end
