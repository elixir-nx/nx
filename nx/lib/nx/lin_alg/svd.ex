defmodule Nx.LinAlg.SVD do
  import Nx.Defn
  @default_eps 1.0e-10

  defn svd(tensor, opts \\ []) do
    opts = keyword!(opts, max_iter: 10_000, eps: @default_eps)
    # {m, n} = Nx.shape(tensor)

    {u, h} = qdwh(tensor, opts)
    {v, s} = Nx.LinAlg.eigh(h)
    # sort_idx = Nx.argsort(s)
    # s_out = Nx.take(s, sort_idx)
    # v_out = Nx.take_along_axis(v, sort_idx, axis: -1)
    u_out = Nx.dot(u, v)

    # Nx.select(s[0] < m * opts[:eps] * s[0], correct_rank_deficiency(u_out), u_out)
    {u_out, s, v}
  end

  defnp qdwh(x, opts \\ []) do
    # {m, n} = Nx.shape(x)
    alpha = Nx.sqrt(Nx.LinAlg.norm(x, ord: 1)) * Nx.sqrt(Nx.LinAlg.norm(x, ord: :inf))
    l = opts[:eps]
    u = x / alpha
    tol_l = 5 * opts[:eps]
    tol_norm = Nx.cbrt(tol_l)

    one_u8 = Nx.tensor(1, type: :u8)

    {u, _l, _num_iters, _is_unconverged, _is_not_max_iteration, _max_iter} =
      while {u, l, iter_idx = 1, is_unconverged = one_u8, is_not_max_iteration = one_u8,
             max_iter = opts[:max_iter]},
            is_unconverged and is_not_max_iteration do
        u_prev = u

        l2 = l ** 2
        dd = Nx.cbrt(4.0 * (1.0 / l2 - 1.0) / l2)
        sqd = Nx.sqrt(1 + dd)
        a = sqd + Nx.sqrt(8 - 4 * dd + 8 * (2 - l2) / (l2 * sqd)) / 2
        b = (a - 1.0) ** 2 / 4
        c = a + b - 1
        l = l * (a + b * l2) / (1 + c * l2)

        u =
          Nx.real(
            if c > 100 do
              qdwh_use_qr(u, x, a, b, c)
            else
              qdwh_use_cholesky(u, x, a, b, c)
            end
          )

        iterating_l = Nx.abs(1.0 - l) > tol_l
        iterating_u = Nx.LinAlg.norm(u - u_prev) > tol_norm
        is_unconverged = iterating_l or iterating_u
        is_not_max_iteration = iter_idx < max_iter
        {u, l, iter_idx + 1, is_unconverged, is_not_max_iteration, max_iter}
      end

    u = 1.5 * u - 0.5 * Nx.dot(u, Nx.dot(Nx.LinAlg.adjoint(u), u))
    h = u |> Nx.LinAlg.adjoint() |> Nx.dot(x)
    h = (h + Nx.LinAlg.adjoint(h)) / 2
    {u, h}
  end

  defn qdwh_use_qr(u, x, a, b, c) do
    {m, n} = Nx.shape(x)
    {_u_m, u_n} = Nx.shape(u)

    y = Nx.concatenate([Nx.sqrt(c) * u, Nx.eye(u_n)])
    {q, _r} = Nx.LinAlg.qr(y)
    q1 = Nx.slice(q, [0, 0], [m, n])
    q2 = Nx.LinAlg.adjoint(q[m..-1//1])
    e = b / c

    e * u + (a - e) / Nx.sqrt(c) * Nx.dot(q1, q2)
  end

  defn qdwh_use_cholesky(u, _x, a, b, c) do
    # {m, n} = Nx.shape(x)
    {_, u_n} = Nx.shape(u)
    x = c * Nx.dot(Nx.LinAlg.adjoint(u), u) + Nx.eye(u_n)
    # TO-DO: mask x with identity

    y = Nx.LinAlg.cholesky(x)

    z =
      Nx.LinAlg.triangular_solve(Nx.conjugate(y), Nx.transpose(u), left_side: true, lower: true)
      |> Nx.conjugate()

    z =
      Nx.LinAlg.triangular_solve(Nx.LinAlg.adjoint(y), z, left_side: true, lower: true)
      |> Nx.LinAlg.adjoint()

    e = b / c

    e * u + (a - e) * z
  end

  # defnp correct_rank_deficiency(u_out) do
  #   {u_out, r} = Nx.LinAlg.qr(u_out)

  #   diag_r = Nx.take_diagonal(r) < 0
  #   sign_r = Nx.select(diag_r, -1, 1)
  #   sign_r = Nx.select(diag_r == 0, 0, sign_r)

  #   # same as Nx.dot(u_out, Nx.make_diagonal(sign_r))
  #   u_out * sign_r
  # end
end
