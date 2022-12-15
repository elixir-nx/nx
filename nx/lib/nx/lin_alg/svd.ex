defmodule Nx.LinAlg.SVD do
  import Nx.Defn
  @default_eps 1.1920929e-07

  defn svd(input_tensor, opts \\ []) do
    opts = keyword!(opts, max_iter: 100, eps: @default_eps)

    {is_flipped, tensor} =
      case Nx.shape(input_tensor) do
        {m, n} when m < n ->
          {true, Nx.LinAlg.adjoint(input_tensor)}

        _ ->
          {false, input_tensor}
      end

    # We always return full matrices for retrocompatibility.
    # Original Jax code has extensions if we want to add that
    # as an option (i.e. mode: :complete | :reduced)

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
      case reduce_to_square do
        true ->
          u = Nx.dot(q, u)
          Nx.concatenate([u, u_null], axis: -1)

        false ->
          u
      end

    res =
      case is_flipped do
        true -> {v, s, Nx.LinAlg.adjoint(u)}
        false -> {u, s, Nx.LinAlg.adjoint(v)}
      end

    custom_grad(res, [input_tensor], fn g ->
      grad(res, input_tensor, g)
    end)
  end

  defnp svd_tall_and_square(a, opts \\ []) do
    {_m, n} = Nx.shape(a)
    {u, h} = qdwh(a, opts)
    # ensure H is hermitian
    h = (h + Nx.LinAlg.adjoint(h)) / 2
    {s, v} = Nx.LinAlg.eigh(h, max_iter: opts[:max_iter], eps: 1.0e-4)

    sign = Nx.select(s < 0, -1, 1)
    v = sign * v
    s = sign * s

    # sort s and v according to
    sort_idx = Nx.argsort(s, direction: :desc)
    s_out = Nx.take(s, sort_idx)
    v_out = Nx.take(v, sort_idx, axis: 1)
    u_out = Nx.dot(u, v_out)

    u_out = Nx.select(s[0] < n * opts[:eps] * s_out[0], correct_rank_deficiency(u_out), u_out)
    {u_out, s_out, v_out}
  end

  defn qdwh(x, opts \\ []) do
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
        # if l2 is too small, dd will tend to infinity.
        # keeping it at the `eps` noise floor helps
        # avoid this problem.
        l2 = Nx.select(l2 < opts[:eps], opts[:eps], l2)
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

    u = 1.5 * u - 0.5 * Nx.dot(u, Nx.dot(Nx.LinAlg.adjoint(u), u))
    h = u |> Nx.LinAlg.adjoint() |> Nx.dot(x)
    h = (h + Nx.LinAlg.adjoint(h)) / 2
    {u, h}
  end

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

    # z = Nx.LinAlg.solve(y, Nx.LinAlg.adjoint(z))
    z =
      Nx.LinAlg.triangular_solve(Nx.LinAlg.adjoint(y), z, left_side: true, lower: false)
      |> Nx.LinAlg.adjoint()

    # # |> Nx.real()

    e = b / c

    e * u + (a - e) * z
  end

  defnp correct_rank_deficiency(u_out) do
    {u_out, r} = Nx.LinAlg.qr(u_out)

    diag_r = Nx.take_diagonal(r) < 0
    sign_r = Nx.select(diag_r, -1, 1)
    sign_r = Nx.select(diag_r == 0, 0, sign_r)

    # same as Nx.dot(u_out, Nx.make_diagonal(sign_r))
    u_out * sign_r
  end

  defnp grad({u, s_input, vt}, input, {du, ds, dvt}) do
    {k} = Nx.shape(s_input)
    {m, n} = Nx.shape(input)

    if m < n do
      raise "grad for Nx.LinAlg.svd/2 not implemented for the wide matrix case"
    end

    u =
      if m == n do
        u
      else
        Nx.slice(u, [0, 0], [m, k])
      end

    du =
      if m == n do
        du
      else
        Nx.slice(du, [0, 0], [m, k])
      end

    # https://j-towns.github.io/papers/svd-derivative.pdf

    eye_k = Nx.eye(k)
    eye_m = Nx.eye(m)
    eye_n = Nx.eye(n)

    s_sq = Nx.power(s_input, 2)
    sub = s_sq |> Nx.new_axis(1) |> Nx.subtract(s_sq) |> Nx.negate() |> Nx.add(eye_k)
    f = Nx.select(eye_k, 0, Nx.divide(1, sub))

    s = s_input |> Nx.make_diagonal()
    s_inv = 1 |> Nx.divide(s_input) |> Nx.make_diagonal()

    ut_du =
      u |> Nx.LinAlg.adjoint() |> Nx.dot(du) |> Nx.subtract(Nx.dot(Nx.LinAlg.adjoint(du), u))

    first_component_du = u |> Nx.dot(Nx.multiply(f, ut_du)) |> Nx.dot(s)

    second_component_du =
      eye_m
      |> Nx.subtract(Nx.dot(u, Nx.LinAlg.adjoint(u)))
      |> Nx.dot(du)
      |> Nx.dot(s_inv)

    du_component = first_component_du |> Nx.add(second_component_du) |> Nx.dot(vt)

    ds_component = u |> Nx.dot(Nx.multiply(eye_k, ds)) |> Nx.dot(vt)

    first_dvt_component =
      vt
      |> Nx.dot(Nx.LinAlg.adjoint(dvt))
      |> Nx.subtract(Nx.dot(dvt, Nx.LinAlg.adjoint(vt)))
      |> Nx.multiply(f)

    first_dvt_component = s |> Nx.dot(first_dvt_component) |> Nx.dot(vt)

    second_dvt_component =
      s_inv |> Nx.dot(dvt) |> Nx.dot(Nx.subtract(eye_n, Nx.dot(Nx.LinAlg.adjoint(vt), vt)))

    dvt_component = Nx.dot(u, Nx.add(first_dvt_component, second_dvt_component))

    da = du_component |> Nx.add(ds_component) |> Nx.add(dvt_component)

    [{input, da}]
  end
end
