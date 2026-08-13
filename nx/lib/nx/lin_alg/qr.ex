defmodule Nx.LinAlg.QR do
  @moduledoc false
  import Nx.Defn

  defn qr(a, opts) do
    opts = keyword!(opts, mode: :reduced, eps: 1.0e-10)
    vectorized_axes = a.vectorized_axes

    result =
      a
      |> Nx.revectorize([collapsed_axes: :auto],
        target_shape: {Nx.axis_size(a, -2), Nx.axis_size(a, -1)}
      )
      |> qr_matrix(opts)
      |> revectorize_result(a.shape, vectorized_axes, opts)

    custom_grad(result, [a], fn g ->
      qr_grad(result, g)
    end)
  end

  deftransformp revectorize_result({q, r}, shape, vectorized_axes, opts) do
    {q_shape, r_shape} = Nx.Shape.qr(shape, opts)

    {
      Nx.revectorize(q, vectorized_axes, target_shape: q_shape),
      Nx.revectorize(r, vectorized_axes, target_shape: r_shape)
    }
  end

  deftransformp wide_mode_extension(a) do
    case Nx.shape(a) do
      {m, n} when m < n ->
        # "Matrix Computations" by Golub and Van Loan: Section 5.4.1
        # describes the problem of computing QR factorization for wide matrices,
        # and suggests adding rows of zeros as a solution.
        a = Nx.pad(a, 0, [{0, n - m, 0}, {0, 0, 0}])
        {a, n, n, n, true, n - 1}

      {m, n} ->
        max_i = if m == n, do: n - 2, else: n - 1
        {a, m, n, min(m, n), false, max_i}
    end
  end

  defnp qr_matrix(a, opts \\ []) do
    mode = opts[:mode]
    eps = opts[:eps]
    {m_in, n_in} = Nx.shape(a)

    {a, m, _n, k, wide_mode, max_i} = wide_mode_extension(a)

    type = Nx.Type.to_floating(Nx.type(a))

    base_h = Nx.eye({m, m}, type: type, vectorized_axes: a.vectorized_axes)
    column_iota = Nx.iota({Nx.axis_size(a, 0)}, vectorized_axes: a.vectorized_axes)

    {{q, r}, _} =
      while {{q = base_h, r = Nx.as_type(a, type)}, {column_iota}}, i <- 0..max_i//1 do
        x = r[[.., i]]
        x = Nx.select(column_iota < i, 0, x)
        {v, scale} = householder_reflector(x, i, eps)

        # v is always a 1D tensor, so we don't have to worry about transposing
        # which is why conjugate_if_complex is in place.
        vh_r = Nx.dot(conjugate_if_complex(v), r)
        r = r - scale * Nx.dot(Nx.new_axis(v, 1), Nx.new_axis(vh_r, 0))

        q_v = Nx.dot(q, v)
        q = q - scale * Nx.outer(q_v, v)
        {{q, r}, {column_iota}}
      end

    q = approximate_zeros(q, eps)
    r = approximate_zeros(r, eps)

    output_mode_handling(q, r, m_in, n_in, k, wide_mode, mode)
  end

  defnp conjugate_if_complex(x) do
    case Nx.type(x) do
      {:c, _} -> Nx.conjugate(x)
      _ -> x
    end
  end

  deftransformp output_mode_handling(q, r, m_in, n_in, k, wide_mode, mode) do
    {m, _} = Nx.shape(q)
    {_, n} = Nx.shape(r)

    cond do
      wide_mode ->
        # output {m, m} and {m, n} from q {n, n} and r {n, n}
        {q[[0..(m_in - 1), 0..(m_in - 1)]], r[[0..(m_in - 1), 0..(n_in - 1)]]}

      mode == :reduced and m > n ->
        # output {m, m} and {n, n} from q {m, n} and r {n, n}
        {q[[.., 0..(k - 1)]], r[[0..(n_in - 1), 0..(n_in - 1)]]}

      true ->
        {q, r}
    end
  end

  defnp approximate_zeros(matrix, eps), do: Nx.select(Nx.abs(matrix) <= eps, 0, matrix)

  defnp norm(x) do
    case Nx.type(x) do
      {:c, _} ->
        n = Nx.dot(x, Nx.conjugate(x))
        {Nx.sqrt(n), n}

      _ ->
        n = Nx.dot(x, x)
        {Nx.sqrt(n), n}
    end
  end

  defn householder_reflector(x, i, eps) do
    {norm_x, norm_x_sq} = norm(x)

    x_i = x[i]

    norm_sq_1on = norm_x_sq - Nx.abs(x_i) ** 2

    case Nx.type(x) do
      {:c, _} ->
        phase = Nx.phase(x_i)
        arg = Nx.complex(0, phase)
        alpha = Nx.exp(arg) * norm_x
        u = Nx.indexed_add(x, Nx.new_axis(i, 0), alpha)
        {n_u, n_u_sq} = norm(u)
        norm_selector = Nx.real(n_u_sq) < eps
        {u / Nx.select(norm_selector, 1, n_u), Nx.select(norm_selector, 0, 2)}

      _type ->
        v_0 = Nx.select(x_i <= 0, x_i - norm_x, -norm_sq_1on / (x_i + norm_x))

        norm_selector = norm_sq_1on < eps

        replace_value =
          Nx.select(norm_selector, Nx.tensor([1], type: x.type), Nx.reshape(v_0, {1}))

        v = Nx.put_slice(x, [i], replace_value)
        v = v / Nx.select(norm_selector, 1, v_0)
        {_, n_v_sq} = norm(v)
        scale_den = Nx.select(norm_selector, 1, n_v_sq)
        scale = Nx.select(norm_selector, 0, 2 / scale_den)
        {v, scale}
    end
  end

  defn qr_grad({q, r}, {dq, dr}) do
    [dispatch_qr_grad(q, r, dq, dr)]
  end

  # Square / reduced-tall / complete-tall: https://arxiv.org/pdf/2009.10071.pdf Equation (3).
  # Wide: Proposition 2 in the same paper (partition R = [U | V]).
  deftransformp dispatch_qr_grad(q, r, dq, dr) do
    rank = tuple_size(Nx.shape(r))
    r_rows = elem(Nx.shape(r), rank - 2)
    r_cols = elem(Nx.shape(r), rank - 1)

    if r_rows < r_cols do
      qr_grad_wide(q, r, dq, dr)
    else
      qr_grad_square(q, r, dq, dr)
    end
  end

  defnp qr_grad_square(q, r, dq, dr) do
    r_sq = take_square_r(r)
    eye = Nx.eye(Nx.shape(r_sq), type: Nx.type(r))
    r_inv = Nx.LinAlg.triangular_solve(r_sq, eye, lower: false)
    batch_axes = batch_axes(r)

    dr_conj =
      case Nx.type(dr) do
        {:c, _} -> Nx.conjugate(dr)
        _ -> dr
      end

    dq_conj =
      case Nx.type(dq) do
        {:c, _} -> Nx.conjugate(dq)
        _ -> dq
      end

    m =
      r
      |> Nx.dot([-1], batch_axes, dr_conj, [-1], batch_axes)
      |> Nx.subtract(Nx.dot(dq_conj, [-2], batch_axes, q, [-2], batch_axes))

    # copyltu
    m_ltu = Nx.tril(m) |> Nx.add(m |> Nx.tril(k: -1) |> Nx.LinAlg.adjoint())

    dq_q_m = dq + Nx.dot(q, [-1], batch_axes, m_ltu, [-2], batch_axes)

    # R may have extra zero rows (complete tall). Invert the leading square block
    # and drop the matching extra columns so the multiply matches R^{-H}.
    dq_q_m = Nx.slice_along_axis(dq_q_m, 0, Nx.axis_size(r_sq, -1), axis: -1)


    r_inv_conj =
      case Nx.type(r_inv) do
        {:c, _} -> Nx.conjugate(r_inv)
        _ -> r_inv
      end

    Nx.dot(dq_q_m, [-1], batch_axes, r_inv_conj, [-1], batch_axes)
  end

  defnp qr_grad_wide(q, r, dq, dr) do
    m = Nx.axis_size(q, -2)
    u = Nx.slice_along_axis(r, 0, m, axis: -1)
    du = Nx.slice_along_axis(dr, 0, m, axis: -1)
    v = Nx.slice_along_axis(r, m, Nx.axis_size(r, -1) - m, axis: -1)
    dv = Nx.slice_along_axis(dr, m, Nx.axis_size(dr, -1) - m, axis: -1)

    batch_axes = batch_axes(r)
    y = Nx.dot(q, [-1], batch_axes, v, [-2], batch_axes)

    # Use implicit transpose in dot instead of adjoint
    dv_conj =
      case Nx.type(dv) do
        {:c, _} -> Nx.conjugate(dv)
        _ -> dv
      end

    dq_prime =
      dq + Nx.dot(y, [-1], batch_axes, dv_conj, [-1], batch_axes)

    dx = qr_grad_square(q, u, dq_prime, du)
    dy = Nx.dot(q, [-1], batch_axes, dv, [-2], batch_axes)
    Nx.concatenate([dx, dy], axis: -1)
  end

  deftransformp take_square_r(r) do
    rank = tuple_size(Nx.shape(r))
    rows = elem(Nx.shape(r), rank - 2)
    cols = elem(Nx.shape(r), rank - 1)
    k = min(rows, cols)

    r
    |> Nx.slice_along_axis(0, k, axis: -2)
    |> Nx.slice_along_axis(0, k, axis: -1)
  end

  deftransformp batch_axes(t) do
    rank = tuple_size(t.shape)
    Enum.to_list(0..(rank - 3)//1)
  end
end
