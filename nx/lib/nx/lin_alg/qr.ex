defmodule Nx.LinAlg.QR do
  import Nx.Defn

  defn qr(a, opts \\ []) do
    opts =
      keyword!(opts, eps: 1.0e-10, mode: :reduced)

    vectorized_axes = a.vectorized_axes

    result =
      a
      |> Nx.revectorize([collapsed_axes: :auto],
        target_shape: {Nx.axis_size(a, -2), Nx.axis_size(a, -1)}
      )
      |> qr_matrix(opts)
      |> revectorize_result(a.shape, vectorized_axes, opts)

    custom_grad(result, [a], fn g ->
      qr_grad(result, a, g)
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
        h = householder_reflector(x, i, eps)
        r = Nx.dot(h, r)
        q = Nx.dot(q, h)
        {{q, r}, {column_iota}}
      end

    q = approximate_zeros(q, eps)
    r = approximate_zeros(r, eps)

    output_mode_handling(q, r, m_in, n_in, k, wide_mode, mode)
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
    # x is a {n} tensor
    {norm_x, norm_x_sq} = norm(x)

    x_i = x[i]

    norm_sq_1on = norm_x_sq - Nx.abs(x_i) ** 2

    {v, scale} =
      case Nx.type(x) do
        {:c, _} ->
          phase = Nx.phase(x_i)
          arg = Nx.complex(0, phase)
          alpha = Nx.exp(arg) * norm_x
          u = Nx.indexed_add(x, Nx.new_axis(i, 0), alpha)
          {n_u, _} = norm(u)
          v = u / n_u
          {v, 2}

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

    selector = Nx.iota({Nx.size(x)}) |> Nx.greater_equal(i) |> then(&Nx.outer(&1, &1))

    eye = Nx.eye(Nx.size(x))
    Nx.select(selector, eye - scale * Nx.outer(v, v), eye)
  end

  defn qr_grad({q, r}, _input, {dq, dr}) do
    # Definition taken from https://arxiv.org/pdf/2009.10071.pdf
    # Equation (3)
    r_inv = Nx.LinAlg.invert(r)

    m = Nx.dot(r, Nx.LinAlg.adjoint(dr)) |> Nx.subtract(Nx.dot(Nx.LinAlg.adjoint(dq), q))

    # copyltu
    m_ltu = Nx.tril(m) |> Nx.add(m |> Nx.tril(k: -1) |> Nx.LinAlg.adjoint())

    da = dq |> Nx.add(Nx.dot(q, m_ltu)) |> Nx.dot(Nx.LinAlg.adjoint(r_inv))

    [da]
  end
end
