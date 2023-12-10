defmodule Nx.LinAlg.QR do
  import Nx.Defn

  defn qr(a, opts \\ []) do
    opts =
      keyword!(opts, eps: 1.0e-10, mode: :reduced)

    vectorized_axes = a.vectorized_axes

    a
    |> Nx.revectorize([collapsed_axes: :auto],
      target_shape: {Nx.axis_size(a, -2), Nx.axis_size(a, -1)}
    )
    |> qr_matrix(opts)
    |> revectorize_result(a.shape, vectorized_axes, opts)
  end

  deftransformp revectorize_result({q, r}, shape, vectorized_axes, opts) do
    {q_shape, r_shape} = Nx.Shape.qr(shape, opts)

    {
      Nx.revectorize(q, vectorized_axes, target_shape: q_shape),
      Nx.revectorize(r, vectorized_axes, target_shape: r_shape)
    }
  end

  # TO-DO: deal with various non-square cases
  #  def qr(input_data, {_, s} = input_type, input_shape, output_type, m_in, k_in, n_in, opts) do
  #   mode = opts[:mode]
  #   eps = opts[:eps]

  #   {input_data, m, n, k, wide_mode} =
  #     if m_in < n_in do
  #       # "Matrix Computations" by Golub and Van Loan: Section 5.4.1
  #       # describes the problem of computing QR factorization for wide matrices,
  #       # and suggests adding rows of zeros as a solution.

  #       ext_size = s * (n_in - m_in) * n_in
  #       extended = input_data <> <<0::size(ext_size)>>
  #       {extended, n_in, n_in, n_in, true}
  #     else
  #       {input_data, m_in, n_in, k_in, false}
  #     end

  #   {q_matrix, r_matrix} =
  #     input_data
  #     |> binary_to_matrix(input_type, input_shape)
  #     |> qr_decomposition(m, n, eps)

  #   {q_matrix, r_matrix} =
  #     cond do
  #       wide_mode ->
  #         # output {m, m} and {m, n} from q {n, n} and r {n, n}
  #         q_matrix =
  #           q_matrix
  #           |> get_matrix_columns(0..(m_in - 1))
  #           |> Enum.take(m_in)

  #         r_matrix = Enum.take(r_matrix, m_in)
  #         {q_matrix, r_matrix}

  #       mode == :reduced and m > n ->
  #         # output {m, m} and {n, n} from q {m, n} and r {n, n}
  #         q_matrix = get_matrix_columns(q_matrix, 0..(k - 1))

  #         r_matrix = Enum.drop(r_matrix, k - m)

  #         {q_matrix, r_matrix}

  #       true ->
  #         {q_matrix, r_matrix}
  #     end

  #   {matrix_to_binary(q_matrix, output_type), matrix_to_binary(r_matrix, output_type)}
  # end

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

    {a, m, n, k, wide_mode, max_i} = wide_mode_extension(a)

    type = Nx.Type.to_floating(Nx.type(a))

    base_h = Nx.eye({m, m}, type: type, vectorized_axes: a.vectorized_axes)
    column_iota = Nx.iota({Nx.axis_size(a, 0)}, vectorized_axes: a.vectorized_axes)

    {{q, r}, _} =
      while {{q = base_h, r = Nx.as_type(a, type)}, {column_iota}}, i <- 0..max_i do
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
        Nx.sqrt(Nx.dot(x, Nx.conjugate(x)))

      _ ->
        Nx.sqrt(Nx.dot(x, x))
    end
  end

  defn householder_reflector(x, i, eps) do
    # x is a {n} tensor
    norm_x = norm(x)

    x_i = x[i]

    norm_sq_1on = norm_x ** 2 - x_i ** 2

    {v, scale} =
      case Nx.type(x) do
        {:c, _} ->
          alpha = Nx.exp(Nx.Constants.i() * Nx.phase(x[i]))
          u = Nx.indexed_add(x, Nx.new_axis(i, 0), alpha * norm_x)
          v = u / norm(u)
          {v, 2}

        type ->
          cond do
            norm_sq_1on < eps ->
              v = Nx.put_slice(x, [i], Nx.tensor([1], type: Nx.type(x)))
              {v, 0}

            true ->
              v_0 =
                if x_i <= 0 do
                  x_i - norm_x
                else
                  -norm_sq_1on / (x_i + norm_x)
                end

              v = Nx.put_slice(x, [i], Nx.reshape(v_0, {1}))
              v = v / v_0
              scale = 2 / Nx.dot(v, v)

              {v, scale}
          end
      end

    selector = Nx.iota({Nx.size(x)}) |> Nx.greater_equal(i) |> then(&Nx.outer(&1, &1))

    eye = Nx.eye(Nx.size(x))
    Nx.select(selector, eye - scale * Nx.outer(v, v), eye)
  end
end
