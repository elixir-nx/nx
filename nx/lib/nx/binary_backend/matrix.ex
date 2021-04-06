defmodule Nx.BinaryBackend.Matrix do
  @moduledoc false
  @default_eps 1.0e-10
  import Nx.Shared

  def ts(a_data, a_type, b_data, b_type, {rows, rows} = shape, output_type, _opts) do
    a_matrix = binary_to_matrix(a_data, a_type, shape)
    b_matrix = binary_to_matrix(b_data, b_type, shape)

    Enum.uniq(1..rows)
    |> Enum.map(fn b_col ->
      b_vector = get_matrix_column(b_matrix, b_col - 1)

      ts(a_matrix, b_vector, 0, [])
    end)
    |> transpose_matrix()
    |> matrix_to_binary(output_type)
  end

  def ts(a_data, a_type, b_data, b_type, {rows}, output_type, _opts) do
    a_matrix = binary_to_matrix(a_data, a_type, {rows, rows})
    b_vector = binary_to_vector(b_data, b_type)

    ts(a_matrix, b_vector, 0, [])
    |> matrix_to_binary(output_type)
  end

  defp ts([row | rows], [b | bs], idx, acc) do
    value = Enum.fetch!(row, idx)

    if value == 0 do
      raise ArgumentError, "can't solve for singular matrix"
    end

    y = (b - dot_matrix(row, acc)) / value
    ts(rows, bs, idx + 1, acc ++ [y])
  end

  defp ts([], [], _idx, acc), do: acc

  def qr(input_data, input_type, input_shape, output_type, m, k, n, opts) do
    {_, input_num_bits} = input_type

    mode = opts[:mode]
    eps = opts[:eps] || @default_eps

    r_matrix =
      if mode == :reduced do
        # Since we want the first k rows of r, we can
        # just slice the binary by taking the first
        # n * k * output_type_num_bits bits from the binary.
        # Trick for r = tensor[[0..(k - 1), 0..(n - 1)]]
        slice_size = n * k * input_num_bits
        <<r_bin::bitstring-size(slice_size), _::bitstring>> = input_data
        binary_to_matrix(r_bin, input_type, {k, n})
      else
        binary_to_matrix(input_data, input_type, input_shape)
      end

    {q_matrix, r_matrix} =
      for i <- 0..(n - 1), reduce: {nil, r_matrix} do
        {q, r} ->
          # a = r[[i..(k - 1), i]]
          a = r |> Enum.slice(i..(k - 1)) |> Enum.map(fn row -> Enum.at(row, i) end)

          h = householder_reflector(a, k, eps)

          # If we haven't allocated Q yet, let Q = H1
          q =
            if is_nil(q) do
              zero_padding = 0 |> List.duplicate(k) |> List.duplicate(m - k)
              h ++ zero_padding
            else
              dot_matrix(q, h)
            end

          r = dot_matrix(h, r)

          {q, r}
      end

    {matrix_to_binary(q_matrix, output_type), matrix_to_binary(r_matrix, output_type)}
  end

  def lu(input_data, input_type, {n, n} = input_shape, p_type, l_type, u_type, opts) do
    a = binary_to_matrix(input_data, input_type, input_shape)
    eps = opts[:eps] || @default_eps

    {p, a_prime} = lu_validate_and_pivot(a, n)

    # We'll work with linear indices because of the way each matrix
    # needs to be updated/accessed
    zeros_matrix = List.duplicate(List.duplicate(0, n), n)

    {l, u} =
      for j <- 0..(n - 1), reduce: {zeros_matrix, zeros_matrix} do
        {l, u} ->
          i_range = if j == 0, do: [0], else: 0..j
          [l_row_j] = get_matrix_rows(l, [j])
          l = replace_rows(l, [j], [replace_vector_element(l_row_j, j, 1.0)])

          u_col_j =
            for i <- i_range,
                reduce: get_matrix_column(u, j) do
              u_col_j ->
                l_row_slice = slice_matrix(l, [i, 0], [1, i])

                u_col_slice = slice_vector(u_col_j, 0, i)

                s = dot_matrix(l_row_slice, u_col_slice)
                [a_elem] = get_matrix_elements(a_prime, [[i, j]])
                replace_vector_element(u_col_j, i, a_elem - s)
            end

          u = replace_cols(u, [j], transpose_matrix(u_col_j))

          [u_jj] = get_matrix_elements(u, [[j, j]])

          l =
            if u_jj < eps do
              l
            else
              i_range = if j == n - 1, do: [n - 1], else: j..(n - 1)

              for i <- i_range, reduce: l do
                l ->
                  u_col_slice = slice_matrix(u, [0, j], [j, 1])
                  [l_row_i] = get_matrix_rows(l, [i])
                  s = dot_matrix(u_col_slice, l_row_i)
                  [a_elem] = get_matrix_elements(a_prime, [[i, j]])

                  l_updated_row = replace_vector_element(l_row_i, j, (a_elem - s) / u_jj)

                  replace_rows(l, [i], [l_updated_row])
              end
            end

          {l, u}
      end

    # Transpose because since P is orthogonal, inv(P) = tranpose(P)
    # and we want to return P such that A = P.L.U
    {p |> transpose_matrix() |> matrix_to_binary(p_type), matrix_to_binary(l, l_type),
     matrix_to_binary(u, u_type)}
  end

  defp lu_validate_and_pivot(a, n) do
    # pivots a tensor so that the biggest elements of each column lie on the diagonal.
    # if any of the diagonal elements ends up being 0, raises an ArgumentError

    identity =
      Enum.map(0..(n - 1), fn i -> Enum.map(0..(n - 1), fn j -> if i == j, do: 1, else: 0 end) end)

    # For each row, find the max value by column.
    # If its index (max_idx) is not in the diagonal (i.e. j != max_idx)
    # we need to swap rows j and max_idx in both the permutation matrix
    # and in the a matrix.
    Enum.reduce(0..(n - 2), {identity, a}, fn j, {p, a} ->
      [max_idx | _] =
        Enum.sort_by(j..(n - 1), fn i -> a |> Enum.at(i) |> Enum.at(j) |> abs() end, &>=/2)

      if max_idx == j do
        {p, a}
      else
        p_row = Enum.at(p, max_idx)
        p_j = Enum.at(p, j)
        p = p |> List.replace_at(max_idx, p_j) |> List.replace_at(j, p_row)

        a_row = Enum.at(a, max_idx)
        a_j = Enum.at(a, j)
        a = a |> List.replace_at(max_idx, a_j) |> List.replace_at(j, a_row)
        {p, a}
      end
    end)
  end

  def svd(input_data, input_type, input_shape, output_type, opts) do
    # This implementation is a mixture of concepts described in [1] and the
    # algorithmic descriptions found in [2], [3] and [4]
    #
    # [1] - Parallel One-Sided Block Jacobi SVD Algorithm: I. Analysis and Design,
    #       by Gabriel Oksa and Marian Vajtersic
    #       Source: https://www.cosy.sbg.ac.at/research/tr/2007-02_Oksa_Vajtersic.pdf
    # [2] - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/client/lib/svd.cc#L784
    # [3] - https://github.com/tensorflow/tensorflow/blob/dcdc6b2f9015829cde2c02b111c04b2852687efc/tensorflow/compiler/xla/client/lib/svd.cc#L386
    # [4] - http://drsfenner.org/blog/2016/03/householder-bidiagonalization/
    # [5] - http://www.mymathlib.com/c_source/matrices/linearsystems/singular_value_decomposition.c
    a = binary_to_matrix(input_data, input_type, input_shape)

    eps = opts[:eps] || @default_eps
    max_iter = opts[:max_iter] || 1000
    {u, d, v} = householder_bidiagonalization(a, input_shape, eps)

    {fro_norm, off_diag_norm} = get_frobenius_norm(d)

    {u, s_matrix, v, _, _} =
      Enum.reduce_while(1..max_iter, {u, d, v, off_diag_norm, fro_norm}, fn
        _, {u, d, v, off_diag_norm, fro_norm} ->
          eps = 1.0e-9 * fro_norm

          if off_diag_norm > eps do
            # Execute a round of jacobi rotations on u, d and v
            {u, d, v} = svd_jacobi_rotation_round(u, d, v, input_shape, eps)

            # calculate a posteriori norms for d, so the next iteration of Enum.reduce_while can decide to halt
            {fro_norm, off_diag_norm} = get_frobenius_norm(d)

            {:cont, {u, d, v, off_diag_norm, fro_norm}}
          else
            {:halt, {u, d, v, nil, nil}}
          end
      end)

    # Make s a vector
    s =
      s_matrix
      |> Enum.with_index()
      |> Enum.map(fn {row, idx} -> Enum.at(row, idx) end)
      |> Enum.reject(&is_nil/1)

    {s, v} = apply_singular_value_corrections(s, v)

    {matrix_to_binary(u, output_type), matrix_to_binary(s, output_type),
     matrix_to_binary(v, output_type)}
  end

  defp svd_jacobi_rotation_round(u, d, v, {_, n}, eps) do
    for p <- 0..(n - 2), q <- (p + 1)..(n - 1), reduce: {u, d, v} do
      {u, d, v} ->
        # We need the values for d_bin at indices [p,p], [p,q], [q,p], [q,q], [[p,q], 0..n-1] for this first iteration
        d_rows_pq = get_matrix_rows(d, [p, q])

        [d_pp, d_pq, d_qp, d_qq] = get_matrix_elements(d, [[p, p], [p, q], [q, p], [q, q]])

        {rot_l, rot_r} = jacobi_rotators(d_pp, d_pq, d_qp, d_qq, eps)

        updated_d_rows_pq = rot_l |> transpose_matrix() |> dot_matrix(d_rows_pq)

        d = replace_rows(d, [p, q], updated_d_rows_pq)
        d_cols = get_matrix_columns(d, [p, q])

        updated_d_cols = dot_matrix(d_cols, rot_r)
        d = replace_cols(d, [p, q], updated_d_cols)

        rotated_u_cols = u |> get_matrix_columns([p, q]) |> dot_matrix(rot_l)
        u = replace_cols(u, [p, q], rotated_u_cols)

        rotated_v_cols = v |> get_matrix_columns([p, q]) |> dot_matrix(rot_r)
        v = replace_cols(v, [p, q], rotated_v_cols)

        {u, d, v}
    end
  end

  defp apply_singular_value_corrections(s, v) do
    # Due to convention, the singular values must be positive.
    # This function fixes any negative singular values.
    # It's important to note that since s left-multiplies v_transposed in
    # the SVD result. Since it also represents a diagonal matrix,
    # changing a sign in s implies a sign change in the
    # corresponding row of v_transposed.

    # This function also sorts singular values from highest to lowest,
    # as this can be convenient.

    # TODO: Use Enum.zip_with on Elixir v1.12
    s
    |> Enum.zip(transpose_matrix(v))
    |> Enum.map(fn
      {singular_value, row} when singular_value < 0 ->
        {-singular_value, Enum.map(row, &(&1 * -1))}

      {singular_value, row} ->
        {singular_value, row}
    end)
    |> Enum.sort_by(fn {s, _} -> s end, &>=/2)
    |> Enum.unzip()
  end

  ## Householder helpers

  defp householder_reflector(a, target_k, eps)

  defp householder_reflector([], target_k, _eps) do
    flat_list =
      for col <- 0..(target_k - 1), row <- 0..(target_k - 1), into: [] do
        if col == row, do: 1, else: 0
      end

    Enum.chunk_every(flat_list, target_k)
  end

  defp householder_reflector([a_0 | tail] = a, target_k, eps) do
    # This is a trick so we can both calculate the norm of a_reverse and extract the
    # head a the same time we reverse the array
    # receives a_reverse as a list of numbers and returns the reflector as a
    # k x k matrix

    norm_a_squared = Enum.reduce(a, 0, fn x, acc -> x * x + acc end)
    norm_a_sq_1on = norm_a_squared - a_0 * a_0

    {v, scale} =
      if norm_a_sq_1on < eps do
        {[1 | tail], 0}
      else
        v_0 =
          if a_0 <= 0 do
            a_0 - :math.sqrt(norm_a_squared)
          else
            -norm_a_sq_1on / (a_0 + :math.sqrt(norm_a_squared))
          end

        v_0_sq = v_0 * v_0
        scale = 2 * v_0_sq / (norm_a_sq_1on + v_0_sq)
        v = [1 | Enum.map(tail, &(&1 / v_0))]
        {v, scale}
      end

    prefix_threshold = target_k - length(v)
    v = List.duplicate(0, prefix_threshold) ++ v

    # dot(v, v) = norm_v_squared, which can be calculated from norm_a as:
    # norm_v_squared = norm_a_squared - a_0^2 + v_0^2

    # execute I - 2 / norm_v_squared * outer(v, v)
    {_, _, reflector_reversed} =
      for col_factor <- v, row_factor <- v, reduce: {0, 0, []} do
        {row, col, acc} ->
          # The current element in outer(v, v) is given by col_factor * row_factor
          # and the current I element is 1 when row == col
          identity_element = if row == col, do: 1, else: 0

          result =
            if row >= prefix_threshold and col >= prefix_threshold do
              identity_element -
                scale * col_factor * row_factor
            else
              identity_element
            end

          acc = [result | acc]

          if col + 1 == target_k do
            {row + 1, 0, acc}
          else
            {row, col + 1, acc}
          end
      end

    # This is equivalent to reflector_reversed |> Enum.reverse() |> Enum.chunk_every(target_k)
    {reflector, _, _} =
      for x <- reflector_reversed, reduce: {[], [], 0} do
        {result_acc, row_acc, col} ->
          row_acc = [x | row_acc]

          if col + 1 == target_k do
            {[row_acc | result_acc], [], 0}
          else
            {result_acc, row_acc, col + 1}
          end
      end

    reflector
  end

  defp householder_bidiagonalization(tensor, {m, n}, eps) do
    # For each column in `tensor`, apply
    # the current column's householder reflector from the left to `tensor`.
    # if the current column is not the penultimate, also apply
    # the corresponding row's householder reflector from the right

    for col <- 0..(n - 1), reduce: {nil, tensor, nil} do
      {ll, a, rr} ->
        # a[[col..m-1, col]] -> take `m - col` rows from the `col`-th column
        row_length = if m < col, do: 0, else: m - col
        a_col = a |> slice_matrix([col, col], [row_length, 1])

        l = householder_reflector(a_col, m, eps)

        ll =
          if is_nil(ll) do
            l
          else
            dot_matrix(ll, l)
          end

        a = dot_matrix(l, a)

        {a, rr} =
          if col <= n - 2 do
            # r = householder_reflector(a[[col, col+1:]], n)

            r =
              a
              |> slice_matrix([col, col + 1], [1, n - col])
              |> householder_reflector(n, eps)

            rr =
              if is_nil(rr) do
                r
              else
                dot_matrix(r, rr)
              end

            a = dot_matrix(a, r)
            {a, rr}
          else
            {a, rr}
          end

        {ll, a, rr}
    end
  end

  defp get_frobenius_norm(tensor) do
    # returns a tuple with {frobenius_norm, off_diagonal_norm}
    {fro_norm_sq, off_diag_norm_sq, _row_idx} =
      Enum.reduce(tensor, {0, 0, 0}, fn row, {fro_norm_sq, off_diag_norm_sq, row_idx} ->
        {fro_norm_sq, off_diag_norm_sq, _} =
          Enum.reduce(row, {fro_norm_sq, off_diag_norm_sq, 0}, fn x,
                                                                  {fro_norm_sq, off_diag_norm_sq,
                                                                   col_idx} ->
            if col_idx == row_idx do
              {fro_norm_sq + x * x, off_diag_norm_sq, col_idx + 1}
            else
              {fro_norm_sq + x * x, off_diag_norm_sq + x * x, col_idx + 1}
            end
          end)

        {fro_norm_sq, off_diag_norm_sq, row_idx + 1}
      end)

    {:math.sqrt(fro_norm_sq), :math.sqrt(off_diag_norm_sq)}
  end

  defp jacobi_rotators(pp, pq, qp, qq, eps) do
    t = pp + qq
    d = qp - pq

    {s, c} =
      if abs(d) < eps do
        {0, 1}
      else
        u = t / d
        den = :math.sqrt(1 + u * u)
        {-1 / den, u / den}
      end

    [[m00, m01], [_, m11]] = dot_matrix([[c, s], [-s, c]], [[pp, pq], [qp, qq]])
    {c_r, s_r} = make_jacobi(m00, m11, m01, eps)
    c_l = c_r * c - s_r * s
    s_l = c_r * s + s_r * c

    rot_l = [[c_l, s_l], [-s_l, c_l]]
    rot_r = [[c_r, s_r], [-s_r, c_r]]

    {rot_l, rot_r}
  end

  defp make_jacobi(pp, qq, pq, eps) do
    if abs(pq) <= eps do
      {1, 0}
    else
      tau = (qq - pp) / (2 * pq)

      t =
        if tau >= 0 do
          1 / (tau + :math.sqrt(1 + tau * tau))
        else
          -1 / (-tau + :math.sqrt(1 + tau * tau))
        end

      c = 1 / :math.sqrt(1 + t * t)
      {c, t * c}
    end
  end

  ## Matrix (2-D array) manipulation

  defp dot_matrix([], _), do: 0
  defp dot_matrix(_, []), do: 0

  defp dot_matrix([h1 | _] = v1, [h2 | _] = v2) when not is_list(h1) and not is_list(h2) do
    v1
    |> Enum.zip(v2)
    |> Enum.reduce(0, fn {x, y}, acc -> x * y + acc end)
  end

  defp dot_matrix(m1, m2) do
    # matrix multiplication which works on 2-D lists
    Enum.map(m1, fn row ->
      m2
      |> transpose_matrix()
      |> Enum.map(fn col ->
        row
        |> Enum.zip(col)
        |> Enum.reduce(0, fn {x, y}, acc -> acc + x * y end)
      end)
    end)
  end

  defp transpose_matrix([x | _] = m) when not is_list(x) do
    Enum.map(m, &[&1])
  end

  defp transpose_matrix(m) do
    m
    |> Enum.zip()
    |> Enum.map(&Tuple.to_list/1)
  end

  defp matrix_to_binary([r | _] = m, type) when is_list(r) do
    match_types [type] do
      for row <- m, number <- row, into: "", do: <<write!(number, 0)>>
    end
  end

  defp matrix_to_binary(list, type) when is_list(list) do
    match_types [type] do
      for number <- list, into: "", do: <<write!(number, 0)>>
    end
  end

  defp binary_to_vector(bin, type) do
    match_types [type] do
      for <<match!(x, 0) <- bin>>, into: [], do: read!(x, 0)
    end
  end

  defp binary_to_matrix(bin, type, {_, num_cols}) do
    bin
    |> binary_to_vector(type)
    |> Enum.chunk_every(num_cols)
  end

  defp slice_vector(a, start, length), do: Enum.slice(a, start, length)

  defp slice_matrix(a, [row_start, col_start], [row_length, col_length]) do
    a
    |> Enum.slice(row_start, row_length)
    |> Enum.flat_map(&Enum.slice(&1, col_start, col_length))
  end

  defp get_matrix_column(m, col) do
    Enum.map(m, fn row ->
      Enum.at(row, col)
    end)
  end

  defp get_matrix_columns(m, columns) do
    Enum.map(m, fn row ->
      for {item, i} <- Enum.with_index(row), i in columns, do: item
    end)
  end

  defp get_matrix_rows(m, rows) do
    for {row, i} <- Enum.with_index(m), i in rows, do: row
  end

  defp get_matrix_elements(m, row_col_pairs) do
    Enum.map(row_col_pairs, fn [row, col] ->
      m
      |> Enum.at(row, [])
      |> Enum.at(col)
      |> case do
        nil -> raise ArgumentError, "invalid index [#{row},#{col}] for matrix"
        item -> item
      end
    end)
  end

  defp replace_rows(m, rows, values) do
    rows
    |> Enum.zip(values)
    |> Enum.reduce(m, fn {idx, row}, m -> List.replace_at(m, idx, row) end)
  end

  defp replace_cols(m, [], _), do: m

  defp replace_cols(m, cols, values) do
    m
    |> transpose_matrix()
    |> replace_rows(cols, transpose_matrix(values))
    |> transpose_matrix()
  end

  defp replace_vector_element(m, row, value), do: List.replace_at(m, row, value)
end
