defmodule Nx.BinaryBackend.Matrix do
  @moduledoc false
  use Complex.Kernel
  import Kernel, except: [abs: 1]
  import Complex, only: [abs: 1]

  import Nx.Shared

  def ts(a_data, a_type, a_shape, b_data, b_type, b_shape, output_type, input_opts) do
    # Here's Paulo Valente's proof in the deleted function ts_handle_opts/3.

    # When lower: false, we need the following procedure
    # for reusing the lower_triangular engine:
    #
    # First, we need to reverse both rows and columns
    # so we can turn an upper-triangular matrix
    # into a lower-triangular one.
    # The result will also be reversed in this case.
    #
    # Proof:
    # For a result [x1, x2, x3.., xn] and a row [a1, a2, a4, ..., an]
    # we have the corresponding b = a1 * x1 + a2 * x2 + a3 * x3 + ...+ an * xn
    # Since the addition of a_i * x_i is commutative, by reversing the columns
    # of a, the yielded x will be reversed.
    # Furthermore, if we reverse the rows of a, we need to reverse the rows of b
    # so each row is kept together with it's corresponding result.
    #
    # For example, the system:
    # A = [[a b c], [0 d e], [0 0 f]]
    # b = [b1, b2, b3, b4]
    # which yields x = [x1, x2, x3, x4]
    # is therefore equivalent to:
    # A = [[f 0 0], [e d 0], [c b a]]
    # b = [b4, b3, b2, b1]
    # which yields [x4, x3, x2, x1]

    # For handling left_side: false
    # Let's notate the system matrix as L when it's a lower triangular matrix
    # and U when it's upper triangular
    #
    # To solve X.L = B, we can then transpose both sides:
    # transpose(X.L) = transpose(L).X_t = U.X_t = b_t
    # This equation, in turn, has the same shape (U.X = B) as the one we can solve through
    # applying `ts_handle_lower_opt` properly, which would yield X_t.
    # Transposing the result suffices for yielding the final result.

    opt_left_side = !!input_opts[:left_side]
    opt_lower = !!input_opts[:lower]
    opt_transpose_a = input_opts[:transform_a] == :transpose

    transpose_a? = opt_left_side == opt_transpose_a
    reverse_a? = opt_left_side == (opt_lower == opt_transpose_a)

    transpose_b? = not opt_left_side and match?({_, _}, b_shape)
    reverse_b? = reverse_a?

    transpose_x? = transpose_b?
    reverse_x? = reverse_b?

    a_matrix =
      a_data
      |> binary_to_matrix(a_type, a_shape)
      |> then(&if(transpose_a?, do: transpose_matrix(&1), else: &1))
      |> then(&if(reverse_a?, do: reverse_matrix(&1), else: &1))

    b_matrix_or_vec =
      case b_shape do
        {_rows, _cols} ->
          b_data |> binary_to_matrix(b_type, b_shape)

        {_rows} ->
          b_data |> binary_to_vector(b_type)
      end
      |> then(&if(transpose_b?, do: transpose_matrix(&1), else: &1))
      |> then(&if(reverse_b?, do: reverse_matrix(&1), else: &1))

    b_shape =
      case {b_shape, transpose_b?} do
        {{rows, cols}, true} -> {cols, rows}
        _ -> b_shape
      end

    result =
      a_matrix
      |> do_ts(b_matrix_or_vec, b_shape)
      |> then(&if(transpose_x?, do: transpose_matrix(&1), else: &1))
      |> then(&if(reverse_x?, do: reverse_matrix(&1), else: &1))

    matrix_to_binary(result, output_type)
  end

  defp do_ts(a_matrix, b_matrix, {_rows, cols}) do
    1..cols//1
    |> Enum.map(fn b_col ->
      b_vector = get_matrix_column(b_matrix, b_col - 1)

      do_ts(a_matrix, b_vector, 0, [])
    end)
    |> transpose_matrix()
  end

  defp do_ts(a_matrix, b_vector, {_}) do
    do_ts(a_matrix, b_vector, 0, [])
  end

  defp do_ts([row | rows], [b | bs], idx, acc) do
    value = Enum.fetch!(row, idx)

    if Complex.abs(value) == 0 do
      raise ArgumentError, "can't solve for singular matrix"
    end

    y = (b - dot_matrix(row, acc)) / value
    do_ts(rows, bs, idx + 1, acc ++ [y])
  end

  defp do_ts([], [], _idx, acc), do: acc

  def lu(input_data, input_type, {n, n} = input_shape, p_type, l_type, u_type, opts) do
    a = binary_to_matrix(input_data, input_type, input_shape)
    eps = opts[:eps]

    {p, a_prime} = lu_validate_and_pivot(a, n)

    # We'll work with linear indices because of the way each matrix
    # needs to be updated/accessed
    zeros_matrix = List.duplicate(List.duplicate(0, n), n)

    {l, u} =
      for j <- 0..(n - 1), reduce: {zeros_matrix, zeros_matrix} do
        {l, u} ->
          l = replace_matrix_element(l, j, j, 1.0)

          u =
            for i <- 0..j, reduce: u do
              u ->
                u_slice = slice_matrix(u, [0, j], [i, 1])
                l_slice = slice_matrix(l, [i, 0], [1, i])
                sum = dot_matrix(u_slice, l_slice)
                [a_ij] = get_matrix_elements(a_prime, [[i, j]])

                value = a_ij - sum

                if abs(value) < eps do
                  replace_matrix_element(u, i, j, 0)
                else
                  replace_matrix_element(u, i, j, value)
                end
            end

          l =
            for i <- j..(n - 1), i != j, reduce: l do
              l ->
                u_slice = slice_matrix(u, [0, j], [i, 1])
                l_slice = slice_matrix(l, [i, 0], [1, i])
                sum = dot_matrix(u_slice, l_slice)

                [a_ij] = get_matrix_elements(a_prime, [[i, j]])
                [u_jj] = get_matrix_elements(u, [[j, j]])

                value =
                  cond do
                    u_jj != 0 ->
                      (a_ij - sum) / u_jj

                    a_ij >= sum ->
                      :infinity

                    true ->
                      :neg_infinity
                  end

                if abs(value) < eps do
                  replace_matrix_element(l, i, j, 0)
                else
                  replace_matrix_element(l, i, j, value)
                end
            end

          {l, u}
      end

    # Transpose because since P is orthogonal, inv(P) = tranpose(P)
    # and we want to return P such that A = P.L.U
    {p |> transpose_matrix() |> matrix_to_binary(p_type),
     l |> approximate_zeros(eps) |> matrix_to_binary(l_type),
     u |> approximate_zeros(eps) |> matrix_to_binary(u_type)}
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

  ## Matrix (2-D array) manipulation

  defp dot_matrix([], _), do: 0
  defp dot_matrix(_, []), do: 0

  defp dot_matrix([h1 | _] = v1, [h2 | _] = v2) when not is_list(h1) and not is_list(h2) do
    Enum.zip_reduce(v1, v2, 0, fn x, y, acc -> x * y + acc end)
  end

  defp dot_matrix(m1, m2) do
    # matrix multiplication which works on 2-D lists
    Enum.map(m1, fn row ->
      m2
      |> transpose_matrix()
      |> Enum.map(fn col ->
        Enum.zip_reduce(row, col, 0, fn x, y, acc -> acc + x * Complex.conjugate(y) end)
      end)
    end)
  end

  defp transpose_matrix([x | _] = m) when not is_list(x) do
    Enum.map(m, &[&1])
  end

  defp transpose_matrix(m) do
    Enum.zip_with(m, & &1)
  end

  defp reverse_matrix([x | _] = m) when not is_list(x) do
    Enum.reverse(m)
  end

  defp reverse_matrix(m) do
    m
    |> Enum.map(&Enum.reverse/1)
    |> Enum.reverse()
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

  defp replace_matrix_element(m, row, col, value) do
    updated = m |> Enum.at(row) |> List.replace_at(col, value)
    List.replace_at(m, row, updated)
  end

  defp approximate_zeros(matrix, tol) do
    do_round = fn x -> if Complex.abs(x) < tol, do: 0 * x, else: x end

    Enum.map(matrix, fn
      row when is_list(row) -> Enum.map(row, do_round)
      e -> do_round.(e)
    end)
  end
end
