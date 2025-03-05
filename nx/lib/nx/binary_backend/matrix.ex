defmodule Nx.BinaryBackend.Matrix do
  @moduledoc false
  use Complex.Kernel

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

  defp get_matrix_column(m, col) do
    Enum.map(m, fn row ->
      Enum.at(row, col)
    end)
  end
end
