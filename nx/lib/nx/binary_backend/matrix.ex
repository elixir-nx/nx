defmodule Nx.BinaryBackend.Matrix do
  @moduledoc false
  import Nx.Shared

  def ts(a_data, a_type, b_data, b_type, shape, output_type, input_opts) do
    a_shape =
      case shape do
        {rows} -> {rows, rows}
        shape -> shape
      end

    a_matrix = binary_to_matrix(a_data, a_type, a_shape)

    b_matrix_or_vec =
      case shape do
        {rows, rows} ->
          binary_to_matrix(b_data, b_type, shape)

        {_rows} ->
          binary_to_vector(b_data, b_type)
      end

    a_matrix
    |> ts_matrix(b_matrix_or_vec, shape, input_opts)
    |> matrix_to_binary(output_type)
  end

  defp ts_matrix(a_matrix, b_matrix_or_vec, shape, input_opts) do
    transform_a = input_opts[:transform_a]
    lower_input = input_opts[:lower]

    # if transform_a != none, the upper will become lower and vice-versa
    lower = transform_a == :none == lower_input

    opts = %{
      lower: lower,
      left_side: input_opts[:left_side]
    }

    a_matrix =
      a_matrix
      |> ts_transform_a(transform_a)
      |> ts_handle_opts(opts, :a)

    b_matrix_or_vec = ts_handle_opts(b_matrix_or_vec, opts, :b)

    a_matrix
    |> do_ts(b_matrix_or_vec, shape)
    |> ts_handle_opts(opts, :result)
  end

  # For ts_handle_opts/3, we need some theoretical proofs:

  # When lower: false, we need the following procedure
  # for reusing the lower_triangular engine:
  #
  # First, we need to reverse both rows and colums
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

  defp ts_handle_opts(
         matrix,
         %{lower: true, left_side: true},
         _matrix_type
       ) do
    # Base case (lower: true, left_side: true)
    matrix
  end

  defp ts_handle_opts(
         matrix,
         %{lower: false, left_side: true},
         matrix_type
       ) do
    # lower: false, left_side: true
    # We need to follow the row-col reversing procedure
    case matrix_type do
      :a ->
        matrix
        |> Enum.map(&Enum.reverse/1)
        |> Enum.reverse()

      _ ->
        Enum.reverse(matrix)
    end
  end

  defp ts_handle_opts(
         [row_or_elem | _] = matrix,
         %{lower: lower, left_side: false},
         matrix_type
       ) do
    # left_side: false
    # transpose both sides of the equation (yielding X_t as the result)
    # We need to treat the transposed result as equivalent to lower: not lower, left_side: true,
    # (not lower) because the triangular matrix is transposed

    new_opts = %{lower: not lower, transform_a: :none, left_side: true}

    case matrix_type do
      :a ->
        matrix
        |> transpose_matrix()
        |> ts_handle_opts(new_opts, :a)

      :b when is_list(row_or_elem) ->
        matrix
        |> transpose_matrix()
        |> ts_handle_opts(new_opts, :b)

      :b ->
        ts_handle_opts(matrix, new_opts, :b)

      :result when is_list(row_or_elem) and lower ->
        matrix
        |> Enum.reverse()
        |> transpose_matrix()

      :result when is_list(row_or_elem) and not lower ->
        transpose_matrix(matrix)

      :result when lower ->
        Enum.reverse(matrix)

      :result when not lower ->
        matrix
    end
  end

  defp ts_transform_a(matrix, :transpose), do: transpose_matrix(matrix)
  defp ts_transform_a(matrix, _), do: matrix

  defp do_ts(a_matrix, b_matrix, {rows, rows}) do
    Enum.uniq(1..rows)
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

    if value == 0 do
      raise ArgumentError, "can't solve for singular matrix"
    end

    y = (b - dot_matrix(row, acc)) / value
    do_ts(rows, bs, idx + 1, acc ++ [y])
  end

  defp do_ts([], [], _idx, acc), do: acc

  def qr(input_data, input_type, input_shape, output_type, m, k, n, opts) do
    {_, input_num_bits} = input_type

    mode = opts[:mode]
    eps = opts[:eps]

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

    {q_matrix, r_matrix} = qr_matrix(r_matrix, m, k, n, eps)

    {matrix_to_binary(q_matrix, output_type), matrix_to_binary(r_matrix, output_type)}
  end

  defp qr_matrix(a_matrix, m, k, n, eps) do
    {q_matrix, r_matrix} =
      for i <- 0..(n - 1), reduce: {nil, a_matrix} do
        {q, r} ->
          a = slice_matrix(r, [i, i], [k - i, 1])

          h = householder_reflector(a, k)

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

    {approximate_zeros(q_matrix, eps), approximate_zeros(r_matrix, eps)}
  end

  def lu(input_data, input_type, {n, n} = input_shape, p_type, l_type, u_type, opts) do
    a = binary_to_matrix(input_data, input_type, input_shape)
    eps = opts[:eps]

    {p, a_prime} = pivot_matrix(a, n)

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

                value = (a_ij - sum) / u_jj

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

  defp pivot_matrix(a, n) do
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

    eps = opts[:eps]
    max_iter = opts[:max_iter] || 1000
    {u, d, v} = householder_bidiagonalization(a, input_shape)

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

    {u |> approximate_zeros(eps) |> matrix_to_binary(output_type),
     s |> approximate_zeros(eps) |> matrix_to_binary(output_type),
     v |> approximate_zeros(eps) |> matrix_to_binary(output_type)}
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
    s
    |> Enum.zip_with(transpose_matrix(v), fn
      singular_value, row when singular_value < 0 ->
        {-singular_value, Enum.map(row, &(&1 * -1))}

      singular_value, row ->
        {singular_value, row}
    end)
    |> Enum.sort_by(fn {s, _} -> s end, &>=/2)
    |> Enum.unzip()
  end

  ## Householder helpers

  defp householder_reflector(a, target_k)

  defp householder_reflector([], target_k) do
    identity_matrix(target_k, target_k)
  end

  defp householder_reflector([a_0 | tail] = a, target_k) do
    # This is a trick so we can both calculate the norm of 'a' and extract its
    # head at the same time we reverse the array.
    # Receives 'a' as a list of numbers and returns the reflector as a
    # k x k matrix

    norm_a_squared = Enum.reduce(a, 0, fn x, acc -> x * x + acc end)
    norm_a_sq_1on = norm_a_squared - a_0 * a_0

    {v, scale} =
      if norm_a_sq_1on == 0 do
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

  defp householder_bidiagonalization(tensor, {m, n}) do
    # For each column in `tensor`, apply
    # the current column's householder reflector from the left to `tensor`.
    # if the current column is not the penultimate, also apply
    # the corresponding row's householder reflector from the right

    for col <- 0..(n - 1), reduce: {nil, tensor, nil} do
      {ll, a, rr} ->
        # a[[col..m-1, col]] -> take `m - col` rows from the `col`-th column
        row_length = if m < col, do: 0, else: m - col
        a_col = a |> slice_matrix([col, col], [row_length, 1])

        l = householder_reflector(a_col, m)

        ll =
          if is_nil(ll) do
            l
          else
            dot_matrix(ll, l)
          end

        a = dot_matrix(l, a)

        {a, rr} =
          if col <= n - 2 do
            r =
              a
              |> slice_matrix([col, col + 1], [1, n - col])
              |> householder_reflector(n)

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

  def eigen(input_data, input_type, {n, n} = input_shape, output_type, opts) do
    # This decomposition is only implemented for square matrices
    # Implementation based on the algorithm described in:
    # [1] - http://www.math.iit.edu/~fass/477577_Chapter_11.pdf
    # [2] - http://article.sciencepublishinggroup.com/html/10.11648.j.pamj.20160504.15.html

    eps = opts[:eps]
    max_iter = opts[:max_iter]

    {h, _q} =
      input_data
      |> binary_to_matrix(input_type, input_shape)
      |> hessenberg_decomposition(input_shape)

    identity = identity_matrix(n, n)

    eigen_val_upper_tri =
      Enum.reduce_while(1..max_iter, h, fn _, a_prev ->
        # Coefficient for eigenvalue shifting as described in [1]
        [c_next] = get_matrix_elements(a_prev, [[n - 1, n - 1]])

        # if the last diagonal element is "zero", we add 'eps' so we can de-stabilize
        # the matrix from a fixed point. Otherwise, spectral shifting is removed by said "zero" element
        c_next =
          cond do
            c_next >= 0 and c_next < eps -> c_next + eps
            c_next < 0 and c_next > -eps -> c_next - eps
            true -> c_next
          end

        # As described in [1] and [2], we want to calculate
        # Q.R = A_prev - c_next.I
        # A_next = R.Q + c_next.I

        shifting_matrix = element_wise_bin_op(c_next, identity, &*/2)
        shifted_matrix = element_wise_bin_op(a_prev, shifting_matrix, &-/2)

        {q_current, r_current} = qr_matrix(shifted_matrix, n, n, n, eps)

        a_next = r_current |> dot_matrix(q_current) |> element_wise_bin_op(shifting_matrix, &+/2)

        if is_approximately_upper_triangular?(a_next, eps) do
          {:halt, a_next}
        else
          {:cont, a_next}
        end
      end)

    diag_indices = for idx <- 0..(n - 1), do: [idx, idx]

    eigenvals = eigen_val_upper_tri |> get_matrix_elements(diag_indices) |> Enum.sort_by(&abs/1)

    eigenvecs =
      eigenvals
      |> Enum.map(&calculate_eigenvector(&1, h, {n, n}, eps))
      |> transpose_matrix()

    {matrix_to_binary(eigenvals, output_type), matrix_to_binary(eigenvecs, output_type)}
  end

  defp hessenberg_decomposition(a, {n, n}) do
    # Calculates the Hessenberg decomposition of a square matrix `a`
    # Returns the transformed hessenberg matrix `hess` and the invertible
    # linear transformation `t` which defines is such that:
    # Hess = T . A . T_transposed
    identity = identity_matrix(n, n)

    for col <- 0..(n - 2), reduce: {a, identity} do
      {a, linear_transform} ->
        reflector = a |> slice_matrix([col + 1, col], [n - col, 1]) |> householder_reflector(n)

        {dot_matrix(reflector, a) |> dot_matrix(transpose_matrix(reflector)),
         dot_matrix(reflector, linear_transform)}
    end
  end

  defp calculate_eigenvector(eigenvalue, matrix, {n, n}, eps) do
    # We want to solve A.v = lambda.v for v, which is equivalent to
    # (A - lambda.I).v = 0

    eye = identity_matrix(n, n)
    lambda_eye = element_wise_bin_op(eye, eigenvalue + 10 * eps, &*/2)

    matrix
    |> element_wise_bin_op(lambda_eye, &-/2)
    |> solve_matrix({n, n}, zeros_matrix(n, 1), eps)
  end

  defp is_approximately_upper_triangular?(matrix, eps) do
    matrix
    |> Enum.with_index()
    |> Enum.all?(fn {row, row_idx} ->
      row
      |> Enum.with_index()
      |> Enum.all?(fn {matrix_elem, col_idx} ->
        col_idx >= row_idx or matrix_elem <= eps
      end)
    end)
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
    Enum.zip_with(m, & &1)
  end

  defp element_wise_bin_op(a, b, fun) when is_list(a) and is_list(b) do
    # Applies a binary operation element-wise between two matrices with the same shape
    Enum.zip_with([a, b], fn
      [a_row, b_row] when is_list(a_row) and is_list(b_row) ->
        Enum.zip_with([a_row, b_row], fn [a_elem, b_elem] -> fun.(a_elem, b_elem) end)

      [a_elem, b_elem] when is_number(a_elem) and is_number(b_elem) ->
        fun.(a_elem, b_elem)
    end)
  end

  defp element_wise_bin_op(a, b, fun) when is_number(a) and is_list(b),
    do: element_wise_bin_op(b, a, &fun.(&2, &1))

  defp element_wise_bin_op(a, b, fun) when is_list(a) and is_number(b) do
    # Applies a binary operation element-wise between a matrix and a scalar
    Enum.map(a, fn
      a_row when is_list(a_row) ->
        Enum.map(a_row, fn a_elem -> fun.(a_elem, b) end)

      a_elem ->
        fun.(a_elem, b)
    end)
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

  defp get_matrix_columns(m, columns) do
    Enum.map(m, fn row ->
      for {item, i} <- Enum.with_index(row), i in columns, do: item
    end)
  end

  defp get_matrix_rows(m, rows) do
    for idx <- rows, do: Enum.at(m, idx)
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

  defp replace_matrix_element(m, row, col, value) do
    updated = m |> Enum.at(row) |> List.replace_at(col, value)
    List.replace_at(m, row, updated)
  end

  defp approximate_zeros(matrix, tol) do
    do_round = fn x -> if abs(x) < tol, do: 0, else: x end

    Enum.map(matrix, fn
      row when is_list(row) -> Enum.map(row, do_round)
      e -> do_round.(e)
    end)
  end

  defp identity_matrix(rows, cols) do
    for row <- 0..(rows - 1) do
      for col <- 0..(cols - 1) do
        if row == col, do: 1, else: 0
      end
    end
  end

  defp zeros_matrix(rows, cols) do
    for _row <- 0..(rows - 1) do
      for _col <- 0..(cols - 1) do
        0
      end
    end
  end

  defp solve_matrix(a, {n, n}, [[_] | _] = b, eps) do
    # Solves A.x = B through gauss-jordan elimination
    # For indeterminate systems, any zero-line will be
    # forced to return x_n = 1

    # Construct the augmented system matrix and pivot it accordingly
    {_p, system_matrix} =
      a
      |> Enum.zip_with(b, &++/2)
      |> pivot_matrix(n)
      |> IO.inspect(label: "nx/lib/nx/binary_backend/matrix.ex:890")

    # Transform into upper triangular
    upper_triangular_system_matrix =
      system_matrix
      |> solve_matrix_iteration(
        0..(n - 1),
        &(&1..(n - 1)),
        fn row, pivot ->
          element_wise_bin_op(row, pivot, &//2)
        end,
        eps
      )
      |> IO.inspect(label: "nx/lib/nx/binary_backend/matrix.ex:903")
      |> solve_matrix_iteration(
        (n - 1)..1//-1,
        &(&1..0//-1),
        fn row, _pivot ->
          row
        end,
        eps
      )
      |> IO.inspect(label: "nx/lib/nx/binary_backend/matrix.ex:911")

    # For each row, ensure the system is solvable

    placeholder_row = List.duplicate(0, n) ++ [1]

    {system_matrix_reversed, updated_idx_reversed} =
      upper_triangular_system_matrix
      |> Enum.with_index()
      |> Enum.reduce({[], []}, fn {row, idx}, {rows, indices} ->
        if Enum.all?(row, &(abs(&1) <= eps)) do
          # For x_idx = 1
          row = List.replace_at(placeholder_row, idx, 1)

          new_indices = if idx > 0, do: [idx | indices], else: indices
          {[row | rows], new_indices}
        else
          {[row | rows], indices}
        end
      end)

    # Apply second backward pass for newly defined rows.
    # The resulting system will be fully diagonalized and normalized,
    # which implies that the solution is in the last column.
    system_matrix_reversed
    |> Enum.reverse()
    |> solve_matrix_iteration(
      updated_idx_reversed,
      &(&1..0//-1),
      fn row, _pivot ->
        row
      end,
      eps
    )
    |> get_matrix_column(n)
  end

  defp solve_matrix_iteration(
         system_matrix,
         iteration_range,
         update_range_function,
         transform_pivot_row_function,
         eps
       ) do
    for idx <- iteration_range, reduce: system_matrix do
      system_matrix ->
        # Find pivot
        [pivot] = get_matrix_elements(system_matrix, [[idx, idx]])

        if abs(pivot) >= eps do
          upd_row_idx = update_range_function.(idx)
          [pivot_row | other_rows] = get_matrix_rows(system_matrix, upd_row_idx)

          normalized_pivot_row = transform_pivot_row_function.(pivot_row, pivot)

          updated_rows =
            Enum.map(other_rows, fn row ->
              coef = Enum.at(row, idx)

              # upd_row = row - coef * normalized_pivot_row
              normalized_pivot_row
              |> element_wise_bin_op(-coef, &*/2)
              |> element_wise_bin_op(row, &+/2)
              |> approximate_zeros(eps)
            end)

          replace_rows(system_matrix, upd_row_idx, [normalized_pivot_row | updated_rows])
        else
          # Pivot is zero, so we skip this iteration
          system_matrix
        end
    end
  end
end
