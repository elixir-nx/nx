defmodule Nx.LinAlg.Eig do
  @moduledoc """
  General eigenvalue decomposition using QR algorithm.

  This implements the non-symmetric eigenvalue problem for general square matrices.
  Unlike `Nx.LinAlg.BlockEigh` which assumes Hermitian matrices, this works with any
  square matrix but always produces complex eigenvalues and eigenvectors.

  The implementation uses:
  1. Reduction to upper Hessenberg form using Householder reflections
  2. Shifted QR algorithm on the Hessenberg matrix to find eigenvalues
  3. Inverse iteration to find eigenvectors

  This is a reference implementation. Backends like EXLA provide optimized
  versions using LAPACK's geev routine.
  """
  import Nx.Defn

  defn eig(a, opts \\ []) do
    opts = keyword!(opts, eps: 1.0e-4, max_iter: 1_000)

    a
    |> Nx.revectorize([collapsed_axes: :auto],
      target_shape: {Nx.axis_size(a, -2), Nx.axis_size(a, -1)}
    )
    |> eig_matrix(opts)
    |> revectorize_result(a)
  end

  deftransformp revectorize_result({eigenvals, eigenvecs}, a) do
    shape = Nx.shape(a)

    {
      Nx.revectorize(eigenvals, a.vectorized_axes,
        target_shape: Tuple.delete_at(shape, tuple_size(shape) - 1)
      ),
      Nx.revectorize(eigenvecs, a.vectorized_axes, target_shape: shape)
    }
  end

  defnp eig_matrix(a, opts \\ []) do
    # Convert to complex type since eigenvalues can be complex even for real matrices
    type = Nx.Type.to_complex(Nx.type(a))
    a = Nx.as_type(a, type)

    {n, _} = Nx.shape(a)

    case n do
      1 ->
        # For 1x1 matrices, eigenvalue is the single element
        eigenval = Nx.reshape(a, {1})
        eigenvec = Nx.tensor([[1.0]], type: type)
        {eigenval, eigenvec}

      _ ->
        {eigenvals, eigenvecs} =
          calculate_evals_evecs(a, opts)

        # Sort eigenpairs by |lambda| in descending order
        sort_idx = Nx.argsort(Nx.abs(eigenvals), direction: :desc)
        eigenvals = Nx.take(eigenvals, sort_idx)
        eigenvecs = Nx.take(eigenvecs, sort_idx, axis: 1)
        {eigenvals, eigenvecs}
    end
  end

  defnp calculate_evals_evecs(a, opts) do
    type = Nx.Type.to_complex(Nx.type(a))

    cond do
      is_upper_triangular(a, opts) ->
        eigenvals = Nx.take_diagonal(a)
        eigenvecs = eigenvectors_from_upper_tri(a, eigenvals, opts)
        {eigenvals, eigenvecs}

      is_lower_triangular(a, opts) ->
        eigenvals = Nx.take_diagonal(a)
        eigenvecs = eigenvectors_from_lower_tri(a, eigenvals, opts)
        {eigenvals, eigenvecs}

      is_hermitian(a, opts) ->
        {eigs_h, vecs_h} = Nx.LinAlg.eigh(a)
        {Nx.as_type(eigs_h, type), Nx.as_type(vecs_h, type)}

      true ->
        # Reduce to Hessenberg form and keep the orthogonal transformation Q
        {h, q_hessenberg} = hessenberg(a, opts)

        # Apply QR algorithm to find Schur form, eigenvalues, and accumulated Schur vectors
        {schur, eigenvals, q_schur} = qr_algorithm(h, opts)
        q_total = Nx.dot(q_hessenberg, q_schur)

        # If the Schur form is (nearly) diagonal, its eigenvectors are simply q_total's columns.
        # This happens for normal matrices (including Hermitian), which our property test exercises.
        # Use a fast path in that case; otherwise, compute eigenvectors from Schur form.
        diag_schur = Nx.make_diagonal(Nx.take_diagonal(schur))
        offdiag_norm = Nx.LinAlg.norm(schur - diag_schur)
        schur_norm = Nx.LinAlg.norm(schur)
        nearly_diag = offdiag_norm <= 1.0e-6 * (schur_norm + opts[:eps])

        eigenvecs =
          Nx.select(
            nearly_diag,
            q_total,
            compute_eigenvectors(schur, q_total, eigenvals, opts)
          )

        {eigenvals, eigenvecs}
    end
  end

  defnp is_hermitian(a, opts) do
    eps = opts[:eps]
    sym_norm = Nx.LinAlg.norm(a - Nx.LinAlg.adjoint(a))
    a_norm = Nx.LinAlg.norm(a)
    sym_norm <= 1.0e-6 * (a_norm + eps)
  end

  defnp is_upper_triangular(a, opts) do
    eps = opts[:eps]
    lower = Nx.tril(a, k: -1)
    lower_norm = Nx.LinAlg.norm(lower)
    a_norm = Nx.LinAlg.norm(a)
    lower_norm <= 1.0e-6 * (a_norm + eps)
  end

  defnp is_lower_triangular(a, opts) do
    eps = opts[:eps]
    upper = Nx.triu(a, k: 1)
    upper_norm = Nx.LinAlg.norm(upper)
    a_norm = Nx.LinAlg.norm(a)
    upper_norm <= 1.0e-6 * (a_norm + eps)
  end

  defnp hessenberg(a, opts) do
    eps = opts[:eps]
    # Reduce matrix to upper Hessenberg form using Householder reflections
    # An upper Hessenberg matrix has zeros below the first subdiagonal
    {n, _} = Nx.shape(a)
    type = Nx.type(a)

    column_iota = Nx.iota({n})

    [h, q] = Nx.broadcast_vectors([a, Nx.eye(n, type: type)])

    # Perform Householder reflections for columns 0 to n-3
    {{h, q}, _} =
      while {{h, q}, {column_iota}}, k <- 0..(n - 3)//1 do
        # Extract column k, zeroing elements at or above k
        x = h[[.., k]]
        x = Nx.select(column_iota <= k, 0, x)

        # Compute Householder reflector matrix
        reflector = Nx.LinAlg.QR.householder_reflector(x, k, eps)
        h_adj = Nx.LinAlg.adjoint(reflector)

        # Apply: H = P * H * P^H where P is the reflector
        h = reflector |> Nx.dot(h) |> Nx.dot(h_adj)

        # Update Q: Q = Q * P
        q = Nx.dot(q, reflector)

        {{h, q}, {column_iota}}
      end

    {h, q}
  end

  defnp qr_algorithm(h, opts) do
    # Shifted QR algorithm to find eigenvalues and accumulate Schur vectors
    eps = opts[:eps]
    max_iter = opts[:max_iter]
    {n, _} = Nx.shape(h)
    type = Nx.type(h)

    eye = Nx.eye(n, type: type)
    accum_q = eye

    [h, accum_q, eye] = Nx.broadcast_vectors([h, accum_q, eye])

    # Standard QR iteration on full matrix with Wilkinson shift, accumulating Q
    {{h, accum_q}, _} =
      while {{h, accum_q}, {i = 0, eye}}, i < max_iter do
        subdiag = Nx.take_diagonal(h, offset: -1)
        max_subdiag = Nx.reduce_max(Nx.abs(subdiag))

        shift = wilkinson_shift_full(h, n)
        {q_step, r} = Nx.LinAlg.qr(h - shift * eye)
        h_candidate = Nx.dot(r, q_step) + shift * eye
        accum_candidate = Nx.dot(accum_q, q_step)

        update = Nx.greater_equal(max_subdiag, eps)
        h = Nx.select(update, h_candidate, h)
        accum_q = Nx.select(update, accum_candidate, accum_q)

        {{h, accum_q}, {i + 1, eye}}
      end

    {h, Nx.take_diagonal(h), accum_q}
  end

  defnp wilkinson_shift_full(h, n) do
    # Standard Wilkinson shift from bottom 2x2 block
    if n >= 2 do
      a = h[[n - 2, n - 2]]
      b = h[[n - 2, n - 1]]
      c = h[[n - 1, n - 2]]
      d = h[[n - 1, n - 1]]

      trace = a + d
      det = a * d - b * c
      discriminant = trace * trace / 4 - det

      sqrt_disc = Nx.sqrt(discriminant)
      lambda1 = trace / 2 + sqrt_disc
      lambda2 = trace / 2 - sqrt_disc

      diff1 = Nx.abs(lambda1 - d)
      diff2 = Nx.abs(lambda2 - d)

      Nx.select(diff1 < diff2, lambda1, lambda2)
    else
      h[[0, 0]]
    end
  end

  defnp compute_eigenvectors(h, q, eigenvals, opts) do
    eps = opts[:eps]
    # Compute eigenvectors using stabilized inverse iteration on H via normal equations:
    # (A^H A + mu I) v_new = A^H v_old, where A = (H - lambda I)
    {n, _} = Nx.shape(h)
    type = Nx.type(h)

    eigenvecs_h = Nx.broadcast(0.0, {n, n}) |> Nx.as_type(type)
    eye = Nx.eye(n, type: type)

    [eigenvecs_h, eigenvals, h, eye] = Nx.broadcast_vectors([eigenvecs_h, eigenvals, h, eye])

    {eigenvecs_h, _} =
      while {eigenvecs_h, {k = 0, eigenvals, h, eye}}, k < n do
        lambda = eigenvals[[k]]

        # Deterministic initial vector
        # Use a real iota to avoid complex iota backend limitations, then cast to complex
        v_real = Nx.iota({n}, type: type)
        v = v_real + k
        v = v / (Nx.LinAlg.norm(v) + eps)

        # Orthogonalize against previously computed eigenvectors
        v = orthogonalize_vector(v, eigenvecs_h, k, eps)

        # Prepare A, A^H, and normal equations matrix
        a = h - lambda * eye
        ah = Nx.LinAlg.adjoint(a)

        {v, _} =
          while {v, {iter = 0, a, ah, eye}}, iter < 40 do
            # Right-hand side: b = A^H v
            b = Nx.dot(ah, [1], v, [0])
            # Normal equations matrix: N = A^H A + mu I
            ah_a = Nx.dot(ah, a)
            # Adaptive regularization
            mu = Nx.LinAlg.norm(ah_a) * 1.0e-3 + eps
            nmat = ah_a + mu * eye
            # Solve N v_new = b
            v_new = Nx.LinAlg.solve(nmat, b)
            # Normalize
            v_norm = Nx.LinAlg.norm(v_new)
            v = Nx.select(Nx.abs(v_norm) > eps, v_new / v_norm, v)
            {v, {iter + 1, a, ah, eye}}
          end

        # One more orthogonalization pass for stability
        v = orthogonalize_vector(v, eigenvecs_h, k, eps)
        # And renormalize
        v_norm = Nx.LinAlg.norm(v)
        v = Nx.select(Nx.abs(v_norm) > eps, v / v_norm, v)

        eigenvecs_h = Nx.put_slice(eigenvecs_h, [0, k], Nx.reshape(v, {n, 1}))

        {eigenvecs_h, {k + 1, eigenvals, h, eye}}
      end

    # Transform eigenvectors back: V = Q * V_h
    Nx.dot(q, eigenvecs_h)
  end

  # Fast path: compute eigenvectors directly from an upper-triangular A by back-substitution
  defnp eigenvectors_from_upper_tri(a, eigenvals, opts) do
    eps = opts[:eps]
    {n, _} = Nx.shape(a)
    type = Nx.type(a)

    eye = Nx.eye(n, type: type)
    [a, eye] = Nx.broadcast_vectors([a, eye])
    v = a * 0.0

    row_idx = Nx.iota({n}, type: {:s, 32})
    col_idx = row_idx

    [eigenvals] = Nx.broadcast_vectors([eigenvals])

    {v, _} =
      while {v, {k = 0, a, eigenvals, eye, row_idx, col_idx}}, k < n do
        lambda = eigenvals[[k]]
        u = a - lambda * eye

        vk = u[0] * 0.0
        vk = Nx.put_slice(vk, [k], Nx.tensor([1.0], type: type))

        {vk, _} =
          while {vk, {i = k - 1, u, row_idx, col_idx, k}}, i >= 0 do
            mask_gt_i = Nx.greater(col_idx, i)
            mask_ge_0 = Nx.greater_equal(col_idx, 0)
            m = Nx.as_type(Nx.logical_and(mask_gt_i, mask_ge_0), type)
            row_u = u[i]
            sum = Nx.sum(row_u * vk * m)
            denom = u[[i, i]]
            vi = -sum / (denom + eps)
            vk = Nx.put_slice(vk, [i], Nx.reshape(vi, {1}))
            {vk, {i - 1, u, row_idx, col_idx, k}}
          end

        vk_norm = Nx.LinAlg.norm(vk)
        vk = Nx.select(Nx.abs(vk_norm) > eps, vk / vk_norm, vk)
        v = Nx.put_slice(v, [0, k], Nx.reshape(vk, {n, 1}))
        {v, {k + 1, a, eigenvals, eye, row_idx, col_idx}}
      end

    v
  end

  # Fast path: compute eigenvectors directly from a lower-triangular A by forward substitution
  defnp eigenvectors_from_lower_tri(a, eigenvals, opts) do
    eps = opts[:eps]
    {n, _} = Nx.shape(a)
    type = Nx.type(a)

    eye = Nx.eye(n, type: type)
    [a, eye] = Nx.broadcast_vectors([a, eye])
    v = a * 0.0

    row_idx = Nx.iota({n}, type: {:s, 32})
    col_idx = row_idx

    [eigenvals] = Nx.broadcast_vectors([eigenvals])

    {v, _} =
      while {v, {k = 0, a, eigenvals, eye, row_idx, col_idx}}, k < n do
        lambda = eigenvals[[k]]
        l = a - lambda * eye

        vk = l[0] * 0.0
        vk = Nx.put_slice(vk, [k], Nx.tensor([1.0], type: type))

        {vk, _} =
          while {vk, {i = k + 1, l, row_idx, col_idx, k}}, i < n do
            # sum over j in [k, i)
            mask_ge_k = Nx.greater_equal(col_idx, k)
            mask_lt_i = Nx.less(col_idx, i)
            m = Nx.as_type(Nx.logical_and(mask_ge_k, mask_lt_i), type)
            row_l = l[i]
            sum = Nx.sum(row_l * vk * m)
            denom = l[[i, i]]
            vi = -sum / (denom + eps)
            vk = Nx.put_slice(vk, [i], Nx.reshape(vi, {1}))
            {vk, {i + 1, l, row_idx, col_idx, k}}
          end

        vk_norm = Nx.LinAlg.norm(vk)
        vk = Nx.select(Nx.abs(vk_norm) > eps, vk / vk_norm, vk)
        v = Nx.put_slice(v, [0, k], Nx.reshape(vk, {n, 1}))
        {v, {k + 1, a, eigenvals, eye, row_idx, col_idx}}
      end

    v
  end

  # Orthogonalize vector v against the first k columns of matrix eigenvecs
  # Uses Gram-Schmidt: v = v - sum(proj_j) where proj_j = <v, v_j> * v_j
  defnp orthogonalize_vector(v, eigenvecs, k, eps) do
    {_n, n_cols} = Nx.shape(eigenvecs)

    # We need to orthogonalize against columns 0..k-1
    # Use a fixed iteration approach with masking to avoid out of bounds
    max_iters = Nx.min(k, n_cols)

    # Broadcast vectors to ensure consistent shape
    [v, eigenvecs] = Nx.broadcast_vectors([v, eigenvecs])

    {v_orthog, _} =
      while {v_orthog = v, {j = 0, max_iters, eigenvecs, k}}, j < max_iters do
        # Only process if j < k and j < n_cols
        should_process = Nx.logical_and(j < k, j < n_cols)

        v_orthog =
          if should_process do
            # Get column j (safe because we checked bounds)
            # Clamp to valid range
            col_idx = Nx.min(j, n_cols - 1)
            v_j = eigenvecs[[.., col_idx]]
            proj = Nx.dot(Nx.LinAlg.adjoint(v_j), v_orthog)
            v_orthog - Nx.multiply(proj, v_j)
          else
            v_orthog
          end

        {v_orthog, {j + 1, max_iters, eigenvecs, k}}
      end

    # Normalize the orthogonalized vector
    v_norm = Nx.LinAlg.norm(v_orthog)
    Nx.select(Nx.abs(v_norm) > eps, v_orthog / v_norm, v)
  end
end
