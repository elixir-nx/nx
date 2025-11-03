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
    # do_sort: 1 = sort by |lambda| (default), 0 = no sorting
    opts = keyword!(opts, eps: 1.0e-4, max_iter: 1_000, do_sort: 1, balance: 1)

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
    type = Nx.Type.to_complex(Nx.Type.to_floating(Nx.type(a)))
    a = Nx.as_type(a, type)

    {n, _} = Nx.shape(a)

    case n do
      1 ->
        # For 1x1 matrices, eigenvalue is the single element
        eigenval = a[[0, 0]]
        eigenvec = Nx.tensor([[1.0]], type: type)
        {Nx.reshape(eigenval, {1}), eigenvec}

      _ ->
        # Reduce to Hessenberg form and keep the orthogonal transformation Q
        # Optionally balance the matrix for improved conditioning: ab = D^-1 * A * D
        {a_bal, dvec} =
          if opts[:balance] == 1 do
            balance(a, opts)
          else
            {a, Nx.broadcast(1.0, {n}) |> Nx.as_type(type)}
          end

        {h, q} = hessenberg(a_bal, opts)

        # Apply QR algorithm to find eigenvalues
        eigenvals = qr_algorithm(h, opts)

        # Compute eigenvectors from the Hessenberg form and transform back to balanced space
        eigenvecs_bal = compute_eigenvectors(h, q, eigenvals, opts)

        # Transform eigenvectors back to original A-space via D: V = D * V_bal
        # ab = D^-1 * A * D => right eigenvectors of A are D times eigenvectors of ab
        eigenvecs =
          if opts[:balance] == 1 do
            # Scale rows by dvec
            scale = Nx.reshape(dvec, {n, 1})
            Nx.multiply(eigenvecs_bal, scale)
          else
            eigenvecs_bal
          end

        # Pre-polish eigenvectors in A-space with the initial eigenvalues to tighten pairing
        eigenvecs = polish_eigenvectors_with_iters(a, eigenvals, eigenvecs, opts, 5)

        # Refine eigenvalues using the Rayleigh quotient with the pre-polished eigenvectors
        eigenvals = refine_eigenvalues(a, eigenvecs, eigenvals, opts)

        # Sort eigenvalues and eigenvectors in decreasing order by magnitude (optional)
        {eigenvals, eigenvecs} =
          if opts[:do_sort] == 1 do
            sort_idx = Nx.argsort(Nx.abs(eigenvals), direction: :desc)
            {Nx.take(eigenvals, sort_idx), Nx.take(eigenvecs, sort_idx, axis: 1)}
          else
            {eigenvals, eigenvecs}
          end

        # Polish eigenvectors directly in A-space to better satisfy A v ≈ λ v
        eigenvecs = polish_eigenvectors(a, eigenvals, eigenvecs, opts)

        {eigenvals, eigenvecs}
    end
  end

  # Refine eigenvalues given eigenvectors via Rayleigh quotient:
  # lambda_i = (v_i^H A v_i) / (v_i^H v_i)
  defnp refine_eigenvalues(a, eigenvecs, eigenvals_init, opts) do
    eps = opts[:eps]
    {n, _} = Nx.shape(a)
    type = Nx.type(a)

    eigenvals_ref = Nx.broadcast(0.0, {n}) |> Nx.as_type(type)

    [eigenvals_ref, a, eigenvecs, eigenvals_init] =
      Nx.broadcast_vectors([eigenvals_ref, a, eigenvecs, eigenvals_init])

    {eigenvals_ref, _} =
      while {eigenvals_ref, {k = 0, a, eigenvecs, eigenvals_init}}, k < n do
        v = eigenvecs[[.., k]]
        # Compute Av and inner products
        av = Nx.dot(a, [1], v, [0])
        num = Nx.dot(Nx.LinAlg.adjoint(v), [0], av, [0])
        den = Nx.dot(Nx.LinAlg.adjoint(v), [0], v, [0])

        # Only refine if the current vector approximately satisfies A v ≈ λ_init v
        lambda_init = eigenvals_init[[k]]
        res = Nx.LinAlg.norm(av - lambda_init * v)
        can_refine = Nx.abs(res) < 1.0e-2

        lambda_raw = num / (den + eps)
        # Safeguards: require stable denominator, decent residual, and avoid magnitude collapse
        den_ok = Nx.abs(den) > eps
        ratio_ok = Nx.abs(lambda_raw) >= 0.5 * (Nx.abs(lambda_init) + eps)
        use_raw = Nx.logical_and(Nx.logical_and(den_ok, can_refine), ratio_ok)
        lambda = Nx.select(use_raw, lambda_raw, lambda_init)

        eigenvals_ref = Nx.put_slice(eigenvals_ref, [k], Nx.reshape(lambda, {1}))
        {eigenvals_ref, {k + 1, a, eigenvecs, eigenvals_init}}
      end

    eigenvals_ref
  end

  defnp hessenberg(a, opts) do
    eps = opts[:eps]
    # Reduce matrix to upper Hessenberg form using Householder reflections
    # An upper Hessenberg matrix has zeros below the first subdiagonal
    {n, _} = Nx.shape(a)
    type = Nx.type(a)

    # Initialize Q as identity
    q = Nx.eye(n, type: type)
    h = a

    # Create index arrays once for masking
    row_idx = Nx.iota({n}, type: {:s, 32})
    col_idx = Nx.iota({n}, type: {:s, 32})

    [h, q] = Nx.broadcast_vectors([h, q])

    # Perform Householder reflections for columns 0 to n-3
    {{h, q}, _} =
      while {{h, q}, {k = 0, row_idx, col_idx}}, k < n - 2 do
        # Extract column k, masking elements at or above k
        x_full = h[[.., k]]
        mask = Nx.greater(row_idx, k)
        x = Nx.select(mask, x_full, Nx.tensor(0.0, type: type))

        # Compute Householder vector (only for elements below diagonal)
        {v_full, beta} = householder_vector(x, mask, eps)

        # Apply Householder reflection: H = I - beta * v * v^H
        # Update H: H = (I - beta*v*v^H) * H
        # v^H * H
        v_conj = Nx.conjugate(v_full)
        vh_h = Nx.dot(v_conj, [0], h, [0])
        update_h = beta * Nx.outer(v_full, vh_h)
        h = h - update_h

        # Update H: H = H * (I - beta*v*v^H)
        # H * v
        h_v = Nx.dot(h, [1], v_full, [0])
        update_h2 = beta * Nx.outer(h_v, v_conj)
        h = h - update_h2

        # Update Q: Q = Q * (I - beta*v*v^H)
        # Q * v
        q_v = Nx.dot(q, [1], v_full, [0])
        update_q = beta * Nx.outer(q_v, v_conj)
        q = q - update_q

        {{h, q}, {k + 1, row_idx, col_idx}}
      end

    {h, q}
  end

  defnp householder_vector(x, mask, eps) do
    # Compute Householder vector v and scalar beta
    # x is already masked - only elements where mask=true are non-zero
    type = Nx.type(x)
    n = Nx.size(x)

    # Compute norm only for masked elements
    norm_x = Nx.sqrt(Nx.sum(Nx.multiply(x, Nx.conjugate(x))))

    # Avoid division by zero
    norm_x = Nx.select(Nx.abs(norm_x) < eps, Nx.tensor(1.0, type: type), norm_x)

    # First non-zero element (use argmax on mask to find it)
    first_idx = Nx.argmax(mask)
    first_elem = x[[first_idx]]

    # Sign to avoid cancellation
    alpha = -Nx.sign(first_elem) * norm_x

    # Create e1 (first unit vector in the masked subspace)
    idx_range = Nx.iota({n}, type: {:s, 32})
    e1 = Nx.select(idx_range == first_idx, Nx.tensor(1.0, type: type), Nx.tensor(0.0, type: type))

    # v = x - alpha * e1 (only in masked region)
    v = Nx.select(mask, x - alpha * e1, Nx.tensor(0.0, type: type))

    # Normalize v in the masked region
    v_norm = Nx.sqrt(Nx.sum(Nx.multiply(v, Nx.conjugate(v))))
    # Convert v_norm to real for comparison (it should already be real, but make it explicit)
    v_norm_real = Nx.abs(v_norm)
    v = Nx.select(v_norm_real < eps, e1, v / (v_norm + eps))

    # beta = 2 for normalized v
    beta = Nx.tensor(2.0, type: type)

    {v, beta}
  end

  defnp qr_algorithm(h, opts) do
    # Shifted QR algorithm to find eigenvalues
    eps = opts[:eps]
    max_iter = opts[:max_iter]
    {n, _} = Nx.shape(h)
    type = Nx.type(h)

    # Standard QR iteration on full matrix with Wilkinson shift
    {h, _} =
      while {h, {i = 0}}, i < max_iter do
        subdiag = Nx.take_diagonal(h, offset: -1)
        max_subdiag = Nx.reduce_max(Nx.abs(subdiag))

        h =
          if max_subdiag < eps do
            h
          else
            shift = wilkinson_shift_full(h, n)
            {q, r} = Nx.LinAlg.qr(h - shift * Nx.eye(n, type: type))
            Nx.dot(r, q) + shift * Nx.eye(n, type: type)
          end

        {h, {i + 1}}
      end

    extract_eigenvalues(h, eps)
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

  defnp extract_eigenvalues(h, _eps) do
    # For now, just extract diagonal elements
    # TODO: Add 2x2 block handling for complex conjugate pairs
    Nx.take_diagonal(h)
  end

  # Simple matrix balancing (scaling) to improve conditioning.
  # Returns {ab, dvec} where ab = D^-1 * A * D and dvec is the diagonal of D.
  defnp balance(a, opts) do
    eps = opts[:eps]
    {n, _} = Nx.shape(a)
    type = Nx.type(a)

    dvec = Nx.broadcast(1.0, {n}) |> Nx.as_type(type)

    [a, dvec] = Nx.broadcast_vectors([a, dvec])

    {a, dvec, _} =
      while {a, dvec, {sweep = 0}}, sweep < 5 do
        {a, dvec, _} =
          while {a, dvec, {i = 0}}, i < n do
            row = Nx.sum(Nx.abs(a[i])) - Nx.abs(a[[i, i]])
            col = Nx.sum(Nx.abs(a[[.., i]])) - Nx.abs(a[[i, i]])

            # s = sqrt(col/row), clipped to [0.5, 2.0]
            s_raw = Nx.sqrt(col / (row + eps))
            s_clipped = Nx.clip(s_raw, 0.5, 2.0)

            s =
              Nx.select(
                Nx.logical_and(row > 0.0, col > 0.0),
                s_clipped,
                Nx.tensor(1.0, type: type)
              )

            # Scale row i by s
            row_i = a[i] * s
            a = Nx.put_slice(a, [i, 0], Nx.reshape(row_i, {1, n}))

            # Scale column i by 1/s
            col_i = a[[.., i]] / s
            a = Nx.put_slice(a, [0, i], Nx.reshape(col_i, {n, 1}))

            # Accumulate scaling into dvec
            dv = dvec[[i]] * s
            dvec = Nx.put_slice(dvec, [i], Nx.reshape(dv, {1}))

            {a, dvec, {i + 1}}
          end

        {a, dvec, {sweep + 1}}
      end

    {a, dvec}
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
        v = Nx.iota({n}, type: type) |> Nx.add(k)
        v = v / (Nx.LinAlg.norm(v) + eps)

        # Orthogonalize against previously computed eigenvectors
        v = orthogonalize_vector(v, eigenvecs_h, k, eps)

        # Prepare A, A^H, and normal equations matrix
        a = h - lambda * eye
        ah = Nx.LinAlg.adjoint(a)

        {v, _} =
          while {v, {iter = 0, a, ah, eye}}, iter < 20 do
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

  # Polish eigenvectors in A-space with fixed eigenvalues using normal equations
  defnp polish_eigenvectors(a, eigenvals, eigenvecs, opts) do
    polish_eigenvectors_with_iters(a, eigenvals, eigenvecs, opts, 25)
  end

  # Variant with configurable iteration count for pre- or post-polish
  defnp polish_eigenvectors_with_iters(a, eigenvals, eigenvecs, opts, iters) do
    eps = opts[:eps]
    {n, _} = Nx.shape(a)
    type = Nx.type(a)

    eye = Nx.eye(n, type: type)
    [a, eye, eigenvals, eigenvecs] = Nx.broadcast_vectors([a, eye, eigenvals, eigenvecs])

    {eigenvecs, _} =
      while {eigenvecs, {k = 0, a, eye, eigenvals}}, k < n do
        lambda = eigenvals[[k]]
        v = eigenvecs[[.., k]]

        a_shift = a - lambda * eye
        ah = Nx.LinAlg.adjoint(a_shift)

        {v, _} =
          while {v, {iter = 0, a_shift, ah, eye}}, iter < iters do
            b = Nx.dot(ah, [1], v, [0])
            ah_a = Nx.dot(ah, a_shift)
            mu = Nx.LinAlg.norm(ah_a) * 1.0e-4 + eps
            nmat = ah_a + mu * eye
            v_new = Nx.LinAlg.solve(nmat, b)
            v_norm = Nx.LinAlg.norm(v_new)
            v = Nx.select(Nx.abs(v_norm) > eps, v_new / v_norm, v)
            {v, {iter + 1, a_shift, ah, eye}}
          end

        # Optional light re-orthogonalization against previously polished vectors
        v = orthogonalize_vector(v, eigenvecs, k, eps)
        v_norm = Nx.LinAlg.norm(v)
        v = Nx.select(Nx.abs(v_norm) > eps, v / v_norm, v)

        eigenvecs = Nx.put_slice(eigenvecs, [0, k], Nx.reshape(v, {n, 1}))

        {eigenvecs, {k + 1, a, eye, eigenvals}}
      end

    eigenvecs
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
      while {v_orthog = v, {j = 0, max_iters, eigenvecs, k}}, j < 5 do
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
