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
    type = Nx.Type.to_complex(Nx.Type.to_floating(Nx.type(a)))
    a = Nx.as_type(a, type)

    {n, _} = Nx.shape(a)

    if n == 1 do
      # For 1x1 matrices, eigenvalue is the single element
      eigenval = a[[0, 0]]
      eigenvec = Nx.tensor([[1.0]], type: type)
      {Nx.reshape(eigenval, {1}), eigenvec}
    else
      # Reduce to Hessenberg form and keep the orthogonal transformation Q
      {h, q} = hessenberg(a, opts)

      # Apply QR algorithm to find eigenvalues
      eigenvals = qr_algorithm(h, opts)

      # Compute eigenvectors from the Hessenberg form and transform back
      eigenvecs = compute_eigenvectors(h, q, eigenvals, opts)

      {eigenvals, eigenvecs}
    end
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
    # This is a simplified version - full implementation would use
    # Francis double shift and deflation
    eps = opts[:eps]
    max_iter = opts[:max_iter]
    {n, _} = Nx.shape(h)
    type = Nx.type(h)

    # Iterate QR decomposition with shifts
    {h, _} =
      while {h, {i = 0}}, i < max_iter do
        # Check convergence - if subdiagonal elements are small enough
        subdiag = Nx.take_diagonal(h, offset: -1)
        max_subdiag = Nx.reduce_max(Nx.abs(subdiag))

        h =
          if max_subdiag < eps do
            h
          else
            # Use Wilkinson shift - the eigenvalue of the bottom 2x2 block
            # closer to the bottom-right element
            shift = wilkinson_shift(h, n)

            # QR decomposition of (H - shift*I)
            {q, r} = Nx.LinAlg.qr(h - shift * Nx.eye(n, type: type))

            # H = R*Q + shift*I
            Nx.dot(r, q) + shift * Nx.eye(n, type: type)
          end

        {h, {i + 1}}
      end

    # Extract eigenvalues from diagonal (and handle 2x2 blocks for complex conjugate pairs)
    extract_eigenvalues(h, eps)
  end

  defnp wilkinson_shift(h, n) do
    # Compute the Wilkinson shift from the bottom 2x2 block
    if n >= 2 do
      a = h[[n - 2, n - 2]]
      b = h[[n - 2, n - 1]]
      c = h[[n - 1, n - 2]]
      d = h[[n - 1, n - 1]]

      # Eigenvalues of 2x2 block
      trace = a + d
      det = a * d - b * c
      discriminant = trace * trace / 4 - det

      # Choose eigenvalue closer to d
      sqrt_disc = Nx.sqrt(discriminant)
      lambda1 = trace / 2 + sqrt_disc
      lambda2 = trace / 2 - sqrt_disc

      diff1 = Nx.abs(lambda1 - d)
      diff2 = Nx.abs(lambda2 - d)

      Nx.select(diff1 < diff2, lambda1, lambda2)
    else
      h[[n - 1, n - 1]]
    end
  end

  defnp extract_eigenvalues(h, _eps) do
    # Extract eigenvalues from the quasi-triangular Hessenberg matrix
    # Diagonal elements are eigenvalues (possibly with small 2x2 blocks for complex pairs)
    {_n, _} = Nx.shape(h)
    _type = Nx.type(h)

    # For simplicity, just take diagonal elements
    # A more sophisticated implementation would properly handle 2x2 blocks
    eigenvals = Nx.take_diagonal(h)

    # Sort eigenvalues by magnitude (descending)
    magnitudes = Nx.abs(eigenvals)
    indices = Nx.argsort(magnitudes, direction: :desc)
    Nx.take(eigenvals, indices)
  end

  defnp compute_eigenvectors(h, q, eigenvals, opts) do
    eps = opts[:eps]
    # Compute eigenvectors using inverse iteration on the Hessenberg matrix H
    # Then transform back to original space using Q
    {n, _} = Nx.shape(h)
    type = Nx.type(h)

    # For each eigenvalue, compute corresponding eigenvector of H
    eigenvecs_h = Nx.broadcast(0.0, {n, n}) |> Nx.as_type(type)

    [eigenvecs_h, eigenvals, h] = Nx.broadcast_vectors([eigenvecs_h, eigenvals, h])

    {eigenvecs_h, _} =
      while {eigenvecs_h, {k = 0, eigenvals, h}}, k < n do
        lambda = eigenvals[[k]]

        # Solve (H - lambda*I)v = 0 using inverse iteration
        # Start with a random-like vector (using k as seed)
        v = Nx.iota({n}, type: type) |> Nx.add(k)
        v = v / Nx.LinAlg.norm(v)

        # Orthogonalize against previously computed eigenvectors using Gram-Schmidt
        # For each column j < k, subtract projection onto v_j
        v = orthogonalize_vector(v, eigenvecs_h, k, eps)

        # Inverse iteration: repeatedly solve (H - lambda*I + eps*I)v = v_old
        # This converges to the eigenvector
        shift = Nx.complex(eps, eps)
        eye = Nx.eye(n, type: type)
        h_shifted = h - lambda * eye + shift * eye

        # Perform a few iterations of inverse iteration
        [v, h_shifted] = Nx.broadcast_vectors([v, h_shifted])

        {v, _} =
          while {v, {iter = 0, h_shifted}}, iter < 10 do
            # Solve h_shifted * v_new = v using triangular solve approximation
            # Since h_shifted is close to singular, we use a regularized solve
            # For simplicity, use a few Richardson iterations

            {v_new, _} =
              while {v_new = v, {i = 0, h_shifted, v}}, i < 5 do
                residual = Nx.dot(h_shifted, [1], v_new, [0]) - v
                v_new = v_new - Nx.multiply(0.1, residual)
                {v_new, {i + 1, h_shifted, v}}
              end

            # Normalize
            v_norm = Nx.LinAlg.norm(v_new)
            v_new = Nx.select(Nx.abs(v_norm) > eps, v_new / v_norm, v)

            {v_new, {iter + 1, h_shifted}}
          end

        # Store eigenvector
        eigenvecs_h = Nx.put_slice(eigenvecs_h, [0, k], Nx.reshape(v, {n, 1}))

        {eigenvecs_h, {k + 1, eigenvals, h}}
      end

    # Transform eigenvectors back to original space: V = Q * V_h
    Nx.dot(q, eigenvecs_h)
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
            col_idx = Nx.min(j, n_cols - 1)  # Clamp to valid range
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
