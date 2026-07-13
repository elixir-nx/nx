# Nx.LinAlg (Torchx)

Torchx-specific notes for `Nx.LinAlg` where behavior diverges from the portable
Nx API or from other backends.

Several linear algebra operations are **not** accelerated on Apple MPS
(`:mps`). On that device, Torchx falls back to Nx's default implementation for
LU, Eigh, Solve, Determinant, and Cholesky. `qr/2`, `svd/2`, and
`triangular_solve/3` temporarily move inputs to CPU, run LibTorch there, and
move results back to MPS.

Cholesky results may differ slightly from `Nx.BinaryBackend` due to LibTorch
numerics and rounding conventions.

## `Nx.LinAlg.solve/2`

Before solving, Torchx checks whether `|det(a)|` is below a dtype-dependent
epsilon (`1.0e-4` for `f32`, `1.0e-10` for `f64`). Singular systems raise
`ArgumentError` with `"can't solve for singular matrix"` instead of failing
with an opaque native error.

## `Nx.LinAlg.qr/2`

  * `:mode` — honored (`:reduced` or `:complete`)
  * `:eps` — ignored on the native LibTorch path (used only by Nx's default
    implementation on fallback paths)

QR factors are not unique (for example sign flips). Prefer checking that
`q · r` reconstructs the input within the dtype's floating-point tolerance
rather than comparing `q` and `r` bit-for-bit across backends.

## `Nx.LinAlg.eigh/2`

  * `:max_iter` and `:eps` — honored only by the default implementation (MPS
    path); ignored by the LibTorch kernel

Eigenvectors are not unique (sign and degenerate subspaces). Compare results
using reconstruction `v · diag(λ) · vᵀ` rather than direct eigenvector equality
across backends.

## `Nx.LinAlg.svd/2`

Complex SVD is **not** supported — raises for complex inputs.

  * `:full_matrices?` — honored
  * `:max_iter` — ignored on the native LibTorch path (LibTorch uses its own
    algorithm)

Results may vary across devices and precision settings. Singular values are the
most stable quantity to compare across backends.

## `Nx.LinAlg.triangular_solve/3`

  * `:lower` — honored
  * `:transform_a` — `:none` and `:transpose` are supported
  * `:left_side` — **only `true` is supported**; `left_side: false` raises
    `ArgumentError`

Uses the same singularity check as `Nx.LinAlg.solve/2` before calling LibTorch.
