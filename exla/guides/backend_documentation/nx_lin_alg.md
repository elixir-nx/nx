# Nx.LinAlg (EXLA)

EXLA-specific notes for `Nx.LinAlg` where behavior diverges from the portable
Nx API or from other backends.

On GPU and TPU, the `:precision` option passed to `Nx.Defn.jit/2` affects
accumulator precision for linear algebra compiled from Nx's default
implementations. Prefer `precision: :highest` when verifying decompositions.

## `Nx.LinAlg.qr/2`

On the **host** client, EXLA provides a native path for supported real dtypes
(`f16`, `bf16`, `f32`, `f64`, and integer types cast to `f32`). Complex inputs
and unsupported types use Nx's default implementation. On **CUDA**, **ROCm**,
and **TPU**, only the default implementation is available.

  * `:mode` — honored (`:reduced` or `:complete`)
  * `:eps` — used only by the default implementation; ignored on the native
    host path

QR factors are not unique (for example sign flips). Prefer checking that
`q · r` reconstructs the input within the dtype's floating-point tolerance
rather than comparing `q` and `r` bit-for-bit across backends.

## `Nx.LinAlg.eigh/2`

On the **host** client, EXLA provides a native path for supported real dtypes
(`f32`, `f64`, and integer types cast to `f32`). Complex inputs and unsupported
types use Nx's default implementation. On **CUDA**, **ROCm**, and **TPU**, only
the default implementation is available.

  * `:max_iter` and `:eps` — honored by the default implementation; ignored on
    the native host path

Eigenvectors are not unique (sign and degenerate subspaces). Compare results
using reconstruction `v · diag(λ) · vᵀ` rather than direct eigenvector equality
across backends.

## `Nx.LinAlg.svd/2`

Uses Nx's iterative SVD. Both `:max_iter` and `:full_matrices?` are honored.

Results may vary across devices and precision settings. Singular values are the
most stable quantity to compare across backends. Use `precision: :highest` on
GPU when verifying decompositions.
