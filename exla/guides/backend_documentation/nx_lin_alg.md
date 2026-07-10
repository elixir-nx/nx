# Nx.LinAlg (EXLA)

EXLA-specific notes for `Nx.LinAlg` where behaviour diverges from the portable
Nx API or from other backends.

On GPU and TPU clients, the `:precision` option passed to `Nx.Defn.jit/2`
affects accumulator precision for many composed linear algebra graphs compiled
from the default callbacks. For decomposition verification tests, prefer
`precision: :highest`.

## qr/2

QR decomposition (`%Nx.Block.LinAlg.QR{}`).

### Supported platforms

* **Host** — native CPU custom call for supported **real** dtypes (`f16`,
`bf16`, `f32`, `f64`, and integer types cast to `f32`); default callback for
complex inputs and unsupported types
* **CUDA / ROCm / TPU** — default callback only (no native QR custom call)

### Options

* `:mode` — honored by whichever implementation is selected (`:reduced` or
`:complete`)
* `:eps` — used by the default callback only; ignored by the native custom call

### Numerical notes

Native and default implementations may differ slightly in orthogonality of `q`
and triangular structure of `r` within floating-point tolerance.
For example, for an eigenpair `(lambda, v)`, `(-lambda, -v)` is
an equally valid eigenpair that would lead to radically different
`q` and `r` matrices. Reconstruction of `q · r` should match the
input within reasonable tolerances for the dtype.

## eigh/2

Hermitian eigendecomposition (`%Nx.Block.LinAlg.Eigh{}`).

### Supported platforms

* **Host** — native CPU custom call for supported **real** dtypes (`f32`,
`f64`, and integer types cast to `f32`); default callback for complex inputs and
unsupported types
* **CUDA / ROCm / TPU** — default callback only

### Options

* `:max_iter` and `:eps` — honored by the default callback; ignored by the
native custom call

### Numerical notes

Eigenvectors are not unique (sign and degenerate subspaces). Compare results
using reconstruction `v · diag(λ) · vᵀ` rather than direct eigenvector equality
across backends.

## svd/2

Singular value decomposition (`%Nx.Block.LinAlg.SVD{}`).

### Options

* `:max_iter` — honored (default iterative algorithm in Nx)
* `:full_matrices?` — honored

### Numerical notes

SVD is iterative; results may vary across devices and precision settings.
Singular values are the most stable quantity to compare across backends.
Use `precision: :highest` on GPU when verifying decompositions.
