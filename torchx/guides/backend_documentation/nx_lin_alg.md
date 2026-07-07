# Nx.LinAlg (Torchx)

Torchx-specific notes for `Nx.LinAlg` where behaviour diverges from the portable
Nx API or from other backends.

Several linear algebra blocks are **not** supported natively on Apple MPS (`:mps`).
On that device, Torchx falls back to the default Nx block implementation instead
of calling LibTorch for LU, Eigh, Solve, Determinant, and Cholesky. `qr/2` and
`svd/2` copy inputs to CPU, run the native LibTorch kernel, and copy results
back to MPS.

## cholesky/1

Cholesky decomposition (`%Nx.Block.LinAlg.Cholesky{}`).

### Platforms

On **MPS**, the default Nx callback is used because Cholesky is not in the
native MPS path.

### Numerical notes

Results may differ slightly from `Nx.BinaryBackend` due to LibTorch numerics
and rounding (half-to-even vs Elixir's half-away-from-zero elsewhere in Torchx).

## solve/2

Linear system solve (`%Nx.Block.LinAlg.Solve{}`).

### Platforms

On **MPS**, the default Nx callback is used.

### Singular matrices

Before calling LibTorch, Torchx checks whether `|det(a)|` is below a dtype-
dependent epsilon (`1.0e-4` for `f32`, `1.0e-10` for `f64`). Singular systems
raise `ArgumentError` with `"can't solve for singular matrix"` instead of
letting LibTorch throw an opaque error.

## qr/2

QR decomposition (`%Nx.Block.LinAlg.QR{}`).

### Platforms

On **MPS**, inputs are copied to CPU, QR is computed there, and `{q, r}` are
copied back to MPS.

### Options

* `:mode` — `:reduced` or `:complete`, passed to LibTorch
* `:eps` — ignored (used only by the default Nx callback on MPS fallback paths)

### Numerical notes

`q` and `r` are not guaranteed to match other backends bit-for-bit. Verify
via reconstruction `q · r ≈ input`.

## eigh/2

Hermitian eigendecomposition (`%Nx.Block.LinAlg.Eigh{}`).

### Platforms

On **MPS**, the default Nx callback is used.

### Options

* `:max_iter` and `:eps` — honoured only by the default Nx callback (MPS
path); ignored by the LibTorch kernel

### Numerical notes

Eigenvectors are not unique. Prefer comparing reconstructions over direct
eigenvector equality.

## svd/2

Singular value decomposition (`%Nx.Block.LinAlg.SVD{}`).

### Platforms

On **MPS**, inputs are copied to CPU, SVD is computed there, and outputs are
copied back to MPS.

### Types

Complex SVD is **not** supported — `Nx.LinAlg.svd/2` raises for complex inputs
on Torchx.

### Options

* `:full_matrices?` — honoured by LibTorch
* `:max_iter` — ignored (LibTorch uses its own algorithm; the default Nx
iterative implementation is not used on the native path)

### Numerical notes

Singular values are the most stable quantity to compare across backends.

## lu/1

LU decomposition with partial pivoting (`%Nx.Block.LinAlg.LU{}`).

### Platforms

On **MPS**, the default Nx callback is used.

## determinant/1

Matrix determinant (`%Nx.Block.LinAlg.Determinant{}`).

### Platforms

On **MPS**, the default Nx callback is used.

## triangular_solve/3

Triangular solve (direct `triangular_solve` callback, not an `Nx.block/4`).

### Platforms

On **MPS**, `a` and `b` are transferred to CPU for the solve; the result is
transferred back to MPS.

### Options

* `:lower` — honoured (`upper` flag inverted for LibTorch)
* `:transform_a` — `:none` and `:transpose` are supported
* `:left_side` — **only `true` is supported**; `left_side: false` raises
`ArgumentError`

### Singular matrices

Same determinant-based singularity check as `solve/2` before calling LibTorch.
