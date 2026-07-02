# Nx.LinAlg (Torchx)

```elixir
iex> a = Nx.tensor([[4.0, 2.0], [2.0, 3.0]])
iex> l = Nx.LinAlg.cholesky(a)
iex> Nx.shape(l)
{2, 2}
```

Torchx implementation notes for `Nx.LinAlg`.

Each function below documents how Torchx handles the corresponding `Nx.LinAlg`
operation when `Torchx.Backend` is the active backend.

Block-tagged operations are dispatched through `Torchx.Backend.block/4`. When
Torchx does not provide a native LibTorch path for a block, it invokes the
default `Nx.block/4` callback, which composes elementary Nx operations on the
same backend.

Several linear algebra blocks are **not** supported natively on Apple MPS (`:mps`).
On that device, Torchx falls back to the default Nx block implementation instead
of calling LibTorch for LU, Eigh, Solve, Determinant, and Cholesky. `qr/2` and
`svd/2` copy inputs to CPU, run the native LibTorch kernel, and copy results
back to MPS.

## cholesky/1

Cholesky decomposition (`%Nx.Block.LinAlg.Cholesky{}`).

### Lowering

On **CPU** and **CUDA**, Torchx calls `Torchx.cholesky/1` (LibTorch
`torch::linalg_cholesky`).

On **MPS**, the default Nx callback is used because Cholesky is not in the
native MPS path.

### Types

Follows Nx promotion to floating point in the public API. LibTorch receives
the tensor's Torch scalar type (`:float`, `:double`, `:complex`, etc.).

### Numerical notes

Results may differ slightly from `Nx.BinaryBackend` due to LibTorch numerics
and rounding (half-to-even vs Elixir's half-away-from-zero elsewhere in Torchx).

## solve/2

Linear system solve (`%Nx.Block.LinAlg.Solve{}`).

### Lowering

On **CPU** and **CUDA**, Torchx calls `Torchx.solve/2` (LibTorch
`torch::linalg_solve`).

On **MPS**, the default Nx callback is used.

### Singular matrices

Before calling LibTorch, Torchx checks whether `|det(a)|` is below a dtype-
dependent epsilon (`1.0e-4` for `f32`, `1.0e-10` for `f64`). Singular systems
raise `ArgumentError` with `"can't solve for singular matrix"` instead of
letting LibTorch throw an opaque error.

### Types

Operands are cast to a merged floating-point Torch type before the solve.

## qr/2

QR decomposition (`%Nx.Block.LinAlg.QR{}`).

### Lowering

Torchx calls `Torchx.qr/2`, mapping `mode: :reduced` to LibTorch's reduced QR
flag.

On **MPS**, inputs are copied to CPU, QR is computed there, and `{q, r}` are
copied back to MPS.

### Options

* `:mode` — `:reduced` or `:complete`, passed to LibTorch
* `:eps` — ignored (used only by the default Nx callback on MPS fallback
paths for other blocks; QR always uses LibTorch when not on MPS)

### Numerical notes

`q` and `r` are not guaranteed to match other backends bit-for-bit. Verify
via reconstruction `q · r ≈ input`.

## eigh/2

Hermitian eigendecomposition (`%Nx.Block.LinAlg.Eigh{}`).

### Lowering

On **CPU** and **CUDA**, integer inputs are promoted to `f32` before calling
`Torchx.eigh/1` (LibTorch `torch::linalg_eigh`).

On **MPS**, the default Nx callback is used.

### Options

* `:max_iter` and `:eps` — honoured only by the default Nx callback (MPS
path); ignored by the LibTorch kernel

### Numerical notes

Eigenvectors are not unique. Prefer comparing reconstructions over direct
eigenvector equality.

## svd/2

Singular value decomposition (`%Nx.Block.LinAlg.SVD{}`).

### Lowering

Torchx calls `Torchx.svd/2`, passing `full_matrices?` to LibTorch.

On **MPS**, inputs are copied to CPU, SVD is computed there, and outputs are
copied back to MPS.

Integer inputs are promoted to `f32` before the native call.

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

### Lowering

On **CPU** and **CUDA**, integer inputs are promoted to `f32` before calling
`Torchx.lu/1` (LibTorch `torch::linalg_lu`).

On **MPS**, the default Nx callback is used.

### Types

Permutation `p` keeps integer semantics from LibTorch; `l` and `u` are
floating point.

## determinant/1

Matrix determinant (`%Nx.Block.LinAlg.Determinant{}`).

### Lowering

On **CPU** and **CUDA**, integer inputs are promoted to `f32` before calling
`Torchx.determinant/1`.

On **MPS**, the default Nx callback is used.

### Types

Output dtype follows LibTorch / Nx promotion rules for the input.

## triangular_solve/3

Triangular solve (direct `triangular_solve` callback, not an `Nx.block/4`).

### Lowering

Torchx calls `Torchx.triangular_solve/4` (LibTorch
`torch::linalg_solve_triangular`).

On **MPS**, `a` and `b` are transferred to CPU for the solve; the result is
transferred back to MPS.

### Options

* `:lower` — honoured (`upper` flag inverted for LibTorch)
* `:transform_a` — `:none` and `:transpose` are supported
* `:left_side` — **only `true` is supported**; `left_side: false` raises
`ArgumentError`

### Singular matrices

Same determinant-based singularity check as `solve/2` before calling LibTorch.

### Numerical notes

Batched inputs are reshaped to `{batch, m, m}` and `{batch, m, n}` internally
when needed.
