# Nx.LinAlg (EXLA)

```elixir
iex> a = Nx.tensor([[4.0, 2.0], [2.0, 3.0]])
iex> l = Nx.LinAlg.cholesky(a)
iex> Nx.shape(l)
{2, 2}
```

EXLA implementation notes for `Nx.LinAlg`.

Each function below documents how EXLA handles the corresponding `Nx.LinAlg`
operation when `compiler: EXLA` is used.

Operations that use `Nx.block/4` are dispatched through `EXLA.Defn`. When
`EXLA.CustomCall.call/4` returns `:skip`, EXLA compiles the default block
callback as an XLA subcomputation. When it returns `{:ok, spec}`, EXLA emits a
StableHLO `custom_call` instead.

On GPU and TPU clients, the `:precision` option passed to `Nx.Defn.jit/2`
affects accumulator precision for many composed linear algebra graphs compiled
from the default callbacks. For decomposition verification tests, prefer
`precision: :highest`.

## cholesky/1

Cholesky decomposition (`%Nx.Block.LinAlg.Cholesky{}`).

### Lowering

EXLA does **not** provide a native custom call for Cholesky. The default
callback (`Nx.LinAlg.Cholesky.cholesky/1`) is compiled as an XLA
subcomputation.

### Supported platforms

All EXLA clients (`:host`, `:cuda`, `:rocm`, `:tpu`) use the compiled default
implementation.

### Types

Follows Nx promotion rules: integer and unsigned inputs are promoted to
floating point in the public API before the block is invoked.

### Numerical notes

Positive-definiteness is not validated at compile time. Ill-conditioned inputs
may produce `NaN` values without raising, consistent with the reference Nx
implementation.

## solve/2

Linear system solve (`%Nx.Block.LinAlg.Solve{}`).

### Lowering

EXLA compiles the default callback, which expresses `solve/2` as an LU
factorization followed by triangular solves.

### Supported platforms

All EXLA clients use the compiled default implementation.

### Types

Output type is floating point per `Nx.LinAlg.solve/2`. The solve block depends
on `lu/1` and `triangular_solve/3` inside the compiled graph.

## qr/2

QR decomposition (`%Nx.Block.LinAlg.QR{}`).

### Lowering

On the **host** client, with **real** input element types, EXLA emits a
native CPU custom call via `EXLA.CustomCall` when the dtype is supported:

* `{:f, 32}` → `qr_cpu_custom_call_f32`
* `{:f, 64}` → `qr_cpu_custom_call_f64`
* `{:f, 16}` → `qr_cpu_custom_call_f16`
* `{:bf, 16}` → `qr_cpu_custom_call_bf16`
* `{:s, _}` and `{:u, _}` → `qr_cpu_custom_call_f32` with operand cast to
`f32`

Otherwise EXLA compiles the default callback (`Nx.LinAlg.QR.qr/2`).

### Supported platforms

* **Host** — native custom call for supported real dtypes; default callback
for complex inputs and unsupported types
* **CUDA / ROCm / TPU** — default callback only (no native QR custom call)

### Options

* `:mode` — honoured by whichever lowering is selected (`:reduced` or
`:complete`)
* `:eps` — used by the default callback only; ignored by the native custom
call

### Numerical notes

Native and default lowerings may differ slightly in orthogonality of `q` and
triangular structure of `r` within floating-point tolerance. Reconstruction
`q · r` should match the input within reasonable tolerances for the dtype.

## eigh/2

Hermitian eigendecomposition (`%Nx.Block.LinAlg.Eigh{}`).

### Lowering

On the **host** client, with **real** input element types, EXLA emits a
native CPU custom call when the dtype is supported:

* `{:f, 32}` → `eigh_cpu_custom_call_f32`
* `{:f, 64}` → `eigh_cpu_custom_call_f64`
* `{:s, _}` and `{:u, _}` → `eigh_cpu_custom_call_f32` with operand cast to
`f32`

Otherwise EXLA compiles the default callback (`Nx.LinAlg.BlockEigh.eigh/2` or
`Nx.LinAlg.Eigh.eigh/2` depending on input rank).

### Supported platforms

* **Host** — native custom call for supported real dtypes; default callback
for complex inputs and unsupported types
* **CUDA / ROCm / TPU** — default callback only

### Options

* `:max_iter` and `:eps` — honoured by the default callback; ignored by the
native custom call

### Numerical notes

Eigenvectors are not unique (sign and degenerate subspaces). Compare results
using reconstruction `v · diag(λ) · vᵀ` rather than direct eigenvector equality
across backends.

## svd/2

Singular value decomposition (`%Nx.Block.LinAlg.SVD{}`).

### Lowering

EXLA compiles the default callback (`Nx.LinAlg.SVD.svd/2`). There is no native
SVD custom call.

### Supported platforms

All EXLA clients use the compiled default implementation.

### Options

* `:max_iter` — honoured (default iterative algorithm in Nx)
* `:full_matrices?` — honoured

### Numerical notes

SVD is iterative; results may vary across devices and precision settings.
Singular values are the most stable quantity to compare across backends.
Use `precision: :highest` on GPU when verifying decompositions.

## lu/1

LU decomposition with partial pivoting (`%Nx.Block.LinAlg.LU{}`).

### Lowering

EXLA compiles the default callback (`Nx.LinAlg.LU.lu/1`) as an XLA
subcomputation. Permutation matrix `p` uses integer (`s32`) type; `l` and `u`
are floating point.

### Supported platforms

All EXLA clients use the compiled default implementation.

### Numerical notes

For rank-deficient inputs, `u` may contain zeros on the diagonal. This matches
the reference Nx behaviour.

## determinant/1

Matrix determinant (`%Nx.Block.LinAlg.Determinant{}`).

### Lowering

EXLA compiles the default callback. Small matrices use closed-form expressions;
larger matrices use LU-based determinant computation inside the compiled graph.

### Supported platforms

All EXLA clients use the compiled default implementation.

### Types

Output is floating point. Complex determinants are supported when the input is
complex.

## triangular_solve/3

Triangular solve (direct `triangular_solve` callback, not an `Nx.block/4`).

### Lowering

Lowered to the StableHLO `triangular_solve` op via `EXLA.MLIR.Value.triangular_solve/4`.

### Supported platforms

All EXLA clients.

### Options

EXLA honours `lower:`, `left_side:`, and `transform_a:` as defined in
`Nx.LinAlg.triangular_solve/3`.
