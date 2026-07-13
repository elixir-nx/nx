# Nx.LinAlg (Torchx)

On Apple MPS (`:mps`), LU, Eigh, Solve, Determinant and Cholesky fall back to
Nx (slower). QR, SVD and triangular solve run on the CPU and copy the results
back.

## `Nx.LinAlg.solve/2`

Singular matrices raise `ArgumentError` with `"can't solve for singular matrix"`.

## `Nx.LinAlg.qr/2`

`:eps` is ignored.

## `Nx.LinAlg.eigh/2`

`:max_iter` and `:eps` are ignored, except on MPS where Nx's implementation is
used.

## `Nx.LinAlg.svd/2`

Complex inputs raise. `:max_iter` is ignored.

## `Nx.LinAlg.triangular_solve/3`

`left_side: false` raises. Singular matrices raise the same error as `solve/2`.
