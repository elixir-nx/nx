# Nx (Torchx)

See also [Nx.LinAlg](backend_documentation-nx_lin_alg.html).

Torchx is a backend, not a `defn` compiler. On Apple MPS, `f64` and `c128`
are not supported and raise.

## `Nx.runtime_call/4`

Outside of a `defn`, the callback runs directly on Torchx tensors. Inside a
`defn`, you need a compiler such as EXLA or `Nx.Defn.Evaluator` — Torchx does
not provide one.

## `Nx.to_pointer/2` and `Nx.from_pointer/5`

Not supported. Both raise.
