# Nx.LinAlg (EXLA)

On GPU and TPU, results can be less accurate unless you pass
`precision: :highest` to `Nx.Defn.jit/2`.

## `Nx.LinAlg.qr/2`

On the host, `:eps` is ignored for real matrices. On GPU and TPU it works as
documented in Nx.

## `Nx.LinAlg.eigh/2`

On the host, `:max_iter` and `:eps` are ignored for real matrices. On GPU and
TPU they work as documented in Nx.
