# Nx (Torchx)

Torchx-specific notes for top-level `Nx` operations where behavior diverges from
the portable Nx API or from other backends.

Linear algebra is documented separately in [Nx.LinAlg](backend_documentation-nx_lin_alg.html).

Torchx is an **Nx backend**, not a `defn` compiler. Operations run eagerly on
LibTorch (or through another compiler, such as `Nx.Defn.Evaluator`, that
invokes backend callbacks).

## `Nx.fft2/2`

Numerical results for signed zeros (and therefore some `inspect` output) may
differ slightly from `Nx.BinaryBackend`. Prefer comparing magnitudes when
checking FFT results across backends.

## `Nx.runtime_call/4`

Outside of a `defn`, Torchx executes the callback directly with tensors on
`Torchx.Backend`.

Inside a `defn`, Torchx does not compile the code — behavior depends on the
chosen compiler (for example `Nx.Defn.Evaluator` or `EXLA`). There is no
device-side Elixir callback path on Torchx itself.

## `Nx.backend_copy/2`

Copies a Torchx tensor to another backend or Torchx device. Copying within
Torchx moves data to the requested device; copying out materializes a binary
on the destination backend.

## `Nx.backend_transfer/2`

Like `Nx.backend_copy/2`, then deallocates the source tensor.

## `Nx.to_pointer/2`

**Not supported.** Raises on `Torchx.Backend`.

## `Nx.from_pointer/5`

**Not supported.** Raises on `Torchx.Backend`.
