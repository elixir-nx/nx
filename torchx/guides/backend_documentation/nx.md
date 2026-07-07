# Nx (Torchx)

Torchx-specific notes for top-level `Nx` operations where behaviour diverges from
the portable Nx API or from other backends.

Linear algebra is documented separately in [Nx.LinAlg](backend_documentation-nx_lin_alg.html).

Torchx is an **Nx backend**, not a `defn` compiler. Block lowerings run through
`Torchx.Backend.block/4` during eager execution or when another compiler
(such as `Nx.Defn.Evaluator`) invokes backend callbacks.

## fft2/2

Two-dimensional FFT (`%Nx.Block.FFT2{}`).

### Numerical notes

Signed zeros and `inspect` formatting may differ from `Nx.BinaryBackend`; see
`Torchx.NxBlockTest` for numerical comparisons.

## phase/1

Complex phase (`%Nx.Block.Phase{}`).

### Behaviour

No dedicated LibTorch wrapper — Torchx invokes the default Nx block callback.

## runtime_call/4

Runtime Elixir callback from `defn`.

### Behaviour

Torchx does not provide a `defn` compiler. Outside `defn`, `runtime_call/4`
executes the callback directly with tensors on `Torchx.Backend`. Inside
`defn`, behaviour depends on the chosen compiler (`Nx.Defn.Evaluator` keeps
tensors on the active backend; EXLA has its own lowering).

There is no LibTorch integration for device-side Elixir callbacks on Torchx.

## backend_copy/2

Copy tensor data to another backend.

### Behaviour

Copying within `Torchx.Backend` uses `Torchx.to_device/2`. Copying to another
backend reads a binary blob via `Torchx.to_blob/1` and calls
`backend.from_binary/3`.

## backend_transfer/2

Transfer tensor data to another backend.

### Behaviour

`backend_copy/2` followed by `backend_deallocate/1` on the source tensor.

## to_pointer/2

Export tensor memory as a pointer.

### Behaviour

**Not supported.** `to_pointer/2` raises on `Torchx.Backend`.

## from_pointer/5

Import tensor memory from a pointer.

### Behaviour

**Not supported.** `from_pointer/5` raises on `Torchx.Backend`.
