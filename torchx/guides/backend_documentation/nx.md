# Nx (Torchx)

```elixir
iex> Nx.take(Nx.tensor([10, 20, 30]), Nx.tensor([0, 2]))
#Nx.Tensor<
  s32[2]
  [10, 30]
>
```

Torchx implementation notes for top-level `Nx` operations.

Each function below documents how Torchx handles the corresponding `Nx` API
when `Torchx.Backend` is active.

Linear algebra is documented separately in [Nx.LinAlg](backend_documentation-nx_lin_alg.html).

Torchx is an **Nx backend**, not a `defn` compiler. Block lowerings run through
`Torchx.Backend.block/4` during eager execution or when another compiler
(such as `Nx.Defn.Evaluator`) invokes backend callbacks.

## take/3

Tensor indexing (`%Nx.Block.Take{}`).

### Lowering

No native LibTorch path — Torchx invokes the default Nx block callback, which
composes `Nx` gather operations on the Torchx backend.

### Options

* `:axis` — honoured by the default callback

## take_along_axis/3

Take along axis (`%Nx.Block.TakeAlongAxis{}`).

### Lowering

Native `Torchx.gather/2` along the configured axis (LibTorch `gather`).

## top_k/2

Top-k values and indices (`%Nx.Block.TopK{}`).

### Lowering

Native `Torchx.top_k/2`.

### Options

* `:k` — passed to LibTorch

## fft2/2

Two-dimensional FFT (`%Nx.Block.FFT2{}`).

### Lowering

Native `Torchx.fft2/3`.

### Numerical notes

Signed zeros and `inspect` formatting may differ from `Nx.BinaryBackend`; see
`Torchx.NxBlockTest` for numerical comparisons.

## ifft2/2

Two-dimensional inverse FFT (`%Nx.Block.IFFT2{}`).

### Lowering

Native `Torchx.ifft2/3`.

## rfft/2

Real FFT (`%Nx.Block.RFFT{}`).

### Lowering

Native `Torchx.rfft/3`.

## irfft/2

Inverse real FFT (`%Nx.Block.IRFFT{}`).

### Lowering

Native `Torchx.irfft/3`.

## cumulative_sum/2

Cumulative sum (`%Nx.Block.CumulativeSum{}`).

### Lowering

Native `Torchx.cumulative_sum/2` with optional `flip` when `:reverse` is true.

### Options

* `:axis`, `:reverse` — honoured

## cumulative_product/2

Cumulative product (`%Nx.Block.CumulativeProduct{}`).

### Lowering

Native `Torchx.cumulative_product/2`.

## cumulative_min/2

Cumulative minimum (`%Nx.Block.CumulativeMin{}`).

### Lowering

Native `Torchx.cumulative_min/2`.

## cumulative_max/2

Cumulative maximum (`%Nx.Block.CumulativeMax{}`).

### Lowering

Native `Torchx.cumulative_max/2`.

## all_close/3

All-close comparison (`%Nx.Block.AllClose{}`).

### Lowering

Native `Torchx.all_close/4` after dtype merge of operands.

### Options

* `:rtol`, `:atol`, `:equal_nan` — passed to LibTorch

## logical_not/1

Logical not (`%Nx.Block.LogicalNot{}`).

### Lowering

Native `Torchx.logical_not/1`.

## phase/1

Complex phase (`%Nx.Block.Phase{}`).

### Lowering

Default Nx block callback (no dedicated LibTorch wrapper in `Torchx.Backend`).

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

## backend_deallocate/1

Deallocate LibTorch tensor storage.

### Behaviour

Calls `Torchx.delete_tensor/1`. Returns `:already_deallocated` if the
underlying reference is invalid.

## to_pointer/2

Export tensor memory as a pointer.

### Behaviour

**Not supported.** `to_pointer/2` raises on `Torchx.Backend`.

## from_pointer/5

Import tensor memory from a pointer.

### Behaviour

**Not supported.** `from_pointer/5` raises on `Torchx.Backend`.

## to_batched/3

Stream tensors in batches.

### Behaviour

Splits via `Torchx.split/2` on the leading axis. When `:leftover` is
`:repeat`, concatenates a tail slice so the final chunk matches `batch_size`.
May drop a partial last chunk when the size is not divisible and leftover is
not `:repeat`.
