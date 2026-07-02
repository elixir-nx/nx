# Nx (EXLA)

```elixir
iex> Nx.take(Nx.tensor([10, 20, 30]), Nx.tensor([0, 2]))
#Nx.Tensor<
  s32[2]
  [10, 30]
>
```

EXLA implementation notes for top-level `Nx` operations.

Each function below documents how EXLA handles the corresponding `Nx` API when
`compiler: EXLA` is used (or when `EXLA.Backend` stores tensors).

Linear algebra is documented separately in [Nx.LinAlg](backend_documentation-nx_lin_alg.html).

Many `Nx` functions delegate to `Nx.block/4`. During EXLA compilation,
specialized lowerings exist for some block tags; otherwise EXLA compiles the
default callback as an XLA subcomputation.

## take/3

Tensor indexing (`%Nx.Block.Take{}`).

### Lowering

Lowered to StableHLO `gather` in `EXLA.Defn` (not a custom call).

### Options

* `:axis` — honoured; indices are gathered along this axis

### Platforms

All EXLA clients.

## take_along_axis/3

Take along axis (`%Nx.Block.TakeAlongAxis{}`).

### Lowering

No dedicated gather lowering — EXLA compiles the default Nx callback as an XLA
subcomputation.

### Options

* `:axis` — honoured by the default callback

## top_k/2

Top-k values and indices (`%Nx.Block.TopK{}`).

### Lowering

Lowered to StableHLO `top_k` via `EXLA.MLIR.Value.top_k/3`.

### Options

* `:k` — number of elements to return per slice

## fft2/2

Two-dimensional FFT (`%Nx.Block.FFT2{}`).

### Lowering

Lowered to StableHLO FFT ops via `EXLA.MLIR.Value.fft/4` with mode `:fft`.

### Options

* `:lengths`, `:axes`, `:eps` — forwarded to the lowering; `:eps` may be used
for cleanup in the default callback when compilation falls back

## ifft2/2

Two-dimensional inverse FFT (`%Nx.Block.IFFT2{}`).

### Lowering

Lowered to StableHLO FFT ops with mode `:ifft`.

## rfft/2

Real FFT (`%Nx.Block.RFFT{}`).

### Lowering

Lowered to StableHLO real FFT via `Value.fft/4` with mode `:rfft`. Input is
treated as real; output type is complex per the expression template.

## irfft/2

Inverse real FFT (`%Nx.Block.IRFFT{}`).

### Lowering

Lowered to StableHLO IRFFT via `Value.fft/4` with mode `:irfft`.

## cumulative_sum/2

Cumulative sum (`%Nx.Block.CumulativeSum{}`).

### Lowering

Default callback compiled as an XLA subcomputation (no native custom call).

### Options

* `:axis`, `:reverse` — honoured by the default callback

## cumulative_product/2

Cumulative product (`%Nx.Block.CumulativeProduct{}`).

### Lowering

Default callback compiled as an XLA subcomputation.

## cumulative_min/2

Cumulative minimum (`%Nx.Block.CumulativeMin{}`).

### Lowering

Default callback compiled as an XLA subcomputation.

## cumulative_max/2

Cumulative maximum (`%Nx.Block.CumulativeMax{}`).

### Lowering

Default callback compiled as an XLA subcomputation.

## all_close/3

All-close comparison (`%Nx.Block.AllClose{}`).

### Lowering

Default callback compiled as an XLA subcomputation.

### Options

* `:rtol`, `:atol`, `:equal_nan` — honoured by the default callback

## logical_not/1

Logical not (`%Nx.Block.LogicalNot{}`).

### Lowering

Default callback compiled as an XLA subcomputation (delegates to element-wise
equality in the reference implementation).

## phase/1

Complex phase (`%Nx.Block.Phase{}`).

### Lowering

Default callback compiled as an XLA subcomputation.

## runtime_call/4

Runtime Elixir callback from `defn` (`runtime_call` expression).

### Lowering

On **host** and **CUDA** clients, EXLA emits a device-side callback bridged
through `EXLA.MLIR.Value.runtime_call/3`. Callback tensors are materialized for
the Elixir function; results must match the output template.

### Platforms

* **Host / CUDA** — supported
* **ROCm / TPU** — raises with a message to use `:host` or `:cuda`

### Outside `defn`

Executes the callback directly on the input backend without compilation.

### Warnings

Avoid `Nx.backend_transfer/2` on callback tensors inside the function when
using `Nx.Defn.Evaluator`. Avoid running other Nx computations on the same GPU
device from within the callback (deadlock risk).

## backend_copy/2

Copy tensor data to another backend (`EXLA.Backend.backend_copy/3`).

### Behaviour

Copying to `EXLA.Backend` on the same client and device returns the tensor
unchanged. Cross-device copies use `EXLA.DeviceBuffer.copy_to_device/3`.
Copying out reads device memory via `EXLA.DeviceBuffer.read/1` and delegates
to the target backend's `from_binary/3`.

### Options

* `:client`, `:device_id` — select the EXLA client and device

## backend_transfer/2

Transfer tensor data to another backend (`EXLA.Backend.backend_transfer/3`).

### Behaviour

Same as `backend_copy/2` followed by deallocation of the source device buffer
when leaving `EXLA.Backend`.

## backend_deallocate/1

Deallocate device memory (`EXLA.Backend.backend_deallocate/1`).

### Behaviour

Calls `EXLA.DeviceBuffer.deallocate/1` for tensors on `EXLA.Backend`.

## to_pointer/2

Export device memory as a pointer (`EXLA.Backend.to_pointer/2`).

### Modes

* `:local` — supported on **host** and **CUDA** clients; returns a local
address integer
* `:ipc` — supported on **host** (`shm_open` handle) and **CUDA** (IPC
handle plus device id)

### Options

* `:permissions` — octal file mode for host IPC shared memory (default
`0o400`)

### Limitations

Not supported for ROCm, TPU, or non-device buffers.

## from_pointer/5

Import device memory from a pointer (`EXLA.Backend.from_pointer/5`).

### Behaviour

Creates an `EXLA.DeviceBuffer` from a local address or IPC handle on **host**
or **CUDA** clients. Pointer `data_size` must match the tensor byte size.

### Options

* `:client`, `:device_id` — destination device
* `:names` — tensor names for the result template

## to_batched/3

Stream tensors in batches (`EXLA.Backend.to_batched/3`).

### Behaviour

Splits the leading axis via XLA slice and concatenate operations on device
buffers. Supports `:leftover` `:repeat` and `:discard` like `Nx.to_batched/2`.
