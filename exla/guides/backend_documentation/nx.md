# Nx (EXLA)

```elixir
iex> Nx.take(Nx.tensor([10, 20, 30]), Nx.tensor([0, 2]))
#Nx.Tensor<
  s32[2]
  [10, 30]
>
```

EXLA-specific notes for top-level `Nx` operations where behaviour diverges from
the portable Nx API or from other backends.

Linear algebra is documented separately in [Nx.LinAlg](backend_documentation-nx_lin_alg.html).

## runtime_call/4

Runtime Elixir callback from `defn` (`runtime_call` expression).

### Platforms

* **Host / CUDA** — supported via a device-side callback bridged through
`EXLA.MLIR.Value.runtime_call/3`
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

### Options

* `:client`, `:device_id` — select the EXLA client and device

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

### Limitations

Not supported for ROCm or TPU.
