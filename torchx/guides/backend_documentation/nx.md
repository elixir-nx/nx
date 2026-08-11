# Nx (Torchx)

See also [Nx.LinAlg](backend_documentation-nx_lin_alg.html).

Torchx is a backend, not a `defn` compiler. On Apple MPS, `f64` and `c128`
are not supported and raise.

## `Nx.runtime_call/4`

Outside of a `defn`, the callback runs directly on Torchx tensors. Inside a
`defn`, you need a compiler such as EXLA or `Nx.Defn.Evaluator` — Torchx does
not provide one.

## `Nx.to_pointer/2`

Supported on CPU, CUDA, and MPS.

Use `mode: :local` for a local address, or `mode: :ipc` for a shareable handle
(POSIX shared memory on CPU; CUDA IPC on CUDA). MPS does not support `:ipc`.
On host IPC, `:permissions` sets the shared-memory file mode (default `0o400`).
Host IPC is not available on Windows.

Keep the exporting tensor alive for as long as the pointer/handle is in use.
Non-contiguous tensors are replaced in-place with a contiguous copy before the
address is exposed; host IPC also repoints the exporter at the shared mapping.

## `Nx.from_pointer/5`

Supported for `kind: :local` on CPU, CUDA, and MPS, and for `kind: :ipc` on
CPU (host shm) and CUDA (CUDA IPC). Raises for MPS IPC.

The pointer's `data_size` must match the Torchx storage byte size for the given
type and shape. Pass `:device` for the destination device.
