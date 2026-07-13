# Nx (EXLA)

EXLA-specific notes for top-level `Nx` operations where behavior diverges from
the portable Nx API or from other backends.

Linear algebra is documented separately in [Nx.LinAlg](backend_documentation-nx_lin_alg.html).

## `Nx.runtime_call/4`

This function allows running Elixir code within `defn` code. Outside of a
`defn`, it executes the callback directly — it does not compile the code.

Currently supported only for host and CUDA backends. Raises for ROCm/TPU
devices.

> #### Avoid device deadlocks {: .warning}
>
> The tensors given to the callback are allocated in the binary backend. You
> must be careful to not use the same device the `defn` itself is running on,
> as that will lead to a deadlock.

## `Nx.backend_copy/2`

Copies an EXLA tensor to another backend or device. Use the `:client` and
`:device_id` options when the destination is `EXLA.Backend` to choose which
EXLA client and device receive the data.

Copying to a non-EXLA backend materializes the tensor on that backend.

## `Nx.backend_transfer/2`

Like `Nx.backend_copy/2`, but deallocates the source tensor when the
destination is a different backend or device. Transferring to the same EXLA
client and device is a no-op.

Accepts the same `:client` and `:device_id` options as `Nx.backend_copy/2`.

## `Nx.to_pointer/2`

Exports an EXLA tensor's device memory as an `Nx.Pointer` for interop with
other native code.

Supported on **host** and **CUDA** only. Raises for ROCm, TPU, or tensors that
are not allocated on a device.

Modes:

  * `:local` — a local address on host or CUDA
  * `:ipc` — a shareable handle (shared memory on host, CUDA IPC on CUDA)

The `:permissions` option sets the octal file mode for host IPC shared memory
(default `0o400`).

## `Nx.from_pointer/5`

Imports device memory from an `Nx.Pointer` into an `EXLA.Backend` tensor on
**host** or **CUDA**. Raises for ROCm or TPU.

The pointer's `data_size` must match the byte size of the result tensor. Use
`:client` and `:device_id` to select the destination device, and `:names` to
set tensor axis names on the result template.
