# Nx (EXLA)

See also [Nx.LinAlg](backend_documentation-nx_lin_alg.html).

## `Nx.runtime_call/4`

This function allows running Elixir code within `defn` code. Outside of a
`defn`, it executes the callback directly, it does not compile the code.

Currently supported only for host and CUDA backends. Raises for ROCm/TPU
devices.

> #### Avoid device deadlocks {: .warning}
>
> The tensors given to the callback are allocated in the binary backend. You
> must be careful to not use the same device the `defn` itself is running on,
> as that will lead to a deadlock.

## `Nx.backend_copy/2` and `Nx.backend_transfer/2`

When copying or transferring to `EXLA.Backend`, pass `:client` and `:device_id`
to pick the destination. Transfer frees the source unless you stay on the same
client and device.

## `Nx.to_pointer/2`

Supported on host and CUDA. Raises for ROCm, TPU, or tensors that are not on a
device.

Use `mode: :local` for a local address, or `mode: :ipc` for a shareable handle.
On host IPC, `:permissions` sets the shared-memory file mode (default `0o400`).

## `Nx.from_pointer/5`

Supported on host and CUDA. Raises for ROCm or TPU.

The pointer's `data_size` must match the tensor byte size. Pass `:client` and
`:device_id` for the destination device.
