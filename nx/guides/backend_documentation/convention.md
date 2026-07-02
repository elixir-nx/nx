# Backend documentation convention

Nx exposes a stable public API (`Nx`, `Nx.LinAlg`, and so on) while individual
backends implement the underlying callbacks and optional `Nx.block/4` lowerings.
Users who need to know *how* a specific backend executes an operation should
not have to read backend source code.

## Documentation guides

Each backend may publish **backend documentation guides** that mirror the Nx
module structure and document backend-specific behaviour for each callback.
The naming convention is:

    guides/backend_documentation/nx.md           # mirrors top-level `Nx`
    guides/backend_documentation/nx_lin_alg.md   # mirrors `Nx.LinAlg`

For example, EXLA documents operations in its [Backend documentation](https://hexdocs.pm/exla/backend_documentation.html)
guides, and Torchx in its [Backend documentation](https://hexdocs.pm/torchx/backend_documentation.html)
guides.

These guides describe primitives that backends implement, without being part of the
callable Nx API.

## What to document

Backends should document, for each mirrored function:

  * whether the operation is lowered natively, compiled from the default
    `Nx.block/4` callback, or handled through another mechanism (for example
    a direct StableHLO op)
  * supported types, devices, and platforms
  * options that the backend honours, ignores, or overrides
  * numerical behaviour, performance trade-offs, and known limitations relative
    to the reference implementation in Nx

Cross-link from the Nx function doc when appropriate (for example
`See EXLA and Torchx backend documentation for qr/2`).

## Relation to `Nx.Backend`

Most tensor operations go through `Nx.Backend` callbacks. Operations that need
a portable default with optional native acceleration use `Nx.block/4` and the
`Nx.Block.*` tags — see `Nx.Backend.block/4`. Backend documentation guides
should cover both callback implementations and block lowerings.

## Example

```elixir
iex> Nx.take(Nx.tensor([10, 20, 30]), Nx.tensor([0, 2]))
#Nx.Tensor<
  s32[2]
  [10, 30]
>
```
