# Backend documentation convention

Nx exposes a stable public API (`Nx`, `Nx.LinAlg`, and so on) while individual
backends implement the underlying callbacks and optional block lowerings.
Users who need to know how a specific backend differs from the portable API
should not have to read backend source code.

## Documentation guides

Each backend may publish **backend documentation guides** that mirror the Nx
module structure. The naming convention is:

    guides/backend_documentation/nx.md           # mirrors top-level `Nx`
    guides/backend_documentation/nx_lin_alg.md   # mirrors `Nx.LinAlg`

For example, EXLA documents operations in its [Backend documentation](https://hexdocs.pm/exla/backend_documentation.html)
guides, and Torchx in its [Backend documentation](https://hexdocs.pm/torchx/backend_documentation.html)
guides.

## What to document

Document an operation only when a backend has **divergent behavior**,
**backend-specific options**, or **trade-offs and limitations** worth calling
out relative to the portable Nx API or to other backends. Examples:

  * an option that a backend honors, ignores, or overrides
  * platform or device support gaps (for example an operation unavailable on MPS)
  * numerical or performance trade-offs that affect how you use the API
  * warnings about deadlocks, unsupported pointer modes, and similar constraints

Write for library users: short prose about what they can rely on, which devices
are supported, and which options matter. Do not document routine lowerings
(for example which StableHLO op or LibTorch kernel is used) when behavior
matches the reference Nx implementation, and avoid referencing internal
modules or functions. If you are implementing or overriding custom blocks,
the backend source remains the best source of truth.

## Example

```elixir
iex> Nx.take(Nx.tensor([10, 20, 30]), Nx.tensor([0, 2]))
#Nx.Tensor<
  s32[2]
  [10, 30]
>
```
