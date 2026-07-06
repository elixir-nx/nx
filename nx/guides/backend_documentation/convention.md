# Backend documentation convention

Nx exposes a stable public API (`Nx`, `Nx.LinAlg`, and so on) while individual
backends implement the underlying callbacks and optional `Nx.block/4` lowerings.
Users who need to know how a specific backend differs from the portable API should
not have to read backend source code.

## Documentation guides

Each backend may publish **backend documentation guides** that mirror the Nx
module structure. The naming convention is:

    guides/backend_documentation/nx.md           # mirrors top-level `Nx`
    guides/backend_documentation/nx_lin_alg.md   # mirrors `Nx.LinAlg`

For example, EXLA documents operations in its [Backend documentation](https://hexdocs.pm/exla/backend_documentation.html)
guides, and Torchx in its [Backend documentation](https://hexdocs.pm/torchx/backend_documentation.html)
guides.

## What to document

Document an operation only when a backend has **divergent behaviour**, **backend-specific
options**, or **trade-offs and limitations** worth calling out relative to the
portable Nx API or to other backends. Examples:

  * an option that a backend honours, ignores, or overrides
  * platform or device support gaps (for example an operation unavailable on MPS)
  * numerical or performance trade-offs that affect how you use the API
  * warnings about deadlock, unsupported pointer modes, and similar constraints

Do not document routine lowerings (for example which StableHLO op or LibTorch
kernel is used) when behaviour matches the reference Nx implementation. If you
are implementing or overriding custom blocks, the backend source remains the best
source of truth.

Cross-link from the Nx function doc when a backend guide has relevant notes for
that operation.

## Relation to `Nx.Backend`

Most tensor operations go through `Nx.Backend` callbacks. Operations that need
a portable default with optional native acceleration use `Nx.block/4` and the
`Nx.Block.*` tags — see `Nx.Backend.block/4`. Backend documentation guides
should cover divergent behaviour for both callback implementations and block
lowerings.

## Example

```elixir
iex> Nx.take(Nx.tensor([10, 20, 30]), Nx.tensor([0, 2]))
#Nx.Tensor<
  s32[2]
  [10, 30]
>
```
