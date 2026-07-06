# Backend documentation

Torchx-specific documentation for Nx backend behaviour.

These guides document divergent behaviour, backend-specific options, and
limitations for Nx operations. They mirror the structure of the Nx API — see
the [backend documentation convention](https://hexdocs.pm/nx/backend_documentation-convention.html).

## Guides

  * [Nx](backend_documentation-nx.html) — top-level `Nx` blocks, transfers, and callbacks
  * [Nx.LinAlg](backend_documentation-nx_lin_alg.html) — linear algebra blocks

Implementation code lives in `Torchx.Backend` and the `Torchx` NIF bindings.

## Example

```elixir
iex> Nx.take(Nx.tensor([10, 20, 30]), Nx.tensor([0, 2]))
#Nx.Tensor<
  s32[2]
  [10, 30]
>
```
