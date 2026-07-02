# Backend documentation

EXLA-specific documentation for Nx backend behaviour.

These guides describe how EXLA lowers and executes Nx operations. They mirror
the structure of the Nx API (see the [backend documentation convention](https://hexdocs.pm/nx/backend_documentation-convention.html))
but are **not callable** — use `Nx` and `compiler: EXLA` in your code.

## Guides

  * [Nx](backend_documentation-nx.html) — top-level `Nx` blocks, transfers, and `defn` integration
  * [Nx.LinAlg](backend_documentation-nx_lin_alg.html) — linear algebra blocks and related lowerings

Implementation code lives in `EXLA.Defn`, `EXLA.CustomCall`, and related modules.

## Example

```elixir
iex> Nx.take(Nx.tensor([10, 20, 30]), Nx.tensor([0, 2]))
#Nx.Tensor<
  s32[2]
  [10, 30]
>
```
