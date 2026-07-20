# Backend documentation convention

Backends may ship guides that mirror the Nx modules, so you can look up how
EXLA or Torchx differ without reading their source:

    guides/backend_documentation/nx.md           # top-level `Nx`
    guides/backend_documentation/nx_lin_alg.md   # `Nx.LinAlg`

See EXLA's [Backend documentation](https://hexdocs.pm/exla/backend_documentation.html)
and Torchx's [Backend documentation](https://hexdocs.pm/torchx/backend_documentation.html).

## What to document

Only what an Nx user would ask themselves:

  * Does this work on my device?
  * Which options are ignored or raise?
  * Will I deadlock, get a surprising error, or hit a known limitation?

Skip how things are lowered. If you are wiring custom blocks, read the source.

## Example

```elixir
iex> Nx.take(Nx.tensor([10, 20, 30]), Nx.tensor([0, 2]))
#Nx.Tensor<
  s32[2]
  [10, 30]
>
```
