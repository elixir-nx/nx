
if Code.ensure_loaded?(Nx) do
  Nx.default_backend(Torchx.Backend)
end
