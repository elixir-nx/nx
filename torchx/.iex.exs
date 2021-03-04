
if :code.module_status(Nx) == :loaded do
  Nx.default_backend(Torchx.Backend)
end
