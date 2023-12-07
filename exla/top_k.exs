Nx.default_backend(EXLA.Backend)
Nx.Defn.default_options(compiler: EXLA, compiler_mode: :mlir)

t = Nx.tensor([:nan, :neg_infinity, :infinity, 0.0])
{values, indices} = Nx.top_k(t, k: 3)

IO.inspect(values, label: "values")
IO.inspect(indices, label: "indices")
