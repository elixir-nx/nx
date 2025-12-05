fun = fn x, y -> {Nx.add(x, y), Nx.multiply(x, y)} end
args = [Nx.iota({8, 2}), Nx.iota({8, 1})]

mesh = EXLA.Sharding.mesh("mesh", x: 2, y: 2, z: 2)

input_shardings = [EXLA.Sharding.sharding("mesh", [["x", "z"], ["y"]]), EXLA.Sharding.sharding("mesh", [["x", "z"], []])]

result = EXLA.to_mlir_module(fun, args, mesh: mesh, input_shardings: input_shardings)

IO.puts(result.mlir_module)

result = EXLA.jit_apply(fun, args, mesh: mesh, input_shardings: input_shardings)
dbg(result)

# run with: XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text --xla_force_host_platform_device_count=10" mix run sharding.exs
