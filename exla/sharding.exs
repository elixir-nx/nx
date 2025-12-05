fun = fn x, y -> {Nx.add(x, y), Nx.multiply(x, y)} end
args = [Nx.iota({2, 2}), Nx.iota({2, 1})]

mesh = EXLA.Sharding.mesh("mesh", x: 2, y: 2)

input_shardings = [EXLA.Sharding.sharding("mesh", [["x"], ["y"]]), EXLA.Sharding.sharding("mesh", [["x"], ["y"]])]

result = EXLA.jit_apply(fun, args, mesh: mesh, input_shardings: input_shardings)

dbg(result)
