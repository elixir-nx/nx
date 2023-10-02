# client = EXLA.Client.fetch!(:host)

# arg_xla_shape = EXLA.Shape.make_shape({:f, 32}, {4, 3, 1})
# mlir_arg_types = Enum.map([arg_xla_shape, arg_xla_shape], &EXLA.MLIR.Type.new/1)
# mlir_ret_type = EXLA.MLIR.Type.new(arg_xla_shape)

# module = EXLA.MLIR.Module.new()
# function = EXLA.MLIR.Module.create_function(module, "main", mlir_arg_types, mlir_ret_type)
# [arg1, arg2] = EXLA.MLIR.Function.get_arguments(function)
# result = EXLA.MLIR.Value.add(arg1, arg2)
# result = EXLA.MLIR.Value.subtract(arg1, result)
# result = EXLA.MLIR.Value.tuple([result])
# result = EXLA.MLIR.Value.get_tuple_element(result, 0)
# :ok = EXLA.NIF.mlir_build(function.ref, result.ref)
# EXLA.NIF.dump_mlir_module(module.ref)

# executable = EXLA.MLIR.Module.compile(
#   module,
#   client,
#   [arg_xla_shape, arg_xla_shape],
#   EXLA.Shape.make_tuple_shape([arg_xla_shape])
# )

# t1 = Nx.broadcast(0.0, {4, 3, 1}) |> Nx.to_binary() |> EXLA.BinaryBuffer.from_binary(arg_xla_shape)
# t2 = Nx.broadcast(1.0, {4, 3, 1}) |> Nx.to_binary() |> EXLA.BinaryBuffer.from_binary(arg_xla_shape)

# [[result]] = EXLA.Executable.run(executable, [[t1, t2]])

# result
# |> EXLA.DeviceBuffer.read()
# |> Nx.from_binary(:f32)
# |> Nx.reshape({4, 3, 1})
# |> IO.inspect(label: "all negative ones")

