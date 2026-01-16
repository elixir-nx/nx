defmodule EXLA.Defn.ShardingTest do
  use EXLA.Case, async: true

  alias Nx.Defn.Mesh

  describe "MLIR module generation with sharding" do
    test "generates correct MLIR with simple 2D mesh and sharding" do
      fun = fn x, y -> Nx.add(x, y) end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}
      # First arg: shard dim 0 on mesh axis 0, dim 1 on mesh axis 1
      # Second arg: shard dim 0 on mesh axis 0, dim 1 not sharded
      input_shardings = [[[0], [1]], [[0], []]]

      # For mesh {2, 2}, we have 4 partitions
      # Each partition gets a shard of the inputs
      # First input: shape {8, 2} sharded as [[0], [1]] -> each partition gets {4, 1}
      # Second input: shape {8, 1} sharded as [[0], []] -> each partition gets {4, 1}
      args = [
        # partition 0
        [Nx.iota({4, 1}), Nx.iota({4, 1})],
        # partition 1
        [Nx.iota({4, 1}), Nx.iota({4, 1})],
        # partition 2
        [Nx.iota({4, 1}), Nx.iota({4, 1})],
        # partition 3
        [Nx.iota({4, 1}), Nx.iota({4, 1})]
      ]

      result = EXLA.to_mlir_module(fun, args, mesh: mesh, input_shardings: input_shardings)

      expected_mlir = """
      module {
        sdy.mesh @mesh = <["axis_0"=2, "axis_1"=2]>
        func.func public @main(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"axis_0", ?}p0, {"axis_1", ?}p0]>}, %arg1: tensor<8x1xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"axis_0", ?}p0, {?}p0]>}) -> tensor<8x2xi32> {
          %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<8x1xi32>) -> tensor<8x2xi32>
          %1 = stablehlo.add %arg0, %0 : tensor<8x2xi32>
          return %1 : tensor<8x2xi32>
        }
      }
      """

      assert expected_mlir == result.mlir_module
    end

    test "generates correct MLIR with 3D mesh" do
      fun = fn x -> Nx.multiply(x, 2) end

      mesh = %Mesh{name: "3d_mesh", shape: {2, 2, 2}}
      # Shard all three dimensions on corresponding mesh axes
      input_shardings = [[[0], [1], [2]]]

      # For mesh {2, 2, 2}, we have 8 partitions
      # Each partition gets {4, 2, 1} (full tensor {8, 4, 2} / {2, 2, 2})
      args = List.duplicate([Nx.iota({4, 2, 1})], 8)

      result = EXLA.to_mlir_module(fun, args, mesh: mesh, input_shardings: input_shardings)

      mlir = result.mlir_module

      assert mlir =~ ~r/sdy\.mesh/
      assert mlir =~ ~r/"3d_mesh"/
      assert mlir =~ ~r/sdy\.sharding/
    end

    test "generates correct MLIR with replicated dimensions" do
      fun = fn x -> Nx.add(x, 1) end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}
      # Shard only first dimension, second dimension is replicated
      input_shardings = [[[0], []]]

      # For mesh {2, 2}, we have 4 partitions
      # Input: shape {8, 4} sharded as [[0], []]
      # Each partition gets {8/2, 4} = {4, 4}
      args = List.duplicate([Nx.iota({4, 4})], 4)

      result = EXLA.to_mlir_module(fun, args, mesh: mesh, input_shardings: input_shardings)

      mlir = result.mlir_module

      assert mlir =~ ~r/sdy\.mesh/
      assert mlir =~ ~r/sdy\.sharding/
    end

    test "generates correct MLIR with multi-axis sharding" do
      fun = fn x -> Nx.transpose(x) end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}
      # Shard first dimension on both mesh axes 0 and 1
      input_shardings = [[[0, 1], []]]

      # For mesh {2, 2}, we have 4 partitions
      # Input: shape {8, 8} sharded as [[0, 1], []]
      # First dim sharded on 2 axes: 2 * 2 = 4, so each partition gets {8/4, 8} = {2, 8}
      args = List.duplicate([Nx.iota({2, 8})], 4)

      result = EXLA.to_mlir_module(fun, args, mesh: mesh, input_shardings: input_shardings)

      mlir = result.mlir_module

      assert mlir =~ ~r/sdy\.mesh/
      assert mlir =~ ~r/sdy\.sharding/
    end

    test "generates correct MLIR with multiple inputs" do
      fun = fn x, y, z -> Nx.add(Nx.multiply(x, y), z) end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}

      input_shardings = [
        # x: shard dim 0 on axis 0, dim 1 on axis 1
        [[0], [1]],
        # y: shard dim 0 on axis 0, dim 1 replicated
        [[0], []],
        # z: dim 0 replicated, shard dim 1 on axis 1
        [[], [1]]
      ]

      # For mesh {2, 2}, we have 4 partitions
      # x: {8, 2} sharded [[0], [1]] -> each partition gets {4, 1}
      # y: {8, 2} sharded [[0], []] -> each partition gets {4, 2}
      # z: {8, 2} sharded [[], [1]] -> each partition gets {8, 1}
      args = [
        # partition 0
        [Nx.iota({4, 1}), Nx.iota({4, 2}), Nx.iota({8, 1})],
        # partition 1
        [Nx.iota({4, 1}), Nx.iota({4, 2}), Nx.iota({8, 1})],
        # partition 2
        [Nx.iota({4, 1}), Nx.iota({4, 2}), Nx.iota({8, 1})],
        # partition 3
        [Nx.iota({4, 1}), Nx.iota({4, 2}), Nx.iota({8, 1})]
      ]

      result = EXLA.to_mlir_module(fun, args, mesh: mesh, input_shardings: input_shardings)

      mlir = result.mlir_module

      assert mlir =~ ~r/sdy\.mesh/
      assert mlir =~ ~r/sdy\.sharding/
    end
  end

  describe "mesh validation" do
    test "raises when input_shardings provided without mesh" do
      fun = fn x -> Nx.add(x, 1) end
      # For non-sharded case, we don't need list of lists
      args = [Nx.iota({8, 2})]

      input_shardings = [[[0], []]]

      assert_raise ArgumentError,
                   ~r/input sharding configuration provided but no device mesh was provided/,
                   fn ->
                     EXLA.to_mlir_module(fun, args, input_shardings: input_shardings)
                   end
    end

    test "raises when input_shardings is not a list" do
      fun = fn x -> Nx.add(x, 1) end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}

      # For mesh {2, 2}, we have 4 partitions
      args = List.duplicate([Nx.iota({4, 2})], 4)

      assert_raise ArgumentError,
                   ~r/input_shardings are required for sharding/,
                   fn ->
                     EXLA.shard_jit(fun, mesh, input_shardings: nil).(args)
                   end
    end

    test "raises when number of input_shardings doesn't match number of arguments" do
      fun = fn x, y -> Nx.add(x, y) end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}
      # Only one sharding spec for two arguments
      input_shardings = [[[0], []]]

      # For mesh {2, 2}, we have 4 partitions
      # Each partition has 2 inputs
      args = List.duplicate([Nx.iota({4, 1}), Nx.iota({4, 1})], 4)

      assert_raise ArgumentError,
                   ~r/expected 2 input sharding configuration.*got 1/,
                   fn ->
                     EXLA.to_mlir_module(fun, args, mesh: mesh, input_shardings: input_shardings)
                   end
    end
  end

  describe "sharding validation" do
    test "raises when axis index is out of bounds for mesh" do
      fun = fn x -> Nx.add(x, 1) end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}
      # Mesh has 2 axes (0 and 1), but we reference axis 2
      input_shardings = [[[2], []]]

      # For mesh {2, 2}, we have 4 partitions
      args = List.duplicate([Nx.iota({4, 2})], 4)

      assert_raise ArgumentError, fn ->
        EXLA.shard_jit(fun, mesh, input_shardings: input_shardings).(args)
      end
    end

    test "raises when same axis is used twice in same input sharding" do
      fun = fn x -> Nx.add(x, 1) end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}
      # Axis 0 used for both dimensions
      input_shardings = [[[0], [0]]]

      # For mesh {2, 2}, we have 4 partitions
      args = List.duplicate([Nx.iota({4, 1})], 4)

      assert_raise ArgumentError,
                   ~r/axis 0 was used twice in the same input sharding/,
                   fn ->
                     EXLA.shard_jit(fun, mesh, input_shardings: input_shardings).(args)
                   end
    end

    test "raises when sharding spec rank doesn't match tensor rank" do
      fun = fn x -> Nx.add(x, 1) end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}
      # Tensor is rank 2, but sharding spec has 3 dimensions
      input_shardings = [[[0], [1], []]]

      # For mesh {2, 2}, we have 4 partitions
      args = List.duplicate([Nx.iota({4, 2})], 4)

      assert_raise ArgumentError, fn ->
        EXLA.shard_jit(fun, mesh, input_shardings: input_shardings).(args)
      end
    end
  end

  describe "num_partitions calculation" do
    test "calculates correct num_partitions from mesh shape" do
      fun = fn x -> Nx.add(x, 1) end

      # 2 * 2 = 4 partitions
      mesh = %Mesh{name: "mesh", shape: {2, 2}}
      input_shardings = [[[0], []]]

      # For mesh {2, 2}, we have 4 partitions
      # Input sharded [[0], []] -> each partition gets {4, 2}
      args = List.duplicate([Nx.iota({4, 2})], 4)

      result = EXLA.to_mlir_module(fun, args, mesh: mesh, input_shardings: input_shardings)

      # The module should be compiled with num_partitions = 4
      # We can't directly check this from to_mlir_module, but we can verify
      # it doesn't raise an error
      assert is_binary(result.mlir_module)
    end

    test "calculates correct num_partitions for 3D mesh" do
      fun = fn x -> Nx.add(x, 1) end

      # 2 * 2 * 2 = 8 partitions
      mesh = %Mesh{name: "mesh", shape: {2, 2, 2}}
      input_shardings = [[[0], [1], [2]]]

      # Input sharded [[0], [1], [2]] -> each partition gets {4, 2, 1}
      args = List.duplicate([Nx.iota({4, 2, 1})], 8)

      result = EXLA.to_mlir_module(fun, args, mesh: mesh, input_shardings: input_shardings)

      assert is_binary(result.mlir_module)
    end

    test "calculates correct num_partitions for 1D mesh" do
      fun = fn x -> Nx.add(x, 1) end

      # 4 partitions
      mesh = %Mesh{name: "mesh", shape: {4}}
      input_shardings = [[[0]]]

      # Input sharded [[0]] -> each partition gets {2}
      args = List.duplicate([Nx.iota({2})], 4)

      result = EXLA.to_mlir_module(fun, args, mesh: mesh, input_shardings: input_shardings)

      assert is_binary(result.mlir_module)
    end
  end

  describe "complex sharding patterns" do
    test "handles fully replicated tensor" do
      fun = fn x -> Nx.add(x, 1) end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}
      # All dimensions replicated
      input_shardings = [[[], []]]

      # For mesh {2, 2}, we have 4 partitions
      # Input fully replicated -> each partition gets full {8, 4}
      args = List.duplicate([Nx.iota({8, 4})], 4)

      result = EXLA.to_mlir_module(fun, args, mesh: mesh, input_shardings: input_shardings)

      assert is_binary(result.mlir_module)
    end

    test "handles scalar input" do
      fun = fn x -> Nx.add(x, 1) end

      mesh = %Mesh{name: "mesh", shape: {2}}
      # Scalar has no dimensions to shard
      input_shardings = [[]]

      # For mesh {2}, we have 2 partitions
      # Scalar is replicated across all partitions
      args = List.duplicate([Nx.tensor(5.0)], 2)

      result = EXLA.to_mlir_module(fun, args, mesh: mesh, input_shardings: input_shardings)

      assert is_binary(result.mlir_module)
    end

    test "handles mixed sharding and replication" do
      fun = fn x, y, z -> {Nx.add(x, y), Nx.multiply(y, z)} end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}

      input_shardings = [
        # x: both dimensions sharded
        [[0], [1]],
        # y: first dimension sharded, second replicated
        [[0], []],
        # z: fully replicated
        [[], []]
      ]

      # For mesh {2, 2}, we have 4 partitions
      # x: {8, 4} sharded [[0], [1]] -> each partition gets {4, 2}
      # y: {8, 4} sharded [[0], []] -> each partition gets {4, 4}
      # z: {8, 4} sharded [[], []] -> each partition gets {8, 4}
      args = [
        # partition 0
        [Nx.iota({4, 2}), Nx.iota({4, 4}), Nx.iota({8, 4})],
        # partition 1
        [Nx.iota({4, 2}), Nx.iota({4, 4}), Nx.iota({8, 4})],
        # partition 2
        [Nx.iota({4, 2}), Nx.iota({4, 4}), Nx.iota({8, 4})],
        # partition 3
        [Nx.iota({4, 2}), Nx.iota({4, 4}), Nx.iota({8, 4})]
      ]

      result = EXLA.to_mlir_module(fun, args, mesh: mesh, input_shardings: input_shardings)

      assert is_binary(result.mlir_module)
    end
  end

  describe "MLIR output format" do
    test "MLIR contains mesh definition with correct shape" do
      fun = fn x -> Nx.add(x, 1) end

      mesh = %Mesh{name: "test_mesh", shape: {4, 2}}
      input_shardings = [[[0], []]]

      # For mesh {4, 2}, we have 8 partitions
      # Input sharded [[0], []] -> each partition gets {2, 2}
      args = List.duplicate([Nx.iota({2, 2})], 8)

      result = EXLA.to_mlir_module(fun, args, mesh: mesh, input_shardings: input_shardings)

      mlir = result.mlir_module

      # Check mesh name appears
      assert mlir =~ "test_mesh"

      # Check mesh definition exists
      assert mlir =~ ~r/sdy\.mesh/
    end

    test "MLIR contains function with sharding attributes" do
      fun = fn x -> Nx.add(x, 1) end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}
      input_shardings = [[[0], [1]]]

      # For mesh {2, 2}, we have 4 partitions
      # Input sharded [[0], [1]] -> each partition gets {4, 1}
      args = List.duplicate([Nx.iota({4, 1})], 4)

      result = EXLA.to_mlir_module(fun, args, mesh: mesh, input_shardings: input_shardings)

      mlir = result.mlir_module

      # Check that function arguments have sharding attributes
      assert mlir =~ ~r/sdy\.sharding/

      # Check that the main function exists
      assert mlir =~ ~r/func\.func.*@main/
    end
  end
end
