defmodule EXLA.Defn.ShardingTest do
  use EXLA.Case, async: true

  alias Nx.Defn.Mesh

  describe "MLIR module generation with sharding" do
    @moduletag :multi_device
    test "output sharding with tuple outputs" do
      # Function that returns a tuple with different output shardings
      fun = fn x, y -> {Nx.add(x, y), Nx.multiply(y, 2)} end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}
      # First input: sharded on both axes (8x2 -> 4x1 per device)
      # Second input: sharded only on axis 0 (8x2 -> 4x2 per device)
      input_shardings = [%{0 => [0], 1 => [1]}, %{0 => [0]}]

      # Output shardings are inferred by Shardy from input shardings
      # First output (x+y): should be sharded on both axes -> {4,1} per device
      # Second output (y*2): should be sharded only on axis 0 -> {4,2} per device

      # Logical tensors:
      # x: [[0,10], [1,11], [2,12], [3,13], [4,14], [5,15], [6,16], [7,17]] (8x2)
      # y: [[100,100], [101,101], [102,102], [103,103], [104,104], [105,105], [106,106], [107,107]] (8x2)
      #
      # First input (x) sharded on both axes -> {4,1} per device:
      #   Dev0: [[0],[1],[2],[3]]    Dev1: [[10],[11],[12],[13]]
      #   Dev2: [[4],[5],[6],[7]]    Dev3: [[14],[15],[16],[17]]
      #
      # Second input (y) sharded only on axis 0 -> {4,2} per device:
      #   Dev0,1: [[100],[101],[102],[103]]
      #   Dev2,3: [[104],[105],[106],[107]]
      args = [
        [
          Nx.tensor([[0], [1], [2], [3]]),
          Nx.tensor([[100], [101], [102], [103]])
        ],
        [
          Nx.tensor([[10], [11], [12], [13]]),
          Nx.tensor([[100], [101], [102], [103]])
        ],
        [
          Nx.tensor([[4], [5], [6], [7]]),
          Nx.tensor([[104], [105], [106], [107]])
        ],
        [
          Nx.tensor([[14], [15], [16], [17]]),
          Nx.tensor([[104], [105], [106], [107]])
        ]
      ]

      results =
        EXLA.shard_jit(fun, mesh, input_shardings: input_shardings).(args)

      # Verify shapes and actual data values for each device
      # Device 0: x={4,1} rows[0-3]col[0], y={4,1} rows[0-3]
      {result1_d0, result2_d0} = Enum.at(results, 0)
      # First output (add x+y): both broadcast to {4,2}, add, then shard on both axes -> {4,1}
      # x[[0],[1],[2],[3]] broadcasts to [[0,0],[1,1],[2,2],[3,3]]
      # y[[100],[101],[102],[103]] broadcasts to [[100,100],[101,101],[102,102],[103,103]]
      # x+y = [[100,100],[102,102],[104,104],[106,106]], device 0 gets col 0
      assert_equal(result1_d0, Nx.tensor([[100], [102], [104], [106]]))
      # Second output (y*2): y broadcasts to {4,2}, multiply by 2, result is {4,2}
      # [[100],[101],[102],[103]] * 2 = [[200],[202],[204],[206]], broadcasts to [[200,200],[202,202],[204,204],[206,206]]
      assert_equal(result2_d0, Nx.tensor([[200, 200], [202, 202], [204, 204], [206, 206]]))

      # Device 1: x={4,1} rows[0-3]col[1], y={4,1} rows[0-3]
      {result1_d1, result2_d1} = Enum.at(results, 1)
      # First output: x[[10],[11],[12],[13]] + y[[100],[101],[102],[103]]
      # broadcasts to [[10,10],[11,11],[12,12],[13,13]] + [[100,100],[101,101],[102,102],[103,103]]
      # = [[110,110],[112,112],[114,114],[116,116]], device 1 gets col 1
      assert_equal(result1_d1, Nx.tensor([[110], [112], [114], [116]]))
      # Second output: same as device 0 (y is replicated across axis 1)
      assert_equal(result2_d1, Nx.tensor([[200, 200], [202, 202], [204, 204], [206, 206]]))

      # Device 2: x={4,1} rows[4-7]col[0], y={4,1} rows[4-7]
      {result1_d2, result2_d2} = Enum.at(results, 2)
      # First output: x[[4],[5],[6],[7]] + y[[104],[105],[106],[107]]
      # = [[108,108],[110,110],[112,112],[114,114]], device 2 gets col 0
      assert_equal(result1_d2, Nx.tensor([[108], [110], [112], [114]]))
      # Second output: [[104],[105],[106],[107]] * 2, broadcasts to [[208,208]...]
      assert_equal(result2_d2, Nx.tensor([[208, 208], [210, 210], [212, 212], [214, 214]]))

      # Device 3: x={4,1} rows[4-7]col[1], y={4,1} rows[4-7]
      {result1_d3, result2_d3} = Enum.at(results, 3)
      # First output: x[[14],[15],[16],[17]] + y[[104],[105],[106],[107]]
      # broadcasts to [[14,14],[15,15],[16,16],[17,17]] + [[104,104],[105,105],[106,106],[107,107]]
      # = [[118,118],[120,120],[122,122],[124,124]], device 3 gets col 1
      assert_equal(result1_d3, Nx.tensor([[118], [120], [122], [124]]))
      # Second output: same as device 2 (y is replicated across axis 1)
      assert_equal(result2_d3, Nx.tensor([[208, 208], [210, 210], [212, 212], [214, 214]]))
    end

    @moduletag :multi_device
    test "output sharding inferred from inputs" do
      fun = fn x, y -> Nx.add(x, y) end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}
      input_shardings = [%{0 => [0], 1 => [1]}, %{0 => [0]}]
      # Output sharding inferred by Shardy should be {0 => [0], 1 => [1]}

      # Each device gets unique shard data
      # Logical x: [[0,10], [1,11], [2,12], [3,13], [4,14], [5,15], [6,16], [7,17]]
      # Logical y: [[100], [101], [102], [103], [104], [105], [106], [107]]
      args = [
        [Nx.tensor([[0], [1], [2], [3]]), Nx.tensor([[100], [101], [102], [103]])],
        [Nx.tensor([[10], [11], [12], [13]]), Nx.tensor([[100], [101], [102], [103]])],
        [Nx.tensor([[4], [5], [6], [7]]), Nx.tensor([[104], [105], [106], [107]])],
        [Nx.tensor([[14], [15], [16], [17]]), Nx.tensor([[104], [105], [106], [107]])]
      ]

      results = EXLA.shard_jit(fun, mesh, input_shardings: input_shardings).(args)

      assert length(results) == 4

      # Each result should be shape {4,1} (sharded on both axes)
      # Device 0: [[0+100], [1+101], [2+102], [3+103]] = [[100], [102], [104], [106]]
      assert_equal(Enum.at(results, 0), Nx.tensor([[100], [102], [104], [106]]))

      # Device 1: [[10+100], [11+101], [12+102], [13+103]] = [[110], [112], [114], [116]]
      assert_equal(Enum.at(results, 1), Nx.tensor([[110], [112], [114], [116]]))

      # Device 2: [[4+104], [5+105], [6+106], [7+107]] = [[108], [110], [112], [114]]
      assert_equal(Enum.at(results, 2), Nx.tensor([[108], [110], [112], [114]]))

      # Device 3: [[14+104], [15+105], [16+106], [17+107]] = [[118], [120], [122], [124]]
      assert_equal(Enum.at(results, 3), Nx.tensor([[118], [120], [122], [124]]))

      # Verify device IDs
      device_ids = for r <- results, do: r.data.buffer.device_id
      assert Enum.sort(device_ids) == [0, 1, 2, 3]
    end

    @moduletag :multi_device
    test "generates correct MLIR with simple 2D mesh and sharding" do
      fun = fn x, y -> Nx.add(x, y) end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}
      # First arg: shard dim 0 on mesh axis 0, dim 1 on mesh axis 1
      # Second arg: shard dim 0 on mesh axis 0, dim 1 not sharded
      input_shardings = [%{0 => [0], 1 => [1]}, %{0 => [0]}]

      # For mesh {2, 2}, we have 4 partitions
      # Each partition gets a shard of the inputs
      # First input: shape {8, 2} sharded as [[0], [1]] -> each partition gets {4, 1}
      # Second input: shape {8, 1} sharded as [[0], []] -> each partition gets {4, 1}
      args = [
        # partition 0
        [Nx.tensor([[0], [1], [2], [3]]), Nx.tensor([[100], [101], [102], [103]])],
        # partition 1
        [Nx.tensor([[10], [11], [12], [13]]), Nx.tensor([[100], [101], [102], [103]])],
        # partition 2
        [Nx.tensor([[4], [5], [6], [7]]), Nx.tensor([[104], [105], [106], [107]])],
        # partition 3
        [Nx.tensor([[14], [15], [16], [17]]), Nx.tensor([[104], [105], [106], [107]])]
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

      assert [result0, result1, result2, result3] =
               EXLA.shard_jit(fun, mesh, input_shardings: input_shardings).(args)

      assert_equal(result0, Nx.tensor([[100, 100], [102, 102], [104, 104], [106, 106]]))
      assert result0.data.buffer.device_id == 0
      assert_equal(result1, Nx.tensor([[110, 110], [112, 112], [114, 114], [116, 116]]))
      assert result1.data.buffer.device_id == 1
      assert_equal(result2, Nx.tensor([[108, 108], [110, 110], [112, 112], [114, 114]]))
      assert result2.data.buffer.device_id == 2
      assert_equal(result3, Nx.tensor([[118, 118], [120, 120], [122, 122], [124, 124]]))
      assert result3.data.buffer.device_id == 3
    end

    @moduletag :multi_device
    test "generates correct MLIR with 3D mesh" do
      fun = fn x -> Nx.multiply(x, 2) end

      mesh = %Mesh{name: "3d_mesh", shape: {2, 1, 2}}
      # Shard all three dimensions on corresponding mesh axes
      input_shardings = [%{0 => [0], 1 => [1], 2 => [2]}]

      # For mesh {2, 1, 2}, we have 4 partitions
      # Each partition gets {4, 4, 1} (full tensor {8, 4, 2} / {2, 1, 2})
      args = [
        [Nx.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])],
        [Nx.tensor([[[8, 9], [10, 11]], [[12, 13], [14, 15]]])],
        [Nx.tensor([[[16, 17], [18, 19]], [[20, 21], [22, 23]]])],
        [Nx.tensor([[[24, 25], [26, 27]], [[28, 29], [30, 31]]])]
      ]

      result = EXLA.to_mlir_module(fun, args, mesh: mesh, input_shardings: input_shardings)

      expected_mlir = """
      module {
        sdy.mesh @"3d_mesh" = <["axis_0"=2, "axis_1"=1, "axis_2"=2]>
        func.func public @main(%arg0: tensor<4x2x4xi32> {sdy.sharding = #sdy.sharding<@"3d_mesh", [{"axis_0", ?}p0, {"axis_1", ?}p0, {"axis_2", ?}p0]>}) -> tensor<4x2x4xi32> {
          %c = stablehlo.constant dense<2> : tensor<i32>
          %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4x2x4xi32>
          %1 = stablehlo.multiply %0, %arg0 : tensor<4x2x4xi32>
          return %1 : tensor<4x2x4xi32>
        }
      }
      """

      assert expected_mlir == result.mlir_module

      assert [result0, result1, result2, result3] =
               EXLA.shard_jit(fun, mesh, input_shardings: input_shardings).(args)

      assert_equal(result0, Nx.tensor([[[0, 2], [4, 6]], [[8, 10], [12, 14]]]))
      assert result0.data.buffer.device_id == 0
      assert_equal(result1, Nx.tensor([[[16, 18], [20, 22]], [[24, 26], [28, 30]]]))
      assert result1.data.buffer.device_id == 1
      assert_equal(result2, Nx.tensor([[[32, 34], [36, 38]], [[40, 42], [44, 46]]]))
      assert result2.data.buffer.device_id == 2
      assert_equal(result3, Nx.tensor([[[48, 50], [52, 54]], [[56, 58], [60, 62]]]))
      assert result3.data.buffer.device_id == 3
    end

    @moduletag :multi_device
    test "generates correct MLIR with multi-axis sharding" do
      fun = fn x -> Nx.transpose(x) end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}
      # Shard first dimension on both mesh axes 0 and 1
      input_shardings = [%{0 => [0, 1], 1 => []}]

      full_input = Nx.iota({4, 4})

      # For mesh {2, 2}, we have 4 partitions
      # Input: shape {4, 4} sharded as [[0, 1], []]
      # First dim sharded on 2 axes: 2 * 2 = 4, so each partition gets {4/4, 4} = {1, 4}
      args = [
        [full_input[0..0]],
        [full_input[1..1]],
        [full_input[2..2]],
        [full_input[3..3]]
      ]

      result = EXLA.to_mlir_module(fun, args, mesh: mesh, input_shardings: input_shardings)

      expected_mlir = """
      module {
        sdy.mesh @mesh = <["axis_0"=2, "axis_1"=2]>
        func.func public @main(%arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"axis_0", "axis_1", ?}p0, {?}p0]>}) -> tensor<4x4xi32> {
          %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xi32>) -> tensor<4x4xi32>
          return %0 : tensor<4x4xi32>
        }
      }
      """

      assert expected_mlir == result.mlir_module

      assert [result0, result1, result2, result3] =
               EXLA.shard_jit(fun, mesh, input_shardings: input_shardings).(args)

      assert_equal(result0, Nx.tensor([[0], [1], [2], [3]]))
      assert result0.data.buffer.device_id == 0
      assert_equal(result1, Nx.tensor([[4], [5], [6], [7]]))
      assert result1.data.buffer.device_id == 1
      assert_equal(result2, Nx.tensor([[8], [9], [10], [11]]))
      assert result2.data.buffer.device_id == 2
      assert_equal(result3, Nx.tensor([[12], [13], [14], [15]]))
      assert result3.data.buffer.device_id == 3
    end

    @moduletag :multi_device
    test "generates correct MLIR with multiple inputs" do
      fun = fn x, y, z -> Nx.add(Nx.multiply(x, y), z) end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}

      input_shardings = [
        # x: shard dim 0 on axis 0, dim 1 on axis 1
        %{0 => [0], 1 => [1]},
        # y: shard dim 0 on axis 0, dim 1 not sharded (logical shape {8,1})
        %{0 => [0]},
        # z: shard dim 0 on axis 0, dim 1 not sharded (logical shape {8,1})
        %{0 => [0]}
      ]

      # For mesh {2, 2}, we have 4 partitions
      # x: {8, 2} sharded [[0], [1]] -> each partition gets {4, 1}
      # y: {8, 1} sharded [[0], []] -> each partition gets {4, 1}
      # z: {8, 1} sharded [[0], []] -> each partition gets {4, 1}
      # output: {8, 2} sharded [[0], [1]] -> each partition gets {4, 1}
      #
      # Logical computation: (x * y) + z where y and z are broadcast to {8,2}
      # x: [[0,10],[1,11],[2,12],[3,13],[4,14],[5,15],[6,16],[7,17]]
      # y: [[1],[1],[1],[1],[2],[2],[2],[2]] broadcast to [[1,1],[1,1]...]
      # z: [[100],[100],[100],[100],[100],[100],[100],[100]] broadcast to [[100,100],[100,100]...]
      args = [
        # partition 0: x rows[0-3]col[0], y rows[0-3], z rows[0-3]
        [
          Nx.tensor([[0], [1], [2], [3]]),
          Nx.tensor([[1], [1], [1], [1]]),
          Nx.tensor([[100], [100], [100], [100]])
        ],
        # partition 1: x rows[0-3]col[1], y rows[0-3], z rows[0-3]
        [
          Nx.tensor([[10], [11], [12], [13]]),
          Nx.tensor([[1], [1], [1], [1]]),
          Nx.tensor([[100], [100], [100], [100]])
        ],
        # partition 2: x rows[4-7]col[0], y rows[4-7], z rows[4-7]
        [
          Nx.tensor([[4], [5], [6], [7]]),
          Nx.tensor([[2], [2], [2], [2]]),
          Nx.tensor([[100], [100], [100], [100]])
        ],
        # partition 3: x rows[4-7]col[1], y rows[4-7], z rows[4-7]
        [
          Nx.tensor([[14], [15], [16], [17]]),
          Nx.tensor([[2], [2], [2], [2]]),
          Nx.tensor([[100], [100], [100], [100]])
        ]
      ]

      result =
        EXLA.to_mlir_module(fun, args,
          mesh: mesh,
          input_shardings: input_shardings
        )

      expected_mlir = """
      module {
        sdy.mesh @mesh = <["axis_0"=2, "axis_1"=2]>
        func.func public @main(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"axis_0", ?}p0, {"axis_1", ?}p0]>}, %arg1: tensor<8x1xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"axis_0", ?}p0, {?}p0]>}, %arg2: tensor<8x1xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"axis_0", ?}p0, {?}p0]>}) -> tensor<8x2xi32> {
          %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<8x1xi32>) -> tensor<8x2xi32>
          %1 = stablehlo.multiply %arg0, %0 : tensor<8x2xi32>
          %2 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<8x1xi32>) -> tensor<8x2xi32>
          %3 = stablehlo.add %1, %2 : tensor<8x2xi32>
          return %3 : tensor<8x2xi32>
        }
      }
      """

      assert expected_mlir == result.mlir_module

      results =
        EXLA.shard_jit(fun, mesh, input_shardings: input_shardings).(args)

      assert length(results) == 4

      # Each device should have {4, 1} output (sharded on both axes)
      # Device 0: x[0-3,0] * y[0-3] + z[0-3]
      #         = [[0],[1],[2],[3]] * [[1],[1],[1],[1]] + [[100],[100],[100],[100]]
      #         = [[0],[1],[2],[3]] + [[100],[100],[100],[100]]
      #         = [[100],[101],[102],[103]]
      assert_equal(Enum.at(results, 0), Nx.tensor([[100], [101], [102], [103]]))

      # Device 1: x[0-3,1] * y[0-3] + z[0-3]
      #         = [[10],[11],[12],[13]] * [[1],[1],[1],[1]] + [[100],[100],[100],[100]]
      #         = [[10],[11],[12],[13]] + [[100],[100],[100],[100]]
      #         = [[110],[111],[112],[113]]
      assert_equal(Enum.at(results, 1), Nx.tensor([[110], [111], [112], [113]]))

      # Device 2: x[4-7,0] * y[4-7] + z[4-7]
      #         = [[4],[5],[6],[7]] * [[2],[2],[2],[2]] + [[100],[100],[100],[100]]
      #         = [[8],[10],[12],[14]] + [[100],[100],[100],[100]]
      #         = [[108],[110],[112],[114]]
      assert_equal(Enum.at(results, 2), Nx.tensor([[108], [110], [112], [114]]))

      # Device 3: x[4-7,1] * y[4-7] + z[4-7]
      #         = [[14],[15],[16],[17]] * [[2],[2],[2],[2]] + [[100],[100],[100],[100]]
      #         = [[28],[30],[32],[34]] + [[100],[100],[100],[100]]
      #         = [[128],[130],[132],[134]]
      assert_equal(Enum.at(results, 3), Nx.tensor([[128], [130], [132], [134]]))

      # Verify device IDs
      device_ids = for r <- results, do: r.data.buffer.device_id
      assert Enum.sort(device_ids) == [0, 1, 2, 3]
    end
  end

  describe "mesh validation" do
    test "raises when input_shardings provided without mesh" do
      fun = fn x -> Nx.add(x, 1) end
      # For non-sharded case, we don't need list of lists
      args = [Nx.iota({8, 2})]

      input_shardings = [%{0 => [0]}]

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
      input_shardings = [%{0 => [0]}]

      # For mesh {2, 2}, we have 4 partitions
      # Each partition has 2 inputs
      args = List.duplicate([Nx.iota({4, 1}), Nx.iota({4, 1})], 4)

      assert_raise ArgumentError,
                   ~r/expected 2 input sharding configuration.*got 1/,
                   fn ->
                     EXLA.to_mlir_module(fun, args,
                       mesh: mesh,
                       input_shardings: input_shardings
                     )
                   end
    end
  end

  describe "sharding validation" do
    test "raises when axis index is out of bounds for mesh" do
      fun = fn x -> Nx.add(x, 1) end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}
      # Mesh has 2 axes (0 and 1), but we reference axis 2
      input_shardings = [%{0 => [2]}]

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
      input_shardings = [%{0 => [0], 1 => [0]}]

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
      input_shardings = [%{0 => [0], 1 => [1], 2 => []}]

      # For mesh {2, 2}, we have 4 partitions
      args = List.duplicate([Nx.iota({4, 2})], 4)

      assert_raise ArgumentError, fn ->
        EXLA.shard_jit(fun, mesh, input_shardings: input_shardings).(args)
      end
    end

    test "raises when negative axis is out of bounds" do
      fun = fn x -> Nx.add(x, 1) end

      mesh = %Mesh{name: "mesh", shape: {2}}
      # Tensor is rank 2, but -3 is out of bounds (only -1 and -2 are valid)
      input_shardings = [%{-3 => [0]}]

      args = List.duplicate([Nx.iota({4, 2})], 2)

      assert_raise ArgumentError,
                   ~r/given axis \(-3\) invalid for shape with rank 2/,
                   fn ->
                     EXLA.shard_jit(fun, mesh, input_shardings: input_shardings).(args)
                   end
    end

    test "raises when positive axis is out of bounds" do
      fun = fn x -> Nx.add(x, 1) end

      mesh = %Mesh{name: "mesh", shape: {2}}
      # Tensor is rank 2, but axis 3 is out of bounds
      input_shardings = [%{3 => [0]}]

      args = List.duplicate([Nx.iota({4, 2})], 2)

      assert_raise ArgumentError,
                   ~r/given axis \(3\) invalid for shape with rank 2/,
                   fn ->
                     EXLA.shard_jit(fun, mesh, input_shardings: input_shardings).(args)
                   end
    end
  end

  describe "num_partitions calculation" do
    @moduletag :multi_device
    test "calculates correct num_partitions for 3D mesh" do
      fun = fn x -> Nx.add(x, 1) end

      # 2 * 1 * 2 = 4 partitions
      mesh = %Mesh{name: "mesh", shape: {2, 1, 2}}
      input_shardings = [%{0 => [0], 1 => [1], 2 => [2]}]

      # Input sharded [[0], [1], [2]] -> each partition gets {1, 2, 2}
      args =
        [
          [Nx.tensor([[[1, 1], [2, 2]]])],
          [Nx.tensor([[[3, 3], [4, 4]]])],
          [Nx.tensor([[[5, 5], [6, 6]]])],
          [Nx.tensor([[[7, 7], [8, 8]]])]
        ]

      assert [result0, result1, result2, result3] =
               EXLA.shard_jit(fun, mesh, input_shardings: input_shardings).(args)

      assert_equal(result0, Nx.tensor([[[2, 2], [3, 3]]]))
      assert_equal(result1, Nx.tensor([[[4, 4], [5, 5]]]))
      assert_equal(result2, Nx.tensor([[[6, 6], [7, 7]]]))
      assert_equal(result3, Nx.tensor([[[8, 8], [9, 9]]]))
    end
  end

  describe "complex sharding patterns" do
    test "handles fully replicated tensor" do
      fun = fn x -> Nx.add(x, 1) end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}
      # All dimensions replicated
      input_shardings = [%{}]

      # For mesh {2, 2}, we have 4 partitions
      # Input fully replicated -> each partition gets full {8, 4}
      args = List.duplicate([Nx.iota({8, 4})], 4)

      result =
        EXLA.to_mlir_module(fun, args,
          mesh: mesh,
          input_shardings: input_shardings
        )

      assert is_binary(result.mlir_module)
    end

    test "handles scalar input" do
      fun = fn x -> Nx.add(x, 1) end

      mesh = %Mesh{name: "mesh", shape: {2}}
      # Scalar has no dimensions to shard
      input_shardings = [%{}]

      # For mesh {2}, we have 2 partitions
      # Scalar is replicated across all partitions
      args = List.duplicate([Nx.tensor(5.0)], 2)

      result =
        EXLA.to_mlir_module(fun, args,
          mesh: mesh,
          input_shardings: input_shardings
        )

      assert is_binary(result.mlir_module)
    end

    test "handles mixed sharding and replication" do
      fun = fn x, y, z -> {Nx.add(x, y), Nx.multiply(y, z)} end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}

      input_shardings = [
        # x: both dimensions sharded
        %{0 => [0], 1 => [1]},
        # y: first dimension sharded, second replicated
        %{0 => [0]},
        # z: fully replicated
        %{}
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

      result =
        EXLA.to_mlir_module(fun, args,
          mesh: mesh,
          input_shardings: input_shardings
        )

      assert is_binary(result.mlir_module)
    end
  end

  describe "MLIR output format" do
    test "MLIR contains mesh definition with correct shape" do
      fun = fn x -> Nx.add(x, 1) end

      mesh = %Mesh{name: "test_mesh", shape: {2, 2}}
      input_shardings = [%{0 => [0]}]

      # For mesh {2, 2}, we have 4 partitions
      # Input sharded [[0], []] -> each partition gets {4, 2}
      args = List.duplicate([Nx.iota({4, 2})], 4)

      result =
        EXLA.to_mlir_module(fun, args,
          mesh: mesh,
          input_shardings: input_shardings
        )

      mlir = result.mlir_module

      # Check mesh name appears
      assert mlir =~ "test_mesh"

      # Check mesh definition exists
      assert mlir =~ ~r/sdy\.mesh/
    end

    test "MLIR contains function with sharding attributes" do
      fun = fn x -> Nx.add(x, 1) end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}
      input_shardings = [%{0 => [0], 1 => [1]}]

      # For mesh {2, 2}, we have 4 partitions
      # Input sharded [[0], [1]] -> each partition gets {4, 1}
      args = List.duplicate([Nx.iota({4, 1})], 4)

      result =
        EXLA.to_mlir_module(fun, args,
          mesh: mesh,
          input_shardings: input_shardings
        )

      mlir = result.mlir_module

      # Check that function arguments have sharding attributes
      assert mlir =~ ~r/sdy\.sharding/

      # Check that the main function exists
      assert mlir =~ ~r/func\.func.*@main/
    end

    test "supports named tensor dimensions" do
      fun = fn x -> Nx.add(x, 1) end

      mesh = %Mesh{name: "mesh", shape: {2}}
      # Use named dimensions instead of indices
      input_shardings = [%{:batch => [0]}]

      # For mesh {2}, we have 2 partitions
      # Named dimension :batch should map to first dimension (index 0)
      args = List.duplicate([Nx.iota({4, 2}, names: [:batch, :features])], 2)

      result =
        EXLA.to_mlir_module(fun, args,
          mesh: mesh,
          input_shardings: input_shardings
        )

      # Should generate valid MLIR with sharding on first dimension
      assert is_binary(result.mlir_module)
      assert result.mlir_module =~ ~r/sdy\.sharding/
    end

    test "supports negative axis indices" do
      fun = fn x -> Nx.add(x, 1) end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}
      # Use negative indices: -1 for last dimension, -2 for second to last
      input_shardings = [%{-2 => [0], -1 => [1]}]

      # For mesh {2, 2}, we have 4 partitions
      # -2 => dim 0 (first), -1 => dim 1 (last) for a rank-2 tensor
      # This should be equivalent to %{0 => [0], 1 => [1]}
      args = List.duplicate([Nx.iota({4, 1})], 4)

      result =
        EXLA.to_mlir_module(fun, args,
          mesh: mesh,
          input_shardings: input_shardings
        )

      expected_mlir = """
      module {
        sdy.mesh @mesh = <["axis_0"=2, "axis_1"=2]>
        func.func public @main(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"axis_0", ?}p0, {"axis_1", ?}p0]>}) -> tensor<8x2xi32> {
          %c = stablehlo.constant dense<1> : tensor<i32>
          %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8x2xi32>
          %1 = stablehlo.add %0, %arg0 : tensor<8x2xi32>
          return %1 : tensor<8x2xi32>
        }
      }
      """

      assert expected_mlir == result.mlir_module
    end

    test "supports mixed positive, negative, and named axes" do
      fun = fn x -> Nx.add(x, 1) end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}
      # Mix: positive index 0, negative index -1, and named dimension
      # For a 3D tensor [batch: 8, height: 4, width: 2]
      # 0 => batch (first), -1 => width (last)
      input_shardings = [%{0 => [0], -1 => [1]}]

      args = List.duplicate([Nx.iota({4, 2, 1}, names: [:batch, :height, :width])], 4)

      result =
        EXLA.to_mlir_module(fun, args,
          mesh: mesh,
          input_shardings: input_shardings
        )

      # Should generate valid MLIR with sharding on first and last dimensions
      assert is_binary(result.mlir_module)
      assert result.mlir_module =~ ~r/sdy\.sharding/
      # Check that it has sharding for both axis_0 and axis_1
      assert result.mlir_module =~ ~r/"axis_0"/
      assert result.mlir_module =~ ~r/"axis_1"/
    end

    # Test all standard Nx types
    @all_types [
      {:s, 8},
      {:s, 16},
      {:s, 32},
      {:s, 64},
      {:u, 8},
      {:u, 16},
      {:u, 32},
      {:u, 64},
      {:f, 8},
      {:f8_e4m3fn, 8},
      {:f, 16},
      {:f, 32},
      {:f, 64},
      {:bf, 16},
      {:c, 64},
      {:c, 128}
    ]

    for nx_type <- @all_types do
      # We use this custom conversion because Nx.Type.to_string/1 does not
      # follow StableHLO conventions
      type_string =
        case nx_type do
          {:s, size} -> "i#{size}"
          {:u, size} -> "ui#{size}"
          {:f, 8} -> "f8E5M2"
          {:f8_e4m3fn, 8} -> "f8E4M3FN"
          {:f, size} -> "f#{size}"
          {:bf, 16} -> "bf16"
          {:c, size} -> "complex<f#{div(size, 2)}>"
        end

      test "works with type #{type_string}" do
        fun = fn x ->
          Nx.add(x, x)
        end

        mesh = %Mesh{name: "mesh", shape: {2, 2}}
        input_shardings = [%{0 => [0]}]

        # Create sharded input (8x2 tensor split to 4x1 per device)
        args = [
          [Nx.reshape(Nx.tensor(1, type: unquote(nx_type)), {1})],
          [Nx.reshape(Nx.tensor(2, type: unquote(nx_type)), {1})],
          [Nx.reshape(Nx.tensor(3, type: unquote(nx_type)), {1})],
          [Nx.reshape(Nx.tensor(4, type: unquote(nx_type)), {1})]
        ]

        result =
          EXLA.to_mlir_module(fun, args,
            mesh: mesh,
            input_shardings: input_shardings
          )

        assert result.mlir_module == """
               module {
                 sdy.mesh @mesh = <["axis_0"=2, "axis_1"=2]>
                 func.func public @main(%arg0: tensor<2x#{unquote(type_string)}> {sdy.sharding = #sdy.sharding<@mesh, [{"axis_0", ?}p0]>}) -> tensor<2x#{unquote(type_string)}> {
                   %0 = stablehlo.add %arg0, %arg0 : tensor<2x#{unquote(type_string)}>
                   return %0 : tensor<2x#{unquote(type_string)}>
                 }
               }
               """

        assert results = EXLA.shard_jit(fun, mesh, input_shardings: input_shardings).(args)

        expected_results = [
          Nx.tensor([2], type: unquote(nx_type)),
          Nx.tensor([4], type: unquote(nx_type)),
          Nx.tensor([6], type: unquote(nx_type)),
          Nx.tensor([8], type: unquote(nx_type))
        ]

        Enum.zip_with([results, expected_results, 0..3], fn [result, expected, i] ->
          assert_equal(result, expected)
          assert result.data.buffer.device_id == i
        end)
      end
    end
  end

  describe "all_gather" do
    @moduletag :multi_device
    test "in all dims results in the same tensor in all devices" do
      fun = fn x, y ->
        Nx.add(x, y)
        |> Nx.Defn.Kernel.all_gather(all_gather_dim: 0, replica_groups: [[0]])
        |> Nx.Defn.Kernel.all_gather(all_gather_dim: 1, replica_groups: [[0]])
      end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}
      # First arg: 0..15 (8x2), shard dim 0 on mesh axis 0, dim 1 on mesh axis 1
      # Second arg: 100..115 (8x2), same sharding — makes sharded results easy to read
      input_shardings = [%{0 => [0], 1 => [1]}, %{0 => [0], 1 => [1]}]

      # For mesh {2, 2}, 4 partitions. Each gets {4, 1}. Full 8x2 row-major: [[0,1],[2,3],...,[14,15]].
      # Partition (axis_0, axis_1): (0,0)=rows 0-3 col 0, (0,1)=rows 0-3 col 1, (1,0)=rows 4-7 col 0, (1,1)=rows 4-7 col 1.
      # So partition 0 gets (0,0),(1,0),(2,0),(3,0) = 0,2,4,6; partition 1 gets (0,1),(1,1),... = 1,3,5,7; etc.
      args = [
        # partition 0: rows 0–3 col 0 -> 0,2,4,6 and 100,102,104,106
        [Nx.tensor([[0], [2], [4], [6]]), Nx.tensor([[100], [102], [104], [106]])],
        # partition 1: rows 0–3 col 1 -> 1,3,5,7 and 101,103,105,107
        [Nx.tensor([[1], [3], [5], [7]]), Nx.tensor([[101], [103], [105], [107]])],
        # partition 2: rows 4–7 col 0 -> 8,10,12,14 and 108,110,112,114
        [Nx.tensor([[8], [10], [12], [14]]), Nx.tensor([[108], [110], [112], [114]])],
        # partition 3: rows 4–7 col 1 -> 9,11,13,15 and 109,111,113,115
        [Nx.tensor([[9], [11], [13], [15]]), Nx.tensor([[109], [111], [113], [115]])]
      ]

      result = EXLA.to_mlir_module(fun, args, mesh: mesh, input_shardings: input_shardings)

      expected_mlir = """
      module {
        sdy.mesh @mesh = <["axis_0"=2, "axis_1"=2]>
        func.func public @main(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"axis_0", ?}p0, {"axis_1", ?}p0]>}, %arg1: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"axis_0", ?}p0, {"axis_1", ?}p0]>}) -> tensor<8x2xi32> {
          %0 = stablehlo.add %arg0, %arg1 : tensor<8x2xi32>
          %1 = "stablehlo.all_gather"(%0) <{all_gather_dim = 0 : i64, replica_groups = dense<0> : tensor<1x1xi64>}> : (tensor<8x2xi32>) -> tensor<8x2xi32>
          %2 = "stablehlo.all_gather"(%1) <{all_gather_dim = 1 : i64, replica_groups = dense<0> : tensor<1x1xi64>}> : (tensor<8x2xi32>) -> tensor<8x2xi32>
          return %2 : tensor<8x2xi32>
        }
      }
      """

      assert expected_mlir == result.mlir_module

      results = EXLA.shard_jit(fun, mesh, input_shardings: input_shardings).(args)

      assert length(results) == 4

      # After all_gather: full first arg 0..15 + full second 100..115 -> 100,102,...,130
      expected_result =
        Nx.tensor([
          [100, 102],
          [104, 106],
          [108, 110],
          [112, 114],
          [116, 118],
          [120, 122],
          [124, 126],
          [128, 130]
        ])

      Enum.with_index(results, fn result, i ->
        assert_equal(result, expected_result)
        assert result.data.buffer.device_id == i
      end)
    end

    @moduletag :multi_device
    test "can return partially sharded results" do
      fun = fn x, y ->
        x
        |> Nx.Defn.Kernel.all_gather(all_gather_dim: 1, replica_groups: [[0]])
        |> Nx.add(y)
      end

      mesh = %Mesh{name: "mesh", shape: {2, 2}}
      # Inputs sharded on both axes
      input_shardings = [%{0 => [0], 1 => [1]}, %{0 => [0]}]

      # Logical x: 8x2, y: 8x2. Each partition gets {4, 1} of x and {4, 2} of y
      args = [
        [
          Nx.tensor([[0], [1], [2], [3]]),
          Nx.tensor([[100, 101], [102, 103], [104, 105], [106, 107]])
        ],
        [
          Nx.tensor([[4], [5], [6], [7]]),
          Nx.tensor([[100, 101], [102, 103], [104, 105], [106, 107]])
        ],
        [
          Nx.tensor([[8], [9], [10], [11]]),
          Nx.tensor([[110, 111], [112, 113], [114, 115], [116, 117]])
        ],
        [
          Nx.tensor([[12], [13], [14], [15]]),
          Nx.tensor([[110, 111], [112, 113], [114, 115], [116, 117]])
        ]
      ]

      assert [result0, result1, result2, result3] =
               EXLA.shard_jit(fun, mesh, input_shardings: input_shardings).(args)

      # After gathering, devices 0 and 1 have the same data as each other, likewise for devices 2 and 3
      assert_equal(result0, Nx.tensor([[100, 105], [103, 108], [106, 111], [109, 114]]))
      assert result0.data.buffer.device_id == 0
      assert_equal(result0, Nx.tensor([[100, 105], [103, 108], [106, 111], [109, 114]]))
      assert result1.data.buffer.device_id == 1
      assert_equal(result2, Nx.tensor([[118, 123], [121, 126], [124, 129], [127, 132]]))
      assert result2.data.buffer.device_id == 2
      assert_equal(result3, Nx.tensor([[118, 123], [121, 126], [124, 129], [127, 132]]))
      assert result3.data.buffer.device_id == 3
    end
  end
end
