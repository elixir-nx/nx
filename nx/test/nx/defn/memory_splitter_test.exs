defmodule Nx.Defn.MemorySplitterTest do
  use ExUnit.Case, async: true

  import Nx.Defn

  describe "split_by_memory/2" do
    test "splits expression based on memory limits" do
      # Create a simple expression that uses increasing amounts of memory
      expr = Nx.Defn.debug_expr(fn x ->
        a = Nx.multiply(x, 2)      # Small operation
        b = Nx.dot(a, a)            # Larger operation (dot product)
        c = Nx.add(b, a)            # Small operation
        d = Nx.dot(c, c)            # Larger operation
        Nx.sum(d)                   # Reduction
      end).(Nx.iota({100, 100}, type: :f32))

      # Set memory limits that will force splits
      # Each float32 element is 4 bytes, so 100x100 = 40KB base
      memory_limits = [
        50_000,   # 50KB - should fit first multiply
        200_000,  # 200KB - should fit first dot
        300_000   # 300KB - should fit remaining ops
      ]

      stages = Nx.Defn.MemorySplitter.split_by_memory(expr, memory_limits)

      # Should have multiple stages
      assert length(stages) > 1

      # Each stage should have proper structure
      Enum.each(stages, fn stage ->
        assert %Nx.Defn.Graph.Stage{} = stage
        assert is_reference(stage.id)
        assert %Nx.Tensor{data: %Nx.Defn.Expr{}} = stage.expr
        assert is_list(stage.arguments)
      end)
    end

    test "handles single runner with unlimited memory" do
      expr = Nx.Defn.debug_expr(fn x, y ->
        x
        |> Nx.add(y)
        |> Nx.multiply(2)
        |> Nx.sin()
      end).(Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6]))

      # Single runner with effectively unlimited memory
      memory_limits = [:infinity]

      stages = Nx.Defn.MemorySplitter.split_by_memory(expr, memory_limits)

      # Should be just one stage
      assert length(stages) == 1
    end

    test "respects accumulator state across operations" do
      # Create expression with known memory usage pattern
      expr = Nx.Defn.debug_expr(fn x ->
        # Each operation on 10x10 f32 tensor = 400 bytes output
        a = Nx.add(x, 1)        # 400 bytes
        b = Nx.multiply(a, 2)   # 400 bytes
        c = Nx.subtract(b, 3)   # 400 bytes
        d = Nx.divide(c, 4)     # 400 bytes
        e = Nx.add(d, 5)        # 400 bytes
        e
      end).(Nx.iota({10, 10}, type: :f32))

      # Set limits that force split after 3 operations
      memory_limits = [
        1200,  # Should fit 3 operations
        1200   # Should fit remaining 2 operations
      ]

      stages = Nx.Defn.MemorySplitter.split_by_memory(expr, memory_limits)

      assert length(stages) == 2
    end
  end

  describe "estimate_memory_usage/1" do
    test "estimates memory for simple expression" do
      expr = Nx.Defn.debug_expr(fn x ->
        x
        |> Nx.multiply(2)
        |> Nx.add(1)
      end).(Nx.iota({100}, type: :f32))

      %{total_memory: total, peak_memory: peak} =
        Nx.Defn.MemorySplitter.estimate_memory_usage(expr)

      # Should have some memory usage
      assert total > 0
      assert peak >= total
    end

    test "accounts for operations with temporary memory" do
      expr = Nx.Defn.debug_expr(fn x ->
        Nx.dot(x, x)  # Dot product needs temporary memory
      end).(Nx.iota({50, 50}, type: :f32))

      %{total_memory: total, peak_memory: peak} =
        Nx.Defn.MemorySplitter.estimate_memory_usage(expr)

      # Dot product should use more memory than just output
      output_size = 50 * 50 * 4  # 50x50 float32
      assert total > output_size
    end
  end

  describe "integration with Graph.run/2" do
    test "split stages can be executed" do
      # Define a computation
      expr = Nx.Defn.debug_expr(fn x, y ->
        a = Nx.add(x, y)
        b = Nx.multiply(a, 2)
        c = Nx.sin(b)
        Nx.sum(c)
      end).(Nx.tensor([1.0, 2.0, 3.0]), Nx.tensor([4.0, 5.0, 6.0]))

      # Split with tight memory constraints
      memory_limits = [100, 100, 100]
      stages = Nx.Defn.MemorySplitter.split_by_memory(expr, memory_limits)

      # Execute the stages
      args = [Nx.tensor([1.0, 2.0, 3.0]), Nx.tensor([4.0, 5.0, 6.0])]
      result = Nx.Defn.Graph.run(stages, args)

      # Verify result is a tensor
      assert %Nx.Tensor{} = result

      # Could also verify the actual computation if needed
      # expected = Nx.sum(Nx.sin(Nx.multiply(Nx.add(args[0], args[1]), 2)))
      # assert_all_close(result, expected)
    end
  end
end
