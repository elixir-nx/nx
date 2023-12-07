defmodule EXLATest do
  use EXLA.Case, async: true

  exclude =
    if Nx.Defn.default_options()[:compiler_mode] == :mlir do
      [stream_cached?: 3]
    else
      []
    end

  doctest EXLA, except: exclude

  describe "integration" do
    @describetag :integration

    test "memory leak test" do
      test_data = Nx.broadcast(0.0, {1000, 1000}) |> Nx.backend_transfer(Nx.BinaryBackend)
      fun = EXLA.jit(fn x -> Nx.tan(x) end)

      for _ <- 1..2000 do
        fun.(test_data)
        Process.sleep(10)
        :erlang.garbage_collect()
      end
    end
  end
end
