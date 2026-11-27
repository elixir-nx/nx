defmodule EXLA.CompileStackTest do
  use EXLA.Case, async: false

  @moduletag :cuda_required
  @moduletag :distributed

  # With xla 0.11.0, the cuDNN compiler pass of XLA needs more stack than a dirty scheduler of
  # the BEAM has. XLA compiles a dot as a cuDNN GEMM fusion when --xla_gpu_cudnn_gemm_fusion_level
  # is set, and the autotuner keeps that fusion when its backends are only cuDNN. XLA reads
  # XLA_FLAGS once per OS process, so the dot compiles on a peer node that starts with the flags.
  # Without a compile on a larger stack, the peer node crashes, and the call fails with
  # :noconnection instead of the whole test run. The HLO dump shows the cuDNN fusion.
  @tag :tmp_dir
  test "a dot compiles as a cuDNN GEMM fusion on the stack of the compile", %{tmp_dir: tmp_dir} do
    flags =
      "--xla_gpu_cudnn_gemm_fusion_level=3 --xla_gpu_experimental_autotune_backends=CUDNN " <>
        "--xla_dump_to=#{tmp_dir} --xla_dump_hlo_as_text"

    {:ok, peer, node} =
      :peer.start(%{
        name: :"cudnn_compile@127.0.0.1",
        env: [{~c"XLA_FLAGS", String.to_charlist(flags)}]
      })

    try do
      true = :erpc.call(node, :code, :set_path, [:code.get_path()])
      {:ok, _} = :erpc.call(node, :application, :ensure_all_started, [:exla])

      a = Nx.iota({512, 512}, type: :f32, backend: Nx.BinaryBackend)
      a = Nx.divide(a, Nx.tensor(512 * 512, type: :f32, backend: Nx.BinaryBackend))
      result = :erpc.call(node, EXLAHelpers, :dot_on_cuda, [Nx.as_type(a, :f16)])

      expected =
        EXLA.jit_apply(&Nx.dot(&1, &1), [a], client: :host) |> Nx.backend_copy(Nx.BinaryBackend)

      assert_all_close(Nx.as_type(result, :f32), expected, atol: 1.0e-2, rtol: 1.0e-2)

      hlo =
        tmp_dir
        |> Path.join("*after_optimizations.txt")
        |> Path.wildcard()
        |> Enum.map_join(&File.read!/1)

      assert hlo =~ "__cudnn$fusion"
    after
      # On a crash of the peer node, the peer process may already be gone.
      try do
        :peer.stop(peer)
      catch
        :exit, _ -> :ok
      end
    end
  end
end
