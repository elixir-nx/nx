defmodule Nx.BackendTest do
  use ExUnit.Case, async: true

  defmodule SolveTestBackend do
    defstruct [:state]
    alias Nx.BinaryBackend

    require Logger

    def from_binary(tensor, data, _opts) do
      %{tensor | data: %__MODULE__{state: data}}
    end

    def backend_transfer(%{data: %{state: data}} = tensor, backend, _opts) do
      %{tensor | data: struct(backend, state: data)}
    end

    def solve(_out, a, b) do
      Logger.info("called custom implementation")

      out =
        Nx.LinAlg.solve(
          Nx.backend_transfer(a, BinaryBackend),
          Nx.backend_transfer(b, BinaryBackend)
        )

      Nx.backend_transfer(out, __MODULE__)
    end
  end

  defmodule NonSolveTestBackend do
    defstruct [:state]
    alias Nx.BinaryBackend

    require Logger

    def from_binary(tensor, data, _opts) do
      %{tensor | data: %__MODULE__{state: data}}
    end

    def backend_transfer(%{data: %{state: data}} = tensor, backend, _opts) do
      %{tensor | data: struct(backend, state: data)}
    end

    def transpose(_, t, _) do
      t
      |> Nx.backend_transfer(BinaryBackend)
      |> Nx.transpose()
      |> Nx.backend_transfer(__MODULE__)
    end

    def qr(_, a, _) do
      a
      |> Nx.backend_transfer(BinaryBackend)
      |> Nx.LinAlg.qr()
      |> Nx.backend_transfer(__MODULE__)
    end

    def triangular_solve(_, a, b, _) do
      a
      |> Nx.backend_transfer(BinaryBackend)
      |> Nx.LinAlg.triangular_solve(Nx.backend_transfer(b, BinaryBackend))
      |> Nx.backend_transfer(__MODULE__)
    end

    def dot(_out, a, _, _, b, _, _) do
      Logger.info("called default implementation")

      out =
        Nx.dot(
          Nx.backend_transfer(a, BinaryBackend),
          Nx.backend_transfer(b, BinaryBackend)
        )

      Nx.backend_transfer(out, __MODULE__)
    end
  end

  setup_all do
    Process.register(self(), __MODULE__)
    :ok
  end

  describe "optional callbacks" do
    test "calls the default impl if specific is not present" do
      a = Nx.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], backend: NonSolveTestBackend)
      b = Nx.tensor([1, 2, 3], backend: NonSolveTestBackend)

      assert ExUnit.CaptureLog.capture_log(fn ->
               Nx.LinAlg.solve(a, b)
             end) =~ "called default implementation"
    end

    test "calls the custom impl if it is present" do
      a = Nx.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], backend: SolveTestBackend)
      b = Nx.tensor([1, 2, 3], backend: SolveTestBackend)

      assert ExUnit.CaptureLog.capture_log(fn ->
               Nx.LinAlg.solve(a, b)
             end) =~ "called custom implementation"
    end
  end
end
