defmodule Nx.OptionalTest do
  use ExUnit.Case, async: true

  defmodule CustomImplTestBackend do
    defstruct [:state]
    alias Nx.BinaryBackend

    def from_binary(tensor, data, _opts) do
      %{tensor | data: %__MODULE__{state: data}}
    end

    def backend_transfer(%{data: %{state: data}} = tensor, backend, _opts) do
      %{tensor | data: struct(backend, state: data)}
    end

    def solve(_out, a, b) do
      send(self(), :called_custom_impl)

      out =
        Nx.LinAlg.solve(
          Nx.backend_transfer(a, BinaryBackend),
          Nx.backend_transfer(b, BinaryBackend)
        )

      Nx.backend_transfer(out, __MODULE__)
    end

    def determinant(_out, tensor) do
      send(self(), :called_custom_impl)

      out = Nx.LinAlg.determinant(Nx.backend_transfer(tensor, BinaryBackend))

      Nx.backend_transfer(out, __MODULE__)
    end
  end

  defmodule NonCustomImplTestBackend do
    defstruct [:state]
    alias Nx.BinaryBackend

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
      send(self(), :called_default_impl)

      out =
        Nx.dot(
          Nx.backend_transfer(a, BinaryBackend),
          Nx.backend_transfer(b, BinaryBackend)
        )

      Nx.backend_transfer(out, __MODULE__)
    end
  end

  describe "optional callbacks (def)" do
    test "calls the default impl if specific is not present" do
      a = Nx.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], backend: NonCustomImplTestBackend)
      b = Nx.tensor([1, 2, 3], backend: NonCustomImplTestBackend)

      Nx.LinAlg.solve(a, b)

      assert_receive :called_default_impl
    end

    test "calls the custom impl if it is present" do
      a = Nx.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], backend: CustomImplTestBackend)
      b = Nx.tensor([1, 2, 3], backend: CustomImplTestBackend)

      Nx.LinAlg.solve(a, b)

      assert_receive :called_custom_impl
    end
  end

  describe "optional callbacks (defn)" do
    test "calls the custom impl if it exists" do
      a = Nx.tensor([[1, 1], [2, 1]], backend: CustomImplTestBackend)

      Nx.LinAlg.determinant(a)

      assert_receive :called_custom_impl
    end
  end
end
