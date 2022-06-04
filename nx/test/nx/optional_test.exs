defmodule Nx.OptionalTest do
  use ExUnit.Case, async: false

  import Nx.Defn, only: [defn: 2]

  defmodule CustomImplTestBackend do
    defstruct [:state]
    alias Nx.BinaryBackend

    def from_binary(tensor, data, _opts) do
      %{tensor | data: %__MODULE__{state: data}}
    end

    def backend_transfer(tensor, Nx.Tensor, opts) do
      backend_transfer(tensor, Nx.BinaryBackend, opts)
    end

    def backend_transfer(%{data: %{state: data}} = tensor, backend, _opts) do
      %{tensor | data: struct(backend, state: data)}
    end

    def iota(t, axis, opts) do
      t
      |> Nx.BinaryBackend.iota(axis, opts)
      |> Nx.backend_transfer(__MODULE__)
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

    def determinant(_out, t) do
      send(self(), :called_custom_impl)

      t
      |> Nx.backend_transfer()
      |> Nx.LinAlg.determinant()
      |> Nx.backend_transfer(__MODULE__)
    end
  end

  defmodule NonCustomImplTestBackend do
    defstruct [:state]
    alias Nx.BinaryBackend

    def from_binary(tensor, data, _opts) do
      %{tensor | data: %__MODULE__{state: data}}
    end

    def backend_transfer(tensor, Nx.Tensor, opts) do
      backend_transfer(tensor, Nx.BinaryBackend, opts)
    end

    def backend_transfer(%{data: %{state: data}} = tensor, backend, _opts) do
      %{tensor | data: struct(backend, state: data)}
    end

    def iota(t, axis, opts) do
      t
      |> Nx.BinaryBackend.iota(axis, opts)
      |> Nx.backend_transfer(__MODULE__)
    end

    def as_type(out, t) do
      t
      |> Nx.backend_transfer(BinaryBackend)
      |> Nx.as_type(out.type)
      |> Nx.backend_transfer(__MODULE__)
    end

    def add(_, a, b) do
      Nx.add(Nx.backend_transfer(a, BinaryBackend), Nx.backend_transfer(b, BinaryBackend))
      |> Nx.backend_transfer(__MODULE__)
    end

    def subtract(_, a, b) do
      Nx.subtract(Nx.backend_transfer(a, BinaryBackend), Nx.backend_transfer(b, BinaryBackend))
      |> Nx.backend_transfer(__MODULE__)
    end

    def reverse(_, t, axes) do
      t
      |> Nx.backend_transfer(BinaryBackend)
      |> Nx.reverse(axes: axes)
      |> Nx.backend_transfer(__MODULE__)
    end

    def product(_, t, opts) do
      t
      |> Nx.backend_transfer(BinaryBackend)
      |> Nx.product(opts)
      |> Nx.backend_transfer(__MODULE__)
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

    def gather(_, t, indices) do
      send(self(), {:called_default_impl, :gather})

      t
      |> Nx.backend_transfer(BinaryBackend)
      |> Nx.gather(indices)
      |> Nx.backend_transfer(__MODULE__)
    end

    def broadcast(_, t, shape, axes) do
      send(self(), {:called_default_impl, :reshape})

      t
      |> Nx.backend_transfer(BinaryBackend)
      |> Nx.broadcast(shape, axes: axes)
      |> Nx.backend_transfer(__MODULE__)
    end

    def reshape(out, t) do
      send(self(), {:called_default_impl, :reshape})

      t
      |> Nx.backend_transfer(BinaryBackend)
      |> Nx.reshape(out.shape)
      |> Nx.backend_transfer(__MODULE__)
    end

    def dot(_, a, _, _, b, _, _) do
      send(self(), {:called_default_impl, :dot})

      out =
        Nx.dot(
          Nx.backend_transfer(a, BinaryBackend),
          Nx.backend_transfer(b, BinaryBackend)
        )

      Nx.backend_transfer(out, __MODULE__)
    end
  end

  defn det_inspect(t) do
    t
    |> Nx.LinAlg.determinant()
    |> inspect_expr()
  end

  describe "optional callbacks (def)" do
    test "calls the default impl if specific is not present" do
      a = Nx.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], backend: NonCustomImplTestBackend)
      b = Nx.tensor([1, 2, 3], backend: NonCustomImplTestBackend)

      Nx.LinAlg.solve(a, b)

      assert_receive {:called_default_impl, :dot}
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

    test "calls the default impl if custom does not exist" do
      a = Nx.tensor([[1, 1], [2, 1]], backend: NonCustomImplTestBackend)

      Nx.LinAlg.determinant(a)

      assert_receive {:called_default_impl, :reshape}
    end
  end

  describe "inspect" do
    for backend <- [NonCustomImplTestBackend, CustomImplTestBackend] do
      test "works with defn call for backend #{backend}" do
        assert ExUnit.CaptureIO.capture_io(fn ->
                 assert 0 ==
                          {3, 3}
                          |> Nx.iota(backend: unquote(backend))
                          |> det_inspect()
                          |> Nx.backend_transfer(Nx.BinaryBackend)
                          |> Nx.sum()
                          |> Nx.to_number()
               end) =~ """
               #Nx.Tensor<
                 f32
               \s\s
                 Nx.Defn.Expr
                 parameter a:0       s64[3][3]
                 b = determinant a   f32
               >
               """
      end
    end

    test "works with direct call" do
      assert ExUnit.CaptureIO.capture_io(fn ->
               Nx.Defn.jit(&det_inspect/1, [Nx.iota({3, 3})])
             end) =~
               """
               #Nx.Tensor<
                 f32
               \s\s
                 Nx.Defn.Expr
                 parameter a:0       s64[3][3]
                 b = determinant a   f32
               >
               """
    end
  end
end
