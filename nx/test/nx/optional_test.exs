defmodule Nx.OptionalTest do
  use ExUnit.Case, async: false

  defmodule CustomImplTestBackend do
    defstruct [:state]
    alias Nx.BinaryBackend

    def from_binary(tensor, data, _opts) do
      %{tensor | data: %__MODULE__{state: data}}
    end

    def backend_transfer(%{data: %{state: data}} = tensor, backend, _opts) do
      %{tensor | data: struct(backend, state: data)}
    end

    def iota(t, axis, opts) do
      t
      |> Nx.BinaryBackend.iota(axis, opts)
      |> Nx.backend_transfer(__MODULE__)
    end

    def sum(out, t, opts) do
      out |> Nx.BinaryBackend.sum(t, opts) |> Nx.backend_transfer(__MODULE__)
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

    def iota(t, axis, opts) do
      t
      |> Nx.BinaryBackend.iota(axis, opts)
      |> Nx.backend_transfer(__MODULE__)
    end

    def sum(out, t, opts) do
      out |> Nx.BinaryBackend.sum(t, opts) |> Nx.backend_transfer(__MODULE__)
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

    def reshape(%{data: %Nx.Defn.Expr{}} = tensor, opts) do
      send(self(), {:called_default_impl, :reshape})

      Nx.Defn.Expr.reshape(tensor, opts)
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

  defmodule DefnInspect do
    import Nx.Defn, only: [defn: 2]

    defn det(t) do
      t
      |> Nx.LinAlg.determinant()
      |> Nx.sum()
      |> inspect_expr()
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
                          |> DefnInspect.det()
                          |> Nx.backend_transfer(Nx.BinaryBackend)
                          |> Nx.to_number()
               end) == """
               #Nx.Tensor<
                 f32
               \s\s
                 Nx.Defn.Expr
                 parameter a:0                            s64[3][3]
                 b = determinant a                        f32
                 c = sum b, axes: nil, keep_axes: false   f32
               >
               """
      end
    end

    test "works with direct call" do
      assert {3, 3}
             |> Nx.iota(backend: Nx.Defn.Expr)
             |> Nx.LinAlg.determinant()
             |> Nx.sum()
             |> inspect() == """
             #Nx.Tensor<
               f32
             \s\s
               Nx.Defn.Expr
               parameter a:0                            s64[3][3]
               b = determinant a                        f32
               c = sum b, axes: nil, keep_axes: false   f32
             >
             """
    end
  end
end
