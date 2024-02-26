defmodule EXLA.Case do
  @moduledoc """
  Test case for tensor assertions
  """

  use ExUnit.CaseTemplate

  using do
    quote do
      import EXLA.Case
    end
  end

  def compiler_mode do
    Application.get_env(:exla, :compiler_mode) || Nx.Defn.default_options()[:compiler_mode]
  end

  defmacro assert_equal(left, right) do
    # Assert against binary backend tensors to show diff on failure
    quote do
      assert unquote(left) |> to_binary_backend() == unquote(right) |> to_binary_backend()
    end
  end

  def to_binary_backend(tensor) do
    Nx.backend_copy(tensor, Nx.BinaryBackend)
  end

  def assert_all_close(left, right, opts \\ []) do
    atol = opts[:atol] || 1.0e-4
    rtol = opts[:rtol] || 1.0e-4

    equals =
      left
      |> Nx.all_close(right, atol: atol, rtol: rtol)
      |> Nx.backend_transfer(Nx.BinaryBackend)

    if equals != Nx.tensor(1, type: {:u, 8}, backend: Nx.BinaryBackend) do
      flunk("""
      expected

      #{inspect(left)}

      to be within tolerance of

      #{inspect(right)}
      """)
    end
  end
end
