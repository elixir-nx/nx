defmodule Nx.Case do
  @moduledoc """
  Test case for tensor assertions
  """

  use ExUnit.CaseTemplate

  using do
    quote do
      import Nx.Case
    end
  end

  def assert_equal(left, right) do
    equals =
      left
      |> Nx.equal(right)
      # |> Nx.logical_or(Nx.is_nan(left) |> Nx.logical_and(Nx.is_nan(right)))
      |> Nx.all()
      |> Nx.to_number()

    if equals != 1 || Nx.shape(left) != Nx.shape(right) do
      flunk("""
      Tensor assertion failed.
      left: #{inspect(left)}
      right: #{inspect(right)}
      """)
    end
  end

  def assert_close(left, right) do
    equals =
      left
      |> Nx.all_close(right, atol: 1.0e-4, rtol: 1.0e-4)
      |> Nx.backend_transfer(Nx.BinaryBackend)

    if equals != Nx.tensor(1, type: {:u, 8}, backend: Nx.BinaryBackend) do
      flunk("""
      Tensor assertion failed.
      left: #{inspect(left)}
      right: #{inspect(right)}
      """)
    end
  end
end
