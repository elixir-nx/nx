defmodule Candlex.Case do
  @moduledoc """
  Test case for tensor assertions
  """

  use ExUnit.CaseTemplate

  using do
    quote do
      import Candlex.Case
    end
  end

  def assert_equal(left, right) do
    equals =
      left
      |> Nx.equal(right)
      # |> Nx.logical_or(Nx.is_nan(left) |> Nx.logical_and(Nx.is_nan(right)))
      |> Nx.all()
      |> Nx.to_number()

    if equals != 1 do
      flunk("""
      Tensor assertion failed.
      left: #{inspect(left)}
      right: #{inspect(right)}
      """)
    end
  end
end
