defmodule Torchx.Case do
  @moduledoc """
  Test case for tensor assertions
  """

  use ExUnit.CaseTemplate

  using do
    quote do
      import Torchx.Case
    end
  end

  def assert_all_close(left, right, opts \\ []) do
    atol = opts[:atol] || 1.0e-4
    rtol = opts[:rtol] || 1.0e-4

    assert left
           |> Nx.all_close?(right, atol: atol, rtol: rtol)
           |> Nx.backend_transfer(Nx.BinaryBackend) ==
             Nx.tensor(1, type: {:u, 8}, backend: Nx.BinaryBackend)
  end
end
