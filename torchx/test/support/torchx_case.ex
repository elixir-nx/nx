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

  defmacro assert_tensor({:==, _, [left, right]}) do
    quote do
      assert Nx.backend_transfer(unquote(left), Nx.BinaryBackend) ==
               Nx.backend_transfer(unquote(right), Nx.BinaryBackend)
    end
  end
end
