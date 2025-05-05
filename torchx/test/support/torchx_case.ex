defmodule Torchx.Case do
  @moduledoc """
  Test case for tensor assertions
  """

  use ExUnit.CaseTemplate

  using do
    quote do
      import Nx.Testing
    end
  end
end
