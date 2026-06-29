defmodule Nx.Case do
  @moduledoc """
  Test case for Nx tensor assertions.
  """

  use ExUnit.CaseTemplate

  using do
    quote do
      import Nx.Case
      import Nx.Testing
    end
  end

  setup tags do
    # Set Logger metadata to track which test emits logs
    if tags[:test] do
      Logger.metadata(
        test: tags[:test],
        test_module: inspect(tags[:module])
      )
    end

    :ok
  end
end
