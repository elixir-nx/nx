defmodule EXLA.Case do
  @moduledoc """
  Test case for tensor assertions
  """

  use ExUnit.CaseTemplate

  using do
    quote do
      import EXLA.Case
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

  def to_binary_backend(tensor) do
    Nx.backend_copy(tensor, Nx.BinaryBackend)
  end

  def is_mac_arm? do
    Application.fetch_env!(:exla, :is_mac_arm)
  end
end
