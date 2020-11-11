defmodule Exla.LocalExecutable do
  alias __MODULE__, as: LocalExecutable
  alias Exla.Options.ExecutableRunOptions

  @enforce_keys[:ref]
  defstruct [:ref]

  def run(%LocalExecutable{ref: exec}, arguments, options \\ %ExecutableRunOptions{}) do
    case Exla.NIF.run(exec, arguments, options) do
      # TODO: Handle return
      {:ok, _} -> :ok
      {:error, msg} -> {:error, msg}
    end
  end
end
