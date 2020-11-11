defmodule Exla.LocalExecutable do
  alias __MODULE__, as: LocalExecutable
  alias Exla.Options.ExecutableRunOptions

  @enforce_keys[:ref]
  defstruct [:ref]

  def run(%LocalExecutable{ref: exec}, arguments, options \\ %ExecutableRunOptions{}) do
    {:ok, ref} = Exla.NIF.run(exec, arguments, options)
    # TODO: Handle return
    :ok
  end
end
