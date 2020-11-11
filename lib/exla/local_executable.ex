defmodule Exla.LocalExecutable do
  alias __MODULE__, as: LocalExecutable
  alias Exla.Options.ExecutableRunOptions
  alias Exla.Tensor
  alias Exla.Client

  @enforce_keys[:ref]
  defstruct [:ref]

  # TODO: Need client for device placement, but maybe we can separate these steps so run only depends on the executable
  def run(client = %Client{}, %LocalExecutable{ref: exec}, arguments, options \\ %ExecutableRunOptions{}) do
    # TODO: This should be a list
    # TODO: For now, we call to_device on every one, but to be more efficient, we should do some checks
    # rather than just doing this naively.
    argument_refs =
      arguments
      |> Tuple.to_list()
      |> Enum.map(&(Tensor.to_device(client, &1)))
      |> Enum.map(fn %Tensor{data: {:ref, ref}} -> ref end)
      |> List.to_tuple()
    {:ok, ref} = Exla.NIF.run(exec, argument_refs, options)
    # TODO: Handle return
    :ok
  end
end
