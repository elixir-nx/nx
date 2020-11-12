defmodule Exla.LocalExecutable do
  alias __MODULE__, as: LocalExecutable
  alias Exla.Options.ExecutableRunOptions
  alias Exla.Tensor
  alias Exla.Client
  alias Exla.Shape

  @enforce_keys[:ref]
  defstruct [:ref]

  # TODO: Need client for device placement, but maybe we can separate these steps so run only depends on the executable
  def run(client = %Client{}, %LocalExecutable{ref: exec}, arguments, options \\ %ExecutableRunOptions{}) do
    # TODO: This should be a list
    # TODO: For now, we call to_device on every one, but to be more efficient, we should do some checks
    # rather than just doing this naively.
    # TODO: If we really want to get efficient we can do the transfers all it once in C++.
    argument_refs =
      arguments
      |> Tuple.to_list()
      |> Enum.map(&(Tensor.to_device(client, &1)))
      |> Enum.map(fn %Tensor{data: {:ref, ref}} -> ref end)
      |> List.to_tuple()
    {:ok, ref} = Exla.NIF.run(exec, argument_refs, options)
    # TODO: There's definitely a more efficient way to handle this
    {:ok, shape} = Exla.NIF.on_host_shape(ref)
    %Tensor{data: {:ref, ref}, shape: %Shape{ref: shape}, device: {:cpu, 0}}
  end
end
