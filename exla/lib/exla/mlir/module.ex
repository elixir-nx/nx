defmodule EXLA.MLIR.Module do
  @moduledoc """
  Representation of an MLIR module.
  """

  defstruct [:ref]

  alias EXLA.MLIR.Function

  alias EXLA.Client
  alias EXLA.Executable
  alias EXLA.Shape

  @doc """
  Creates a new MLIR module.
  """
  def new() do
    ref = EXLA.NIF.new_mlir_module() |> unwrap!()
    %__MODULE__{ref: ref}
  end

  @doc """
  Creates a new MLIR function with the given name belonging
  to the given MLIR module.
  """
  def create_function(
        %__MODULE__{ref: module_ref} = module,
        name,
        arg_shapes,
        %Shape{ref: return_shape_ref} = return_shape
      )
      when is_binary(name) do
    arg_shape_refs =
      Enum.map(arg_shapes, fn %Shape{ref: ref} -> ref end)

    ref =
      EXLA.NIF.create_mlir_function(module_ref, name, arg_shape_refs, return_shape_ref)
      |> unwrap!()

    %Function{module: module, ref: ref, name: name, return_shape: return_shape}
  end

  @doc """
  Compiles a module into an executable.

  ## Options

    * `:device_id` - the device id to compile to and run the executable on.
      Defaults to the `:default_device_id` on the client. If `:num_replicas`
      or `:num_partitions` are given, this option is ignored and the device
      id is set to `-1`.

    * `:num_replicas` - the number of replicas this computation will run on.
      It defaults to 1 but you can set it if you want to enable single-program
      multiple data.

    * `:use_spmd` - enables Single-Program Multiple-Data partioning.
      This is set to true if `:num_partitions` is more than one, otherwise is `false`.

  Currently those options do not have an effect as they related to running the
  same compiled executable on multiple replicas.

  Some options apply to TPU only:

    * `:num_partitions` - the number of partitions this computation will run on.

  """
  def compile(
        module = %__MODULE__{},
        client = %Client{},
        argument_shapes,
        return_shape,
        options \\ []
      ) do
    num_replicas = Keyword.get(options, :num_replicas, 1)
    num_partitions = Keyword.get(options, :num_partitions, 1)

    # JAX comments say SPMD can lead to subtle bugs so they only enable
    # when strictly necessary, which is when num_partitions is greater than 1.
    use_spmd = if Keyword.get(options, :use_spmd, true) or num_partitions >= 1, do: 1, else: 0

    device_id =
      if num_replicas > 1 or num_partitions > 1,
        do: -1,
        else: Keyword.get(options, :device_id, client.default_device_id)

    ref =
      EXLA.NIF.mlir_compile(
        client.ref,
        module.ref,
        Enum.map(argument_shapes, & &1.ref),
        num_replicas,
        num_partitions,
        use_spmd,
        device_id
      )
      |> unwrap!()

    %Executable{
      client: client,
      ref: ref,
      output_shape: return_shape,
      num_replicas: num_replicas,
      num_partitions: num_partitions,
      device_id: device_id
    }
  end

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end
