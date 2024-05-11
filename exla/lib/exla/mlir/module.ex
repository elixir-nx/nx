defmodule EXLA.MLIR.Module do
  @moduledoc false
  # Representation of an MLIR module.

  defstruct [:ref]

  alias EXLA.MLIR.ContextPool
  alias EXLA.MLIR.Function

  alias EXLA.Client
  alias EXLA.Executable

  @doc """
  Creates a new MLIR module.
  """
  def new(arg_typespecs, return_typespecs, fun) when is_function(fun, 1) do
    ContextPool.checkout(fn context ->
      ref = context |> EXLA.NIF.mlir_new_module() |> unwrap!()

      %__MODULE__{ref: ref}
      |> create_function("main", arg_typespecs, return_typespecs, true)
      |> fun.()
    end)
  end

  @doc """
  Adds a new function to the MLIR module.
  """
  def add_function(module, name, arg_typespecs, return_typespecs) do
    create_function(module, name, arg_typespecs, return_typespecs, false)
  end

  defp create_function(
         %__MODULE__{ref: module_ref} = module,
         name,
         arg_typespecs,
         return_typespecs,
         is_public
       )
       when is_binary(name) do
    arg_types = EXLA.MLIR.Value.typespecs_to_mlir_types(arg_typespecs)
    return_types = EXLA.MLIR.Value.typespecs_to_mlir_types(return_typespecs)

    ref =
      EXLA.NIF.mlir_create_function(
        module_ref,
        name,
        arg_types,
        return_types,
        if(is_public, do: 1, else: 0)
      )
      |> unwrap!()

    %Function{module: module, ref: ref, name: name, return_typespecs: return_typespecs}
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
        argument_typespecs,
        return_typespecs,
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
        Enum.map(argument_typespecs, &EXLA.Typespec.nif_encode/1),
        num_replicas,
        num_partitions,
        use_spmd,
        device_id
      )
      |> unwrap!()

    %Executable{
      client: client,
      ref: ref,
      output_typespecs: return_typespecs,
      num_replicas: num_replicas,
      num_partitions: num_partitions,
      device_id: device_id
    }
  end

  @doc """
  Returns a human-readable representation of the module using MLIR
  syntax.
  """
  def to_string(module = %__MODULE__{}) do
    EXLA.NIF.mlir_module_to_string(module.ref) |> unwrap!()
  end

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end
