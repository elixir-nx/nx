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
      ref = EXLA.NIF.mlir_new_module(context)

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
        is_public
      )

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

    * `:use_spmd` - enables Single-Program Multiple-Data partitioning.
      This is set to true if `:num_partitions` is more than one, otherwise is `false`.

    * `:module_compilation` - either `:to_mlir` or `:to_pjrt`. The default is `:to_pjrt`.

      * `:to_pjrt` - the `EXLA.Executable` `:ref` field will hold the reference to a PjRt executable.
      * `:to_mlir` - the `EXLA.Executable` `:ref` field will hold the reference to an MLIR module.

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
    callback_server_pid = Keyword.get(options, :callback_server_pid, nil)

    # JAX comments say SPMD can lead to subtle bugs so they only enable
    # when strictly necessary, which is when num_partitions is greater than 1.
    use_spmd = Keyword.get(options, :use_spmd, true) or num_partitions >= 1

    device_id =
      if num_replicas > 1 or num_partitions > 1,
        do: -1,
        else: Keyword.get(options, :device_id, client.default_device_id)

    # Uncomment to debug the module MLIR source
    # module |> as_string() |> IO.puts()

    ref =
      case Keyword.get(options, :module_compilation, :to_pjrt) do
        :to_mlir ->
          module.ref

        :to_pjrt ->
          EXLA.NIF.mlir_compile(
            client.ref,
            module.ref,
            argument_typespecs,
            num_replicas,
            num_partitions,
            use_spmd,
            device_id,
            callback_server_pid
          )
      end

    %Executable{
      client: client,
      ref: ref,
      output_typespecs: return_typespecs,
      num_replicas: num_replicas,
      num_partitions: num_partitions,
      device_id: device_id,
      mesh: Keyword.get(options, :mesh),
      input_shardings: Keyword.get(options, :input_shardings)
    }
  end

  @doc """
  Adds a device mesh definition to the module.
  """
  def add_mesh(%__MODULE__{ref: module_ref}, %Nx.Mesh{name: name, shape: shape}) do
    # Convert shape tuple to axes list with auto-generated names
    # E.g., {2, 4} -> [{"axis_0", 2}, {"axis_1", 4}]
    axes =
      shape
      |> Tuple.to_list()
      |> Enum.with_index(fn size, idx -> {"axis_#{idx}", size} end)

    EXLA.NIF.mlir_add_mesh(module_ref, name, axes)
    :ok
  end

  @doc """
  Returns a human-readable representation of the module using MLIR
  syntax.
  """
  def as_string(module = %__MODULE__{}) do
    EXLA.NIF.mlir_module_to_string(module.ref)
  end
end
