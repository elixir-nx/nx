defmodule EXLAHelpers do
  @doc """
  Returns the default EXLA client.
  """
  def client(), do: EXLA.Client.fetch!(EXLA.Client.default_name())

  @doc """
  Returns peer nodes started by test helper.
  """
  def test_peer_nodes(), do: Application.get_env(:exla, :test_peer_nodes, [])

  @doc """
  Creates a host tensor and exports a read-only IPC pointer (default 0o400).
  """
  def export_host_ipc_pointer(list) do
    tensor = Nx.tensor(list, backend: {EXLA.Backend, client: :host})
    {:ok, pid} = Agent.start(fn -> tensor end)
    pointer = Nx.to_pointer(tensor, mode: :ipc)
    {pointer, tensor.type, tensor.shape, Nx.to_binary(tensor), pid}
  end

  @doc """
  Creates a host tensor and exports a writable IPC pointer (0o600).
  Returns `{pointer, type, shape}`.
  """
  def export_writable_ipc_pointer(list) do
    tensor = Nx.tensor(list, backend: {EXLA.Backend, client: :host})
    {:ok, pid} = Agent.start(fn -> tensor end)
    pointer = Nx.to_pointer(tensor, mode: :ipc, permissions: 0o600)
    {pointer, tensor.type, tensor.shape, pid}
  end

  @doc """
  Imports an IPC pointer on this node and stores the tensor in a named Agent
  so the mapping stays alive across calls.  Use `get_held_ipc_binary/0` to
  read and `stop_held_ipc/0` to clean up.
  """
  def hold_ipc_pointer(pointer, type, shape) do
    tensor = Nx.from_pointer({EXLA.Backend, client: :host}, pointer, type, shape)
    Agent.start(fn -> tensor end, name: :ipc_mutation_test)
  end

  @doc """
  Returns `Nx.to_binary/1` of the tensor held by `hold_ipc_pointer/3`.
  Always re-reads from the EXLA buffer (and therefore from the underlying shm).
  """
  def get_held_ipc_binary() do
    Agent.get(:ipc_mutation_test, &Nx.to_binary/1)
  end

  @doc """
  Stops the Agent started by `hold_ipc_pointer/3`.
  """
  def stop_held_ipc() do
    Agent.stop(:ipc_mutation_test)
  end

  @doc """
  Compiles the given function.

  It expects a list of shapes which will be given as parameters.
  """
  def compile(typespecs, opts \\ [], output \\ nil, fun) do
    compile_fn = fn builder ->
      params = EXLA.MLIR.Function.get_arguments(builder)
      [_callback_pid_param | params_without_callback_pid] = params

      fun
      |> apply([builder | params_without_callback_pid])
      |> then(&EXLA.MLIR.Value.func_return(builder, List.wrap(&1)))

      EXLA.MLIR.Module.compile(
        builder.module,
        client(),
        Enum.map(params, &EXLA.MLIR.Value.get_typespec/1),
        builder.return_typespecs,
        opts
      )
    end

    typespecs = [EXLA.Executable.callback_server_pid_typespec() | List.wrap(typespecs)]
    EXLA.MLIR.Module.new(typespecs, List.wrap(output), compile_fn)
  end

  @doc """
  Compiles and runs the given function.

  It expects a list of buffers which will be have their shapes
  used for compilation and then given on execution.
  """
  def run_one(args, opts \\ [], output \\ nil, fun) do
    exec = compile(Enum.map(args, & &1.typespec), opts, output, fun)
    [result] = EXLA.Executable.run(exec, [args], opts)
    result
  end
end
