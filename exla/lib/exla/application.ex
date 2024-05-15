defmodule EXLA.Application do
  @moduledoc false

  def start(_args, _type) do
    # We need this in order for the compile NIF to start `ptxas` using a TF Subprocess.
    # The subprocess relies on `waitpid` which fails under normal circumstances because
    # ERTS sets SIGCHLD to SIGIGN.
    case :os.type() do
      {:win32, _} -> :ok
      _ -> :os.set_signal(:sigchld, :default)
    end

    EXLA.MLIR.IREE.global_initialize()
    {:ok, device} = EXLA.MLIR.IREE.setup_runtime()
    :persistent_term.put({EXLA.MLIR.IREE, :device}, device)

    children = [
      EXLA.Logger,
      {NimblePool,
       worker: {EXLA.MLIR.IREE.InstancePool, :pool_state},
       pool_size: 1,
       name: EXLA.MLIR.IREE.InstancePool,
       lazy: true},
      {NimblePool,
       worker: {EXLA.MLIR.ContextPool, :pool_state},
       pool_size: System.schedulers_online(),
       name: EXLA.MLIR.ContextPool,
       lazy: true},
      EXLA.Client,
      EXLA.Defn.Lock,
      EXLA.Defn.LockedCache,
      {Task.Supervisor, name: EXLA.Defn.TaskSupervisor}
    ]

    Supervisor.start_link(children, name: __MODULE__, strategy: :one_for_one)
  end
end
