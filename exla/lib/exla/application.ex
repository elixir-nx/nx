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

    children = [
      {Registry, name: EXLA.Registry, keys: :unique},
      EXLA.Client,
      EXLA.LockedCache,
      EXLA.Logger
    ]

    Supervisor.start_link(children, name: __MODULE__, strategy: :one_for_one)
  end
end
