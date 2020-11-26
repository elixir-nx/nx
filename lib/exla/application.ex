defmodule Exla.Application do
  @moduledoc false

  def start(_args, _type) do
    # We need this in order for the compile NIF to start `ptxas` using a TF Subprocess.
    # The subprocess relies on `waitpid` which fails under normal circumstances because
    # ERTS sets SIGCHLD to SIGIGN.
    :os.set_signal(:sigchld, :default)

    Supervisor.start_link([Exla.LockedCache], name: __MODULE__, strategy: :one_for_one)
  end
end
