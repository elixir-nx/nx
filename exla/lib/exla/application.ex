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
    {:ok, device} = EXLA.MLIR.IREE.setup_runtime(~c"metal://0000000100000971")
    # {:ok, device} = EXLA.MLIR.IREE.setup_runtime(~c"local-sync://")
    :persistent_term.put({EXLA.MLIR.IREE, :device}, device)

    {:ok, instance} = EXLA.MLIR.IREE.create_instance()
    :persistent_term.put({EXLA.MLIR.IREE, :instance}, instance)

    :persistent_term.put({EXLA.Telemetry, :checkout}, {0, 0, 0, nil})

    :telemetry.attach(
      :exla_telemetry_checkout,
      [:exla, :mlir, :iree, :instance_pool, :checkout],
      fn _name, %{duration: duration}, _meta, _config ->
        {total, count, max, min} = :persistent_term.get({EXLA.Telemetry, :checkout})
        total = total + duration
        count = count + 1
        max = max(max, duration)
        min = min(min, duration)

        File.write(
          "/tmp/checkout.txt",
          "#{total / count / 1_000} ms | #{max / 1_000} ms | #{min / 1_000} ms\n"
        )

        :persistent_term.put({EXLA.Telemetry, :checkout}, {total, count, max, min})
      end,
      nil
    )

    :persistent_term.put({EXLA.Telemetry, :compile}, {0, 0, 0, nil})

    :telemetry.attach(
      :exla_telemetry_compile,
      [:exla, :mlir, :iree, :compile],
      fn _name, %{duration: duration}, _meta, _config ->
        {total, count, max, min} = :persistent_term.get({EXLA.Telemetry, :compile})
        total = total + duration
        count = count + 1
        max = max(max, duration)
        min = min(min, duration)

        File.write(
          "/tmp/compile.txt",
          "#{total / count / 1_000} ms | #{max / 1_000} ms | #{min / 1_000} ms\n"
        )

        :persistent_term.put({EXLA.Telemetry, :compile}, {total, count, max, min})
      end,
      nil
    )

    children = [
      EXLA.Logger,
      # {NimblePool,
      #  worker: {EXLA.MLIR.IREE.InstancePool, :pool_state},
      #  pool_size: System.schedulers_online(),
      #  name: EXLA.MLIR.IREE.InstancePool,
      #  lazy: true},
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
