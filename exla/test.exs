:os.getpid() |> IO.inspect(label: "PID: ")
IO.gets("Press enter")
EXLA.jit(&Nx.logical_not/1, compiler_mode: :mlir).(Nx.tensor([-1, 0, 1])) |> IO.inspect()
