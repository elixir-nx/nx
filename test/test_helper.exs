case System.fetch_env("EXLA_TARGET") do
  {:ok, "cuda"} -> :ok
  _ -> ExUnit.configure(exclude: :cuda)
end

ExUnit.start()
