defmodule EXLA.Profiler do
  defstruct [:ref]

  @doc """
  Starts a profiler session.
  """
  def start() do
    ref = EXLA.NIF.start_profiler() |> unwrap!()
    struct(__MODULE__, ref: ref)
  end

  @doc """
  Stops and exports a profile.
  """
  def stop_and_export(%__MODULE__{ref: ref}, location) do
    :ok = EXLA.NIF.stop_profiler(ref, location)
  end

  defp unwrap!({:ok, res}), do: res
  defp unwrap!(error), do: raise "#{error}"
end