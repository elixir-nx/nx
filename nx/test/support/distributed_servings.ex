defmodule DistributedServings do
  import Nx.Defn

  def multiply(parent, opts) do
    serving =
      Nx.Serving.jit(&Nx.multiply(&1, 2))
      |> Nx.Serving.distributed_postprocessing(fn output ->
        send(parent, {:post, node(), output})
        output
      end)

    {:ok, _} = Nx.Serving.start_link([serving: serving] ++ opts)
    wait_for_parent(parent)
  end

  def add_five_round_about(parent, opts) do
    serving =
      Nx.Serving.jit(&add_five_round_about/1)
      |> Nx.Serving.streaming(hooks: [:double, :plus_ten])
      |> Nx.Serving.distributed_postprocessing(fn output ->
        Stream.transform(output, :ok, fn data, :ok -> {[{data, node()}], :ok} end)
      end)

    {:ok, _} = Nx.Serving.start_link([serving: serving] ++ opts)
    wait_for_parent(parent)
  end

  defnp add_five_round_about(batch) do
    batch
    |> Nx.multiply(2)
    |> hook(:double)
    |> Nx.add(10)
    |> hook(:plus_ten)
    |> Nx.divide(2)
    |> hook(:to_be_ignored)
  end

  defp wait_for_parent(parent) do
    ref = Process.monitor(parent)

    receive do
      {:DOWN, ^ref, _, _, _} -> :ok
    end
  end
end
