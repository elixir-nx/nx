defmodule DistributedServings do
  def multiply(parent, opts) do
    serving =
      Nx.Serving.new(fn defn -> Nx.Defn.jit(&Nx.multiply(&1, 2), defn) end)
      |> Nx.Serving.distributed_postprocessing(fn output ->
        send(parent, {:post, node(), output})
        output
      end)

    {:ok, _} = Nx.Serving.start_link([serving: serving] ++ opts)
    wait_for_parent(parent)
  end

  defp wait_for_parent(parent) do
    send(parent, :spawned)
    ref = Process.monitor(parent)

    receive do
      {:DOWN, ^ref, _, _, _} -> :ok
    end
  end
end
