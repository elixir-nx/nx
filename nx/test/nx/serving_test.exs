defmodule Nx.ServingTest do
  use ExUnit.Case, async: true

  # TODO: test custom client pre/post

  defmodule Simple do
    @behaviour Nx.Serving

    @impl true
    def init(type, pid) do
      send(pid, {:init, type})
      {:ok, pid}
    end

    @impl true
    def handle_batch(batch, pid) do
      send(pid, {:batch, batch})

      fun = fn ->
        send(pid, :execute)
        {Nx.Defn.jit_apply(&Nx.multiply(&1, 2), [batch]), :metadata}
      end

      {:execute, fun, pid}
    end
  end

  test "run" do
    serving = Nx.Serving.new(Simple, self())
    batch = Nx.Batch.stack([Nx.tensor([1, 2, 3])])

    assert Nx.Serving.run(serving, batch) == Nx.tensor([[2, 4, 6]])
    assert_received {:init, :inline}
    assert_received {:batch, batch}
    assert_received :execute
    assert batch.size == 1
    assert Nx.Defn.jit_apply(&Function.identity/1, [batch]) == Nx.tensor([[1, 2, 3]])
  end
end
