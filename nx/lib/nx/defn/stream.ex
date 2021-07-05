defmodule Nx.Defn.Stream do
  # Default implementation for Nx.Stream
  @moduledoc false
  use GenServer

  @doc false
  @enforce_keys [:pid, :input, :acc]
  defstruct [:pid, :input, :acc]

  @doc false
  def start_link(input, acc, fun) do
    {backend, backend_options} = Nx.default_backend()
    {:ok, pid} = GenServer.start_link(__MODULE__, {backend, backend_options, acc, fun})
    %Nx.Defn.Stream{input: input, acc: acc, pid: pid}
  end

  @impl true
  def init({backend, backend_options, acc, fun}) do
    Nx.default_backend({backend, backend_options})
    {:ok, {:queue.new(), acc, fun}}
  end

  @impl true
  def handle_cast({:send, input}, {output, acc, fun}) do
    {entry, acc} = fun.(input, acc)
    {:noreply, {:queue.in(entry, output), acc, fun}}
  end

  @impl true
  def handle_call(:recv, _from, {output, acc, fun}) do
    {response, output} = :queue.out(output)
    {:reply, response, {output, acc, fun}}
  end

  @impl true
  def handle_call(:done, _from, {output, acc, fun}) do
    {:stop, :normal, acc, {output, acc, fun}}
  end

  defimpl Nx.Stream do
    def send(%{pid: pid, input: input}, data) do
      unless compatible?(input, data) do
        raise ArgumentError, """
        Nx stream expected a tensor of type, shape, and names:

        #{inspect(input)}

        But got tensor:

        #{inspect(data)}
        """
      end

      GenServer.cast(pid, {:send, data})
    end

    def recv(%{pid: pid}) do
      # TODO: See if recv can be blocking on EXLA and, if so, support it
      case GenServer.call(pid, :recv, :infinity) do
        {:value, value} -> value
        :empty -> raise "nothing to receive from Nx.Stream"
      end
    end

    def done(%{pid: pid}) do
      GenServer.call(pid, :done, :infinity)
    end

    defp compatible?(%Nx.Tensor{} = left, right), do: Nx.compatible?(left, right)

    defp compatible?(left, right) when tuple_size(left) == tuple_size(right) do
      Tuple.to_list(left)
      |> Enum.zip(Tuple.to_list(right))
      |> Enum.all?(fn {l, r} -> compatible?(l, r) end)
    end

    defp compatible?(left, right) when map_size(left) == map_size(right) do
      Enum.all?(left, fn {k, v1} ->
        case right do
          %{^k => v2} -> compatible?(v1, v2)
          %{} -> false
        end
      end)
    end
  end
end
