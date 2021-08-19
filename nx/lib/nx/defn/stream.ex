defmodule Nx.Defn.Stream do
  # Default implementation for Nx.Stream
  @moduledoc false
  use GenServer

  @doc false
  @enforce_keys [:pid, :input, :output]
  defstruct [:pid, :input, :output]

  @doc false
  def start_link(input, acc, fun) do
    {backend, backend_options} = Nx.default_backend()
    {:ok, pid} = GenServer.start_link(__MODULE__, {backend, backend_options, acc, fun})
    %Nx.Defn.Stream{input: input, output: Nx.to_template(acc), pid: pid}
  end

  @impl true
  def init({backend, backend_options, acc, fun}) do
    Nx.default_backend({backend, backend_options})
    {:ok, {:queue.new(), :queue.new(), acc, fun}}
  end

  @impl true
  def handle_cast({:send, input}, {output, waiting, acc, fun}) do
    {data, acc} = fun.(input, acc)

    case :queue.out(waiting) do
      {:empty, waiting} ->
        {:noreply, {:queue.in(data, output), waiting, acc, fun}}

      {{:value, from}, waiting} ->
        GenServer.reply(from, {:ok, data})
        {:noreply, {output, waiting, acc, fun}}
    end
  end

  @impl true
  def handle_call(:recv, from, {output, waiting, acc, fun}) do
    case :queue.out(output) do
      {:empty, output} ->
        {:noreply, {output, :queue.in(from, waiting), acc, fun}}

      {{:value, data}, output} ->
        {:reply, {:ok, data}, {output, waiting, acc, fun}}
    end
  end

  @impl true
  def handle_call(:done, _from, {output, waiting, acc, fun}) do
    if :queue.is_empty(output) do
      for from <- :queue.to_list(waiting) do
        GenServer.reply(from, :done)
      end

      {:stop, :normal, {:ok, acc}, {output, waiting, acc, fun}}
    else
      {:reply, :recv_pending, {output, waiting, acc, fun}}
    end
  end

  defimpl Nx.Stream do
    def send(%{pid: pid, input: input}, data) do
      unless compatible?(input, data) do
        raise ArgumentError, """
        Nx stream expected a tensor of type, shape, and names on send:

        #{inspect(input)}

        But got tensor:

        #{inspect(data)}
        """
      end

      GenServer.cast(pid, {:send, data})
    end

    def recv(%{pid: pid, output: output}) do
      case GenServer.call(pid, :recv, :infinity) do
        {:ok, data} ->
          unless compatible?(output, data) do
            raise ArgumentError, """
            Nx stream expected a tensor of type, shape, and names on recv:

            #{inspect(output)}

            But got tensor:

            #{inspect(data)}
            """
          end

          data

        :done ->
          raise "cannot recv from stream because it has been terminated"
      end
    end

    def done(%{pid: pid}) do
      case GenServer.call(pid, :done, :infinity) do
        {:ok, acc} ->
          acc

        :recv_pending ->
          raise "cannot mark stream as done when there are recv messages pending"
      end
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

    defp compatible?(_, _), do: false
  end
end
