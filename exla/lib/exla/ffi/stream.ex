defmodule EXLA.FFI.Stream do
  @moduledoc """
  Experimental BEAM process for managing infeed/outfeed queues for XLA FFI custom calls.

  This module provides a GenServer that acts as a stream coordinator between
  EXLA computations and Elixir code, using XLA FFI custom calls instead of
  the traditional token-based infeed/outfeed mechanism.

  ## Usage

      # Start a stream process
      {:ok, pid} = EXLA.FFI.Stream.start_link(name: :my_stream)

      # Push values to be consumed by infeed
      EXLA.FFI.Stream.push_infeed(:my_stream, tensor_value)

      # Pop values produced by outfeed
      {:ok, result} = EXLA.FFI.Stream.pop_outfeed(:my_stream)

  The stream process can be used with EXLA compiler options:

      Nx.Defn.default_options(
        compiler: EXLA,
        compiler_options: [stream_name: :my_stream]
      )

  """

  use GenServer
  require Logger

  @type name :: atom()
  @type stream_ref :: pid() | name()

  ## Public API

  @doc """
  Starts a stream process.

  ## Options

    * `:name` - Required. The name to register the process under.

  ## Examples

      {:ok, pid} = EXLA.FFI.Stream.start_link(name: :my_stream)

  """
  @spec start_link(Keyword.t()) :: GenServer.on_start()
  def start_link(opts) do
    name = Keyword.fetch!(opts, :name)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Pushes a value to the infeed queue.

  The value will be consumed by the next infeed custom call in the XLA computation.

  ## Examples

      EXLA.FFI.Stream.push_infeed(:my_stream, tensor_value)

  """
  @spec push_infeed(stream_ref(), term()) :: :ok
  def push_infeed(stream_ref, value) do
    GenServer.cast(stream_ref, {:push_infeed, value})
  end

  @doc """
  Pops a value from the outfeed queue.

  Returns `{:ok, value}` if a value is available, or `:empty` if the queue is empty.

  ## Examples

      case EXLA.FFI.Stream.pop_outfeed(:my_stream) do
        {:ok, value} -> process_value(value)
        :empty -> :no_data
      end

  """
  @spec pop_outfeed(stream_ref(), timeout()) :: {:ok, term()} | :empty
  def pop_outfeed(stream_ref, timeout \\ 5_000) do
    GenServer.call(stream_ref, :pop_outfeed, timeout)
  end

  @doc """
  Called by NIF via nif_call to get the next infeed value.

  This is an internal function called from the C++ custom call handler.
  Returns `{:ok, payload_term}` if a value is available, or `{:error, :empty}` if not.
  """
  @spec next_infeed(stream_ref(), timeout()) :: {:ok, term()} | {:error, :empty}
  def next_infeed(stream_ref, timeout \\ 5_000) do
    GenServer.call(stream_ref, :next_infeed, timeout)
  end

  ## GenServer callbacks

  @impl true
  def init(opts) do
    name = Keyword.fetch!(opts, :name)

    state = %{
      name: name,
      infeed: :queue.new(),
      outfeed: :queue.new()
    }

    {:ok, state}
  end

  @impl true
  def handle_cast({:push_infeed, value}, %{infeed: in_q} = state) do
    {:noreply, %{state | infeed: :queue.in(value, in_q)}}
  end

  @impl true
  def handle_call(:pop_outfeed, _from, state) do
    case :queue.out(state.outfeed) do
      {{:value, value}, outfeed} ->
        {:reply, {:ok, value}, %{state | outfeed: outfeed}}

      {:empty, _} ->
        {:reply, :empty, state}
    end
  end

  @impl true
  def handle_call(:next_infeed, _from, state) do
    case :queue.out(state.infeed) do
      {{:value, value}, infeed} ->
        {:reply, {:ok, value}, %{state | infeed: infeed}}

      {:empty, _} ->
        {:reply, {:error, :empty}, state}
    end
  end

  @impl true
  def handle_info({:exla_outfeed, payload}, state) do
    # Received a message from the NIF outfeed custom call
    outfeed = :queue.in(payload, state.outfeed)
    {:noreply, %{state | outfeed: outfeed}}
  end

  @impl true
  def handle_info(msg, state) do
    Logger.warning("EXLA.FFI.Stream received unexpected message: #{inspect(msg)}")
    {:noreply, state}
  end
end
