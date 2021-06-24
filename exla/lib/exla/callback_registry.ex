defmodule EXLA.CallbackRegistry do
  @moduledoc """
  Callback registry which handles Host Callbacks.
  """
  use GenServer

  alias EXLA.Shape

  @doc """
  Starts Callback Registry.
  """
  def start_link(_) do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  @doc """
  Registers `callback_fun` with `name`.
  """
  def register(name, operand, callback_fun) do
    # TODO(seanmor5): Hash on callback_fun?
    GenServer.call(__MODULE__, {:register, name, operand, callback_fun})
  end

  @doc """
  Runs registered callback function.
  """
  def run(name) do
    GenServer.call(__MODULE__, {:run, name})
  end

  @impl true
  def init(_) do
    {:ok, %{}}
  end

  @impl true
  def handle_call({:register, name, operand, fun}, _from, callbacks) do
    # No need to register new names
    callbacks =
      case callbacks do
        %{^name => _} = callbacks ->
          callbacks

        %{} ->
          Map.put(callbacks, name, fun)
      end

    result = EXLA.Op.host_callback(operand, self(), name)
    {:reply, result, callbacks}
  end

  @impl true
  def handle_info({:run, name, data, shape}, callbacks) do
    case callbacks[name] do
      {fun, _} when is_function(fun) ->
        %Shape{dtype: type, dims: dims} = Shape.get_shape_info(shape)
        tensor =
          data
          |> Nx.from_binary(type: type)
          |> Nx.reshape(dims)

        fun.(tensor)

      _ ->
        raise "Callback #{name} not found"
    end
  end
end