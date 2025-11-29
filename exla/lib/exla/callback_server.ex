defmodule EXLA.CallbackServer do
  @moduledoc """
  Dispatcher and registry for `Nx.elixir_call/3` callbacks used by EXLA.

  This server has two responsibilities:

    * Assign a stable integer callback id for each Elixir function + output
      template pair that participates in `Nx.elixir_call/3` when using the
      EXLA compiler.

    * Receive callback requests from the native EXLA bridge thread, execute
      the Elixir function, validate the result against the expected output
      template, and reply back to native through a NIF.

  The native side is expected to:

    * Lower `:elixir_call` nodes to a CPU-only host `CustomCall` named
      `"exla_elixir_callback"` with a callback id encoded in its attributes.

    * Run a bridge thread that sends messages of the form:

          {:exla_elixir_call, callback_id :: integer, args :: [Nx.Tensor.t()], reply_tag :: term()}

      to this process and waits on a native future associated with `reply_tag`.

    * Provide a NIF `EXLA.NIF.elixir_callback_reply/2` that completes the
      native future when we send the reply back.
  """

  use GenServer

  require Logger

  @type callback_id :: non_neg_integer()

  defstruct next_id: 1,
            callbacks: %{}

  @type t :: %__MODULE__{
          next_id: non_neg_integer(),
          # We store the original function, its output template, and any
          # static (non-tensor) arguments that should always be appended to
          # the decoded tensor arguments coming from native.
          callbacks: %{callback_id() => {fun(), Nx.t() | tuple(), [term()]}}
        }

  ## Public API

  @doc """
  Starts the callback server and registers it as the EXLA dispatcher process.

  The EXLA NIF is notified of the dispatcher PID so it can route
  `:exla_elixir_call` messages to this process.
  """
  def start_link(_init_arg) do
    GenServer.start_link(__MODULE__, :ok)
  end

  @doc """
  Registers a callback function, its output template, argument template, and options,
  returning a callback id.

  The same `{fun, out_template, arg_template, static_arguments}` quadruple will always return the
  same id for the lifetime of this VM. This id is what the EXLA compiler encodes into
  the host `CustomCall` so the native side can reference the right callback.
  """
  @spec register(pid(), fun(), Nx.t() | tuple(), term(), [term()]) :: callback_id()
  def register(callback_server_pid, fun, out_template, arg_template, static_arguments) when is_function(fun) do
    GenServer.call(callback_server_pid, {:register, fun, out_template, arg_template, static_arguments})
  end

  ## GenServer callbacks

  @impl true
  def init(:ok) do
    # Inform native side that this process is the dispatcher for elixir callbacks
    _ = EXLA.NIF.start_elixir_callback_bridge(self())

    {:ok, %__MODULE__{}}
  end

  @impl true
  def terminate(_reason, _state) do
    try do
      EXLA.NIF.clear_elixir_callback_bridge(self())
    rescue
      _ -> :ok
    end
  end

  @impl true
  def handle_call({:register, fun, out_template, arg_template, opts}, _from, %__MODULE__{} = state) do
    key = {fun, out_template, arg_template, opts}

    case find_existing_id(state.callbacks, key) do
      {:ok, id} ->
        {:reply, id, state}

      :error ->
        id = state.next_id
        state = put_in(state.callbacks[id], {fun, out_template, arg_template, opts})
        state = %{state | next_id: id + 1}
        {:reply, id, state}
    end
  end

  @impl true
  def handle_info({:exla_elixir_call, callback_id, args_spec, reply_tag}, %__MODULE__{} = state) do
    reply_payload =
      try do
        case Map.fetch(state.callbacks, callback_id) do
          {:ok, {fun, out_template, arg_template, opts}} ->
            args_spec
            |> decode_args(arg_template)
            |> run_callback(fun, opts, out_template)
            |> encode_reply()

          :error ->
            Logger.error(
              "EXLA.CallbackServer received callback id #{inspect(callback_id)} that is not registered"
            )

            encode_reply({:error, :unknown_callback})
        end
      rescue
        exception ->
          msg = Exception.message(exception)
          encode_reply({:error, {:runtime_error, "Elixir callback server crashed: " <> msg}})
      catch
        kind, reason ->
          encode_reply({:error, {:runtime_error, "Elixir callback server #{kind}: #{inspect(reason)}"}})
      end

    send_reply(reply_tag, reply_payload)
    {:noreply, state}
  end

  def handle_info(other, state) do
    Logger.debug("EXLA.CallbackServer ignoring unexpected message: #{inspect(other)}")
    {:noreply, state}
  end

  ## Internal helpers

  defp find_existing_id(callbacks, key) do
    Enum.reduce_while(callbacks, :error, fn {id, value}, _acc ->
      if value == key, do: {:halt, {:ok, id}}, else: {:cont, :error}
    end)
  end

  defp run_callback({:error, reason}, _fun, _opts, _out_template), do: {:error, reason}

  defp run_callback({:ok, tensor_args}, fun, opts, out_template) do
    result =
      try do
        fun.(tensor_args, opts)
      rescue
        exception ->
          {:error, {:exception, exception, __STACKTRACE__}}
      catch
        kind, reason ->
          {:error, {kind, reason}}
      end

    case result do
      {:error, _} = error ->
        error

      value ->
        if Nx.compatible?(value, out_template) do
          {:ok, value}
        else
          {:error, {:shape_mismatch, value, out_template}}
        end
    end
  end

  defp decode_args(args_spec, arg_template) when is_list(args_spec) do
    result =
      Enum.reduce_while(args_spec, {:ok, []}, fn
        {bin, {type, shape_list}}, {:ok, acc} ->
          try do
            tensor =
              bin
              |> Nx.from_binary(type)
              |> Nx.reshape(List.to_tuple(shape_list))

            {:cont, {:ok, [tensor | acc]}}
          rescue
            exception ->
              {:halt, {:error, {:decode_failed, exception}}}
          end

        other, _acc ->
          {:halt, {:error, {:invalid_args_spec, other}}}
      end)

    case result do
      {:ok, tensors} ->
        tensors = Enum.reverse(tensors)
        materialize_args(arg_template, tensors)

      {:error, _} = error ->
        error
    end
  end

  defp decode_args(other, _arg_template), do: {:error, {:invalid_args_spec, other}}

  defp encode_reply({:ok, value}), do: {:ok, encode_outputs(value)}

  # Shape mismatch between callback result and output template.
  defp encode_reply({:error, {:shape_mismatch, left, right}}) do
    msg =
      "expected the elixir_call function to match the given output template " <>
        "#{inspect(right)}, got: #{inspect(left)}"

    {:error, {:argument_error, msg}}
  end

  # Callback returned something that isn't a tensor/tuple matching the template.
  defp encode_reply({:error, {:invalid_result, left, right}}) do
    msg =
      "expected the elixir_call function to return a value compatible with the output " <>
        "template #{inspect(right)}, got: #{inspect(left)}"

    {:error, {:argument_error, msg}}
  end

  # Argument decoding failures.
  defp encode_reply({:error, {:decode_failed, exception}}) do
    msg = Exception.message(exception)
    msg = "failed to decode Elixir callback arguments: #{msg}"
    {:error, {:runtime_error, msg}}
  end

  defp encode_reply({:error, {:invalid_args_spec, other}}) do
    msg = "invalid args_spec for Elixir callback: #{inspect(other)}"
    {:error, {:runtime_error, msg}}
  end

  # Unknown callback id from native.
  defp encode_reply({:error, :unknown_callback}) do
    msg = "unknown EXLA elixir_call callback id"
    {:error, {:runtime_error, msg}}
  end

  # User-raised exceptions.
  defp encode_reply({:error, {:exception, exception, _stack}}) do
    msg = Exception.message(exception)
    msg = "Elixir callback raised: #{msg}"
    {:error, {:runtime_error, msg}}
  end

  # Catches other error tuples (throws, exits, etc).
  defp encode_reply({:error, {kind, reason}}) do
    msg = "Elixir callback #{kind}: #{inspect(reason)}"
    {:error, {:runtime_error, msg}}
  end

  defp encode_reply({:error, reason}) do
    msg = "Elixir callback error: #{inspect(reason)}"
    {:error, {:runtime_error, msg}}
  end

  defp materialize_args(arg_template, tensors) do
    {container, remaining} =
      Nx.Defn.Composite.traverse(arg_template, tensors, fn
        %Nx.Tensor{} = _template, [next | rest] ->
          {next, rest}

        other, acc ->
          {other, acc}
      end)

    case remaining do
      [] -> {:ok, container}
      _ -> {:error, {:invalid_args_spec, :extra_values}}
    end
  end

  defp encode_outputs(container) do
    [container]
    |> Nx.Defn.Composite.flatten_list()
    |> Enum.map(&Nx.to_binary/1)
  end

  defp send_reply(reply_tag, {status, result}) do
    try do
      EXLA.NIF.elixir_callback_reply(reply_tag, status, result)
    rescue
      _ ->
        Logger.error(
          "EXLA.CallbackServer failed to send reply to native for tag #{inspect(reply_tag)}"
        )
    end
  end
end
