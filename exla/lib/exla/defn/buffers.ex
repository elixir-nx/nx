defmodule EXLA.Defn.Buffers do
  @moduledoc false

  @doc """
  Filter inputs based on index.
  """
  def filter_by_indexes(args, inputs, callback \\ fn x, _ -> x end)

  def filter_by_indexes(args, inputs, callback) when is_list(inputs),
    do: filter_by_indexes_list(args, 0, inputs, callback)

  def filter_by_indexes(args, inputs, callback) when is_map(inputs),
    do: filter_by_indexes_map(args, 0, inputs, callback)

  defp filter_by_indexes_list([var | vars], i, [i | inputs], callback),
    do: [callback.(var, i) | filter_by_indexes_list(vars, i + 1, inputs, callback)]

  defp filter_by_indexes_list([_var | vars], i, inputs, callback),
    do: filter_by_indexes_list(vars, i + 1, inputs, callback)

  defp filter_by_indexes_list([], _i, [], _callback),
    do: []

  defp filter_by_indexes_map([var | vars], i, inputs, callback) when is_map_key(inputs, i),
    do: [callback.(var, i) | filter_by_indexes_map(vars, i + 1, inputs, callback)]

  defp filter_by_indexes_map([_var | vars], i, inputs, callback),
    do: filter_by_indexes_map(vars, i + 1, inputs, callback)

  defp filter_by_indexes_map([], _i, _, _callback),
    do: []

  @doc """
  Splits the given args by value and returns them as is.

  Entries with a map entry are discarded.
  """
  def split_by_value(args, %{} = map, callback) do
    {_i, left, right} =
      Enum.reduce(args, {0, [], []}, fn arg, {i, left, right} ->
        case map do
          %{^i => nil} -> {i + 1, [callback.(arg, i, nil) | left], right}
          %{^i => value} -> {i + 1, left, [callback.(arg, i, value) | right]}
          %{} -> {i + 1, left, right}
        end
      end)

    {left, right}
  end

  @doc """
  binary + EXLA.DeviceBuffer + EXLA.BinaryBuffer -> Nx.
  """
  def to_nx!(buffers, outputs) do
    {res, []} =
      Nx.Defn.Composite.traverse(outputs, buffers, fn
        %Nx.Tensor{} = hole, [buffer | acc] ->
          {%{hole | data: buffer_to_data(hole, buffer)}, acc}
      end)

    res
  end

  defp buffer_to_data(hole, buffer) when is_binary(buffer) do
    if Nx.byte_size(hole) != byte_size(buffer) do
      raise "internal bug! Nx.Defn expected a tensor with byte size #{inspect(Nx.byte_size(hole))} " <>
              "but got #{inspect(byte_size(buffer))}"
    end

    %Nx.BinaryBackend{state: buffer}
  end

  defp buffer_to_data(tensor, %EXLA.DeviceBuffer{typespec: typespec} = buffer) do
    validate_shape!(tensor, typespec)
    %EXLA.Backend{buffer: buffer}
  end

  defp buffer_to_data(tensor, %EXLA.BinaryBuffer{data: data, typespec: typespec}) do
    validate_shape!(tensor, typespec)
    %Nx.BinaryBackend{state: data}
  end

  defp validate_shape!(%Nx.Tensor{} = t, typespec) do
    %{type: type, shape: shape} = Nx.devectorize(t)

    nx_type = to_nx_type(typespec.type)
    nx_shape = typespec.shape

    if type != nx_type do
      raise "internal bug! Nx.Defn expected a tensor with type #{inspect(type)} " <>
              "but got #{inspect(nx_type)}"
    end

    if shape != nx_shape do
      raise "internal bug! Nx.Defn expected a tensor with shape #{inspect(shape)} " <>
              "but got #{inspect(nx_shape)}"
    end
  end

  defp to_nx_type({:pred, 8}), do: {:u, 8}
  defp to_nx_type(type), do: type

  @doc """
  Nx -> EXLA.DeviceBuffer + EXLA.BinaryBuffer.
  """
  def from_nx!(fun, executable, transfer? \\ true) do
    %Nx.Tensor{data: data} = tensor = Nx.devectorize(fun.())

    case data do
      %EXLA.Backend{buffer: %EXLA.DeviceBuffer{ref: ref} = buffer}
      when node(ref) != node() ->
        binary =
          try do
            :erpc.call(node(ref), EXLA.DeviceBuffer, :read, [buffer])
          catch
            :error, {:exception, reason, stacktrace} ->
              reraise Exception.normalize(:error, reason, stacktrace), stacktrace
          end

        EXLA.BinaryBuffer.from_binary(binary, to_typespec(tensor))

      %EXLA.Backend{buffer: %EXLA.DeviceBuffer{} = buffer}
      when transfer? and buffer.client_name != executable.client.name
      when transfer? and buffer.device_id != executable.device_id ->
        buffer_client = EXLA.Client.fetch!(buffer.client_name)

        if buffer_client.automatic_transfers do
          EXLA.DeviceBuffer.copy_to_device(buffer, executable.client, executable.device_id)
        else
          default = EXLA.Client.fetch!(EXLA.Client.default_name())

          raise ArgumentError, """
          EXLA computation (defn) is allocated on client #{executable.client.name} \
          ##{executable.device_id} (#{executable.client.platform}) \
          but one of the input tensors are allocated on #{buffer_client.name} \
          ##{buffer.device_id} (#{buffer_client.platform}).

          EXLA by default only transfers tensors allocated on host to other clients. \
          You can force `:host` as your default backend with:

              # via config
              config :nx, default_backend: {EXLA.Backend, client: :host}

              # via API
              Nx.global_default_backend({EXLA.Backend, client: :host})

          Otherwise ensure your tensors are allocated on the same client-device \
          pair as your numerical definitions (defn). The default client-device is \
          #{default.name} ##{default.default_device_id} (#{default.platform})
          """
        end

      %EXLA.Backend{buffer: buffer} ->
        buffer

      %Nx.Defn.Expr{} ->
        raise ArgumentError,
              "cannot pass a tensor expression as argument to defn, got: #{inspect(tensor)}"

      _ ->
        EXLA.BinaryBuffer.from_binary(Nx.to_binary(tensor), to_typespec(tensor))
    end
  end

  defp to_typespec(%Nx.Tensor{type: type, shape: shape}), do: EXLA.Typespec.tensor(type, shape)
end
