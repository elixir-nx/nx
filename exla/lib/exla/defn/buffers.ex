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

  defp buffer_to_data(tensor, %EXLA.DeviceBuffer{shape: exla_shape} = buffer) do
    validate_shape!(tensor, exla_shape)
    %EXLA.Backend{buffer: buffer}
  end

  defp buffer_to_data(tensor, %EXLA.BinaryBuffer{data: data, shape: exla_shape}) do
    validate_shape!(tensor, exla_shape)
    %Nx.BinaryBackend{state: data}
  end

  defp validate_shape!(%Nx.Tensor{} = t, exla_shape) do
    %{type: type, shape: shape} = Nx.devectorize(t)

    nx_type = to_nx_type(exla_shape.dtype)
    nx_shape = exla_shape.dims

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
      %EXLA.Backend{buffer: %EXLA.DeviceBuffer{} = buffer}
      when transfer? and buffer.client_name != executable.client.name ->
        buffer_client = EXLA.Client.fetch!(buffer.client_name)

        if buffer_client.platform == :host do
          EXLA.DeviceBuffer.copy_to_device(buffer, executable.client, executable.device_id)
        else
          raise ArgumentError, """
          EXLA computation (defn) is allocated on client #{executable.client.name} (#{executable.client.platform})
          but one of the input tensors are allocated on #{buffer_client.name} (#{buffer_client.platform}).

          EXLA only automatically transfers allocated on host to other client.
          You need to either transfer your tensors to the same client as the executable
          or compile the defn with a client that matches your input tensors
          """
        end

      %EXLA.Backend{buffer: buffer} ->
        buffer

      %Nx.Defn.Expr{} ->
        raise ArgumentError,
              "cannot pass a tensor expression as argument to defn, got: #{inspect(tensor)}"

      _ ->
        EXLA.BinaryBuffer.from_binary(Nx.to_binary(tensor), to_exla_shape(tensor))
    end
  end

  defp to_exla_shape(%Nx.Tensor{type: type, shape: shape}), do: EXLA.Shape.make_shape(type, shape)
end
