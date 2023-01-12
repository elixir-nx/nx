defmodule EXLA.Defn.Buffers do
  @moduledoc false

  @doc """
  Filter inputs based on index.
  """
  def filter_by_indexes(args, inputs), do: filter_by_indexes(args, 0, inputs)

  defp filter_by_indexes([var | vars], i, [i | inputs]),
    do: [var | filter_by_indexes(vars, i + 1, inputs)]

  defp filter_by_indexes([_var | vars], i, inputs),
    do: filter_by_indexes(vars, i + 1, inputs)

  defp filter_by_indexes([], _i, []),
    do: []

  @doc """
  binary + EXLA.DeviceBuffer + EXLA.BinaryBuffer -> Nx.
  """
  def to_nx!(buffers, outputs) do
    {res, []} =
      Nx.Defn.Composite.traverse(outputs, buffers, fn %Nx.Tensor{} = hole, [buffer | acc] ->
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

  defp validate_shape!(%{type: type, shape: shape}, exla_shape) do
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
  def from_nx!(funs, _executable) do
    for fun <- funs do
      %Nx.Tensor{data: data} = tensor = fun.()

      case data do
        # TODO: Make this customizable based on lazy transfers
        # %EXLA.Backend{buffer: %EXLA.DeviceBuffer{} = buffer}
        # when buffer.client_name != executable.client.name ->
        #   buffer_client = EXLA.Client.fetch!(buffer.client_name)

        #   if buffer_client.platform == :host do
        #     EXLA.DeviceBuffer.copy_to_device(buffer, executable.client, executable.device_id)
        #   else
        #     raise ArgumentError, """
        #     EXLA computation (defn) is allocated on client #{executable.client.name} (#{executable.client.platform})
        #     but one of the input tensors are allocated on #{buffer_client.name} (#{buffer_client.platform}).

        #     EXLA only automatically transfers allocated on host to other client.
        #     You need to either transfer your tensors to the same client as the executable
        #     or compile the defn with a client that matches your input tensors
        #     """
        #   end

        %EXLA.Backend{buffer: buffer} ->
          buffer

        %Nx.Defn.Expr{} ->
          raise ArgumentError,
                "cannot pass a tensor expression as argument to defn, got: #{inspect(tensor)}"

        _ ->
          EXLA.BinaryBuffer.from_binary(Nx.to_binary(tensor), to_exla_shape(tensor))
      end
    end
  end

  defp to_exla_shape(%Nx.Tensor{type: type, shape: shape}), do: EXLA.Shape.make_shape(type, shape)
end
