defmodule EXLA.Defn.Buffer do
  @moduledoc false

  @doc """
  EXLA.Buffer -> Nx.
  """
  def to_nx!(buffers, outputs, fun \\ & &1) do
    {res, []} =
      Nx.Defn.Composite.traverse(outputs, buffers, fn %Nx.Tensor{} = hole, [buffer | acc] ->
          {fun.(%{hole | data: buffer_to_data(hole, buffer)}), acc}
      end)

    res
  end

  defp buffer_to_data(_hole, buffer) when is_binary(buffer) do
    %Nx.BinaryBackend{state: buffer}
  end

  defp buffer_to_data(%{type: type, shape: shape}, %EXLA.Buffer{shape: exla_shape} = buffer) do
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

    buffer_to_data(buffer)
  end

  defp buffer_to_data(%EXLA.Buffer{ref: ref, data: nil}), do: %EXLA.DeviceBackend{state: ref}
  defp buffer_to_data(%EXLA.Buffer{ref: nil, data: data}), do: %Nx.BinaryBackend{state: data}

  defp to_nx_type({:pred, 8}), do: {:u, 8}
  defp to_nx_type(type), do: type

  defp to_exla_shape(%Nx.Tensor{type: type, shape: shape}), do: EXLA.Shape.make_shape(type, shape)

  @doc """
  Nx -> EXLA.Buffer.
  """
  def from_nx!(tensors) do
    for tensor <- tensors do
      %Nx.Tensor{data: data} = tensor

      case data do
        %EXLA.DeviceBackend{state: ref} -> EXLA.Buffer.from_ref(ref, to_exla_shape(tensor))
        _ -> EXLA.Buffer.from_binary(Nx.to_binary(tensor), to_exla_shape(tensor))
      end
    end
  end
end
