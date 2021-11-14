defmodule EXLA.Defn.Buffer do
  @moduledoc false

  @doc """
  EXLA.Buffer -> Nx.
  """
  def to_nx!(buffers, outputs) do
    {result, []} = each_to_nx(outputs, buffers)
    result
  end

  defp each_to_nx({:tuple, outputs}, acc) when is_list(outputs) do
    {exprs, acc} = Enum.map_reduce(outputs, acc, &each_to_nx/2)
    {List.to_tuple(exprs), acc}
  end

  defp each_to_nx({:struct, outputs, mod}, acc) when is_list(outputs) do
    {exprs, acc} =
      Enum.map_reduce(outputs, acc, fn {k, v}, acc ->
        {v, acc} = each_to_nx(v, acc)
        {{k, v}, acc}
      end)

    {struct(mod, exprs), acc}
  end

  defp each_to_nx({:map, outputs}, acc) when is_list(outputs) do
    {exprs, acc} =
      Enum.map_reduce(outputs, acc, fn {k, v}, acc ->
        {v, acc} = each_to_nx(v, acc)
        {{k, v}, acc}
      end)

    {Map.new(exprs), acc}
  end

  defp each_to_nx(hole, [%EXLA.Buffer{shape: shape} = buffer | acc]) do
    nx_type = to_nx_type(shape.dtype)
    nx_shape = shape.dims

    if hole.type != nx_type do
      raise "internal bug! Nx.Defn expected a tensor with type #{inspect(hole.type)} " <>
              "but got #{inspect(nx_type)}"
    end

    if hole.shape != nx_shape do
      raise "internal bug! Nx.Defn expected a tensor with shape #{inspect(hole.shape)} " <>
              "but got #{inspect(nx_shape)}"
    end

    {%{hole | data: buffer_to_data(buffer)}, acc}
  end

  defp buffer_to_data(%EXLA.Buffer{ref: ref, data: nil}),
    do: %EXLA.DeviceBackend{state: ref}

  defp buffer_to_data(%EXLA.Buffer{ref: nil, data: data}),
    do: %Nx.BinaryBackend{state: data}

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
        %EXLA.DeviceBackend{state: ref} -> EXLA.Buffer.buffer(ref, to_exla_shape(tensor))
        _ -> EXLA.Buffer.buffer(Nx.to_binary(tensor), to_exla_shape(tensor))
      end
    end
  end
end
