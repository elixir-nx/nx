defmodule Exla.Tensor do
  alias __MODULE__, as: Tensor
  alias Exla.Client
  alias Exla.Shape
  # TODO: Different namespace?
  @enforce_keys [:data, :shape, :device]
  defstruct [:data, :shape, :device]

  @doc """
  Creates a scalar tensor with value `value` and type `dtype`. Optionally places the tensor on a `device`.
  """
  # After thinking about it more, `device` should be an accurate representation of where the Tensor is...
  # so we have a device specification that is `{platform, ordinal}` where platform can be :cpu, :gpu, :tpu,
  # :beam...for physical devices ordinal is a physical device. For :beam, the ordinal is the Tensor's owning
  # process.
  def scalar(value, dtype, device \\ {:beam, 0}) when is_number(value) and is_atom(dtype) do
    shape = Shape.make_shape(dtype, {})
    %Tensor{data: {:binary, number_to_bin_segment(value, dtype)}, shape: shape, device: device}
  end

  def vector(length, value, dtype, device \\ {:beam, 0}) when is_number(value) and is_atom(dtype) do
    shape = Shape.make_shape(dtype, {length})
    %Tensor{data: {:binary, :binary.copy(number_to_bin_segment(value, dtype), length)}, shape: shape, device: device}
  end

  defp number_to_bin_segment(value, :int32), do: <<value::32-integer-little>>
  defp number_to_bin_segment(value, :float32), do: <<value::32-float-little>>
  defp number_to_bin_segment(value, :int64), do: <<value::64-integer-little>>
  defp number_to_bin_segment(value, :float64), do: <<value::64-float-little>>

  # TODO: This doesn't actually match devices, it just places on host but the idea is the same
  # TODO: We can build on this by matching a reference, accepting a device arg, and switching to
  # that device.
  # TODO: Check if the client actually supports the specified device.
  # This function can be called ahead of time by the user. Inside `run`, we just need to check if
  # the data is on device or not AND we need to ensure the device fields match.
  def to_device(
        %Client{ref: client},
        %Tensor{data: {:binary, data}, shape: shape},
        device \\ {:cpu, 0}
      ) do
    {:ok, ref} = Exla.NIF.binary_to_shaped_buffer(client, data, shape.ref, 0)
    %Tensor{data: {:ref, ref}, shape: shape, device: device}
  end
end
