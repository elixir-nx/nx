defmodule Torchx.Macro do
  defmacro deftorch(call) do
    {name, args} = Macro.decompose_call(call)

    if Enum.any?(args, &match?({:device, _, nil}, &1)) do
      quote do
        def unquote(name)(unquote_splicing(args)) do
          Torchx.NIF.call(unquote(name), var!(device), [unquote_splicing(args)])
          |> wrap(var!(device))
        end
      end
    else
      quote do
        def unquote(name)(unquote_splicing(args)) do
          {device, prepared_args} = prepare_args(unquote(args))

          Torchx.NIF.call(unquote(name), device, prepared_args)
          |> wrap(device)
        end
      end
    end
  end

  defguardp is_tensor(t)
            when is_tuple(t) and is_integer(elem(elem(t, 0), 0)) and
                   is_integer(elem(elem(t, 0), 1)) and is_reference(elem(t, 1))

  # TODO: select tensor args by variable names
  # (args starting with 'tensor' should be considered {device, ref} tensor tuples)
  def prepare_args(args) do
    {prepared_args, device} =
      Enum.map_reduce(args, nil, fn
        {dev, ref} = t, nil when is_tensor(t) -> {ref, dev}
        {dev, ref} = t, dev when is_tensor(t) -> {ref, dev}
        {dev, ref} = t, other_dev when is_tensor(t) -> raise "cannot perform across devices"
        var, dev -> {var, dev}
      end)

    {device, prepared_args} |> IO.inspect()
  end
end

defmodule Torchx do
  alias Torchx.NIF

  import Torchx.Macro

  @doc """
  Check if device of the given type is available for Torchx.
  Device atom can be any of:

    * :cpu
    * :cuda
    * :mkldnn
    * :opengl
    * :opencl
    * :ideep
    * :hip
    * :fpga
    * :msnpu
    * :xla
    * :vulkan
    * :metal
    * :xpu

  But only :cuda availability check is supported for now.
  """
  def device_available?(:cuda), do: NIF.cuda_is_available()
  def device_available?(:cpu), do: true
  def device_available?(_), do: raise("Only CUDA device availability check is supported for now.")

  @doc """
  Return devices quantity for the given device type. Only :cuda is supported for now.
  """
  def device_count(:cuda), do: NIF.cuda_device_count()
  def device_count(_), do: raise("Only CUDA devices can be counted for now.")

  # LibTorch API bindings

  ## Creation

  deftorch arange(from, to, step, type, device)
  deftorch full(shape, scalar, type, device)
  deftorch scalar_tensor(scalar, type, device)
  deftorch ones(shape, type, device)
  deftorch eye(size, type, device)

  ## Operations

  deftorch add(tensorA, tensorB)
  deftorch subtract(tensorA, tensorB)

  def tensordot(left, right, left_axes, right_axes) do
    {device, [left_ref, right_ref]} = to_refs([left, right])

    NIF.call(:tensordot, device, [left_ref, right_ref, left_axes, right_axes])
    |> wrap(device)
  end

  ## Utils

  @doc false
  def type_of({_device, ref}), do: type_of(ref)
  def type_of(ref), do: NIF.scalar_type(ref) |> unwrap!()

  @doc false
  def shape_of({_device, ref}), do: shape_of(ref)
  def shape_of(ref), do: NIF.shape(ref) |> unwrap!()

  @doc false
  def device_of({_device, ref}), do: device_of(ref)

  def device_of(ref),
    do: NIF.device_of(ref) |> unwrap!() |> List.to_string() |> parse_torch_device_str()

  defp parse_torch_device_str(str) when is_binary(str) do
    str
    |> String.split(":")
    |> case do
      [type, index] ->
        {String.to_existing_atom(type), String.to_integer(index)}

      [type] ->
        String.to_existing_atom(type)
    end
  end

  @devices %{
    cpu: 0,
    cuda: 1,
    mkldnn: 2,
    opengl: 3,
    opencl: 4,
    ideep: 5,
    hip: 6,
    fpga: 7,
    msnpu: 8,
    xla: 9,
    vulkan: 10,
    metal: 11,
    xpu: 12
  }

  @doc false
  def torch_device({device, index}) when is_atom(device) and is_integer(index),
    do: {@devices[device], index}

  def torch_device(device) when is_atom(device), do: {@devices[device], -1}

  defp unwrap!({:ok, result}), do: result
  defp unwrap!({:error, error}), do: raise("Torchx: " <> List.to_string(error))

  defp to_refs([{device, ref} | t]) do
    refs =
      Enum.map(t, fn
        {^device, ref} ->
          ref

        {other_device, _ref} ->
          raise "cannot perform operations on across devices: #{inspect(device)} and #{
                  inspect(other_device)
                }"
      end)

    {device, [ref | refs]}
  end

  defp wrap(maybe_ref, device), do: {device, unwrap!(maybe_ref)}
end
