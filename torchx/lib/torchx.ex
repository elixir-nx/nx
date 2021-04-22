defmodule Torchx.Macro do
  defmacro __using__(_opts) do
    quote do
      import unquote(__MODULE__)
      Module.register_attribute(Torchx, :torch_function, accumulate: true)
    end
  end

  defmacro defdevice(call) do
    {name, args} = Macro.decompose_call(call)

    args_names = args_names(args)

    if(not Enum.any?(args, &match?({:device, _, nil}, &1)),
      do: raise("At least one argument of defdevice function should be named 'device'.")
    )

    quote do
      @torch_function {unquote(name), unquote(args_names)}
      def unquote(name)(unquote_splicing(args)) do
        Torchx.NIF.call(unquote(name), var!(device), [unquote_splicing(args)])
        |> wrap(var!(device))
      end
    end
  end

  defmacro deftorch(call) do
    {name, args} = Macro.decompose_call(call)

    args_names = args_names(args)

    quote do
      @torch_function {unquote(name), unquote(args_names)}
      def unquote(name)(unquote_splicing(args)) do
        {device, prepared_args} = prepare_args(unquote(args))

        Torchx.NIF.call(unquote(name), device, prepared_args)
        |> wrap(device)
      end
    end
  end

  defp args_names(args), do: Enum.map(args, &elem(&1, 0))

  defguardp is_tensor(t)
            when is_tuple(t) and is_atom(elem(elem(t, 0), 0)) and
                   is_integer(elem(elem(t, 0), 1)) and is_reference(elem(t, 1))

  # TODO: select tensor args by variable names
  # (args starting with 'tensor' should be considered {device, ref} tensor tuples)
  def prepare_args(args) do
    {prepared_args, device} =
      Enum.map_reduce(args, nil, fn
        {dev, ref} = t, nil when is_tensor(t) -> {ref, dev}
        {dev, ref} = t, dev when is_tensor(t) -> {ref, dev}
        {_dev, _ref} = t, _other_dev when is_tensor(t) -> raise "cannot perform across devices"
        var, dev -> {var, dev}
      end)

    {device, prepared_args}
  end
end

defmodule Torchx do
  alias Torchx.NIF

  use Torchx.Macro

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

  defdevice randint(min, max, shape, type, device)
  defdevice rand(min, max, shape, type, device)
  defdevice normal(mu, sigma, shape, type, device)

  defdevice arange(from, to, step, type, device)
  defdevice arange(from, to, step, type, device, shape)
  defdevice full(shape, scalar, type, device)
  defdevice scalar_tensor(scalar, type, device)
  defdevice ones(shape, type, device)
  defdevice eye(size, type, device)

  ## Manipulation

  deftorch reshape(tensor, shape)
  deftorch to_type(tensor, type)
  deftorch to_device(tensor, device)
  defdevice from_blob(blob, shape, type, device)
  deftorch to_blob(tensor)
  deftorch to_blob(tensor, limit)

  deftorch delete_tensor(tensor)
  deftorch squeeze(tensor)
  deftorch squeeze(tensor, axis)
  deftorch broadcast_to(tensor, shape)
  deftorch transpose(tensor, dim0, dim1)
  deftorch permute(tensor, dims)
  deftorch split(tensor, split_size)
  deftorch narrow(tensor, dim, start, length)
  deftorch as_strided(tensor, size, strides, offset)

  ## Aggregation

  deftorch sum(tensor, axes, keep_axes)
  deftorch argmax(tensor, axis, keep_axes)
  deftorch argmin(tensor, axis, keep_axes)

  ## Operations

  ### Binary

  deftorch add(tensorA, tensorB)

  deftorch subtract(tensorA, tensorB)
  deftorch multiply(tensorA, tensorB)
  deftorch power(tensorA, tensorB)
  deftorch remainder(tensorA, tensorB)
  deftorch divide(tensorA, tensorB)
  deftorch atan2(tensorA, tensorB)
  deftorch min(tensorA, tensorB)
  deftorch max(tensorA, tensorB)
  deftorch quotient(tensorA, tensorB)

  deftorch left_shift(tensorA, tensorB)
  deftorch right_shift(tensorA, tensorB)

  deftorch equal(tensorA, tensorB)
  deftorch not_equal(tensorA, tensorB)
  deftorch greater(tensorA, tensorB)
  deftorch less(tensorA, tensorB)
  deftorch greater_equal(tensorA, tensorB)
  deftorch less_equal(tensorA, tensorB)

  deftorch logical_and(tensorA, tensorB)
  deftorch logical_or(tensorA, tensorB)
  deftorch logical_xor(tensorA, tensorB)

  deftorch bitwise_and(tensorA, tensorB)
  deftorch bitwise_or(tensorA, tensorB)
  deftorch bitwise_xor(tensorA, tensorB)

  deftorch outer(tensorA, tensorB)

  ### Unary

  deftorch exp(tensor)
  deftorch expm1(tensor)
  deftorch log(tensor)
  deftorch log1p(tensor)
  deftorch logistic(tensor)
  deftorch cos(tensor)
  deftorch sin(tensor)
  deftorch tan(tensor)
  deftorch cosh(tensor)
  deftorch sinh(tensor)
  deftorch tanh(tensor)
  deftorch acos(tensor)
  deftorch asin(tensor)
  deftorch atan(tensor)
  deftorch acosh(tensor)
  deftorch asinh(tensor)
  deftorch atanh(tensor)
  deftorch sqrt(tensor)
  deftorch rsqrt(tensor)
  deftorch cbrt(tensor)
  deftorch erf(tensor)
  deftorch erfc(tensor)
  deftorch erf_inv(tensor)

  deftorch abs(tensor)
  deftorch bitwise_not(tensor)
  deftorch ceil(tensor)
  deftorch floor(tensor)
  deftorch negate(tensor)
  deftorch round(tensor)
  deftorch sign(tensor)

  ## Transformations

  deftorch cholesky(tensor)
  deftorch cholesky(tensor, upper)
  deftorch qr(tensor)
  deftorch qr(tensor, reduced)

  def tensordot(left, right, left_axes, right_axes) do
    {device, [left_ref, right_ref]} = to_refs([left, right])

    NIF.call(:tensordot, device, [left_ref, right_ref, left_axes, right_axes])
    |> wrap(device)
  end

  IO.inspect(@torch_function)

  def __torch__, do: @torch_function

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
