defmodule Torchx.Macro do
  @moduledoc false

  defmacro __using__(_opts) do
    quote do
      import unquote(__MODULE__)
      Module.register_attribute(Torchx, :torch_function, accumulate: true)
    end
  end

  @doc """
  Function that receives a device and allocates a tensor.
  """
  defmacro defdevice(call) do
    {name, args} = Macro.decompose_call(call)

    unless has_device?(args) do
      raise("At least one argument of defdevice function should be named 'device'.")
    end

    tensors =
      case tensors(args) do
        [] -> :ok
        tensors -> quote do: {unquote(tensors), _} = prepare_tensors!(unquote(tensors))
      end

    quote do
      @torch_function {unquote(name), unquote(length(args))}
      def unquote(name)(unquote_splicing(args)) do
        unquote(tensors)
        {user_device, index} = normalize_device!(var!(device))
        var!(device) = torch_device!(user_device, index)

        case user_device do
          :cpu -> Torchx.NIF.unquote(:"#{name}_cpu")(unquote_splicing(args))
          _ -> Torchx.NIF.unquote(:"#{name}_io")(unquote_splicing(args))
        end
        |> unwrap_tensor!(user_device)
      end
    end
  end

  @doc """
  Generates a call that returns a tensor (or a tuple/list of tensors).

  All tensor variables must start with the name tensor.
  """
  defmacro deftensor(call) do
    defcall(call, :unwrap_tensor!, [Macro.var(:device, __MODULE__)])
  end

  @doc """
  Generates a call that returns a value (not a tensor).

  All tensor variables must start with the name tensor.
  """
  defmacro defvalue(call) do
    defcall(call, :unwrap!, [])
  end

  defp defcall(call, unwrapper, extra) do
    {name, args} = Macro.decompose_call(call)
    tensors = tensors(args)

    if tensors == [] do
      raise ArgumentError, "at least one tensor required in #{name}/#{length(args)}"
    end

    quote do
      @torch_function {unquote(name), unquote(length(args))}
      def unquote(name)(unquote_splicing(args)) do
        {unquote(tensors), device} = prepare_tensors!(unquote(tensors))

        case device do
          :cpu -> Torchx.NIF.unquote(:"#{name}_cpu")(unquote_splicing(args))
          device -> Torchx.NIF.unquote(:"#{name}_io")(unquote_splicing(args))
        end
        |> unquote(unwrapper)(unquote_splicing(extra))
      end
    end
  end

  defp has_device?(args) do
    Enum.any?(args, &match?({:device, _, nil}, &1))
  end

  defp tensors(args) do
    Enum.filter(args, fn {name, _, _} -> match?("tensor" <> _, Atom.to_string(name)) end)
  end
end

defmodule Torchx do
  @valid_devices_md_list """
    * `:cpu`
    * `:cuda`
    * `:mkldnn`
    * `:opengl`
    * `:opencl`
    * `:ideep`
    * `:hip`
    * `:fpga`
    * `:msnpu`
    * `:xla`
    * `:vulkan`
    * `:metal`
    * `:xpu`
    * `:mps`
  """

  @moduledoc """
  Bindings and Nx integration for [PyTorch](https://pytorch.org/).

  Torchx provides an Nx backend through `Torchx.Backend`, which
  allows for integration with both the CPU and GPU functionality
  that PyTorch provides. To enable Torchx as the default backend
  you can add the following line to your desired config environment (`config/config.exs`,
  `config/test.exs`, etc):

      import Config
      config :nx, :default_backend, Torchx.Backend

  This will ensure that by default all tensors are created PyTorch tensors.
  It's important to keep in mind that the default device is the CPU. If you
  wish to allocated tensors to the GPU by default, you can pass the `:device`
  option to the config line, as follows:

      import Config
      config :nx, :default_backend, {Torchx.Backend, device: :cuda}

  The `device_available?/1` function can be used to determine whether
  `:cuda` is available. If you have CUDA installed but it doesn't show
  as available, check out the _Installation_ README section.

  ## Types

  Torchx implements specific names for PyTorch types, which have Nx
  counterparts as in the following table:

    Nx Type    |  Torchx Type    | Description
   ----------- | --------------- | --------------------------------------------------------
   `{:u, 8}`   | `:byte`           | Unsigned 8-bit integer
   `{:s, 8}`   | `:char`           | Signed 8-bit integer
   `{:s, 16}`  | `:short`          | Signed 16-bit integer
   `{:s, 32}`  | `:int`            | Signed 32-bit integer
   `{:s, 64}`  | `:long`           | Signed 64-bit integer
   `{:bf, 16}` | `:brain`          | 16-bit brain floating-point number
   `{:f, 16}`  | `:half`           | 16-bit floating-point number
   `{:f, 32}`  | `:float`          | 32-bit floating-point number
   `{:f, 64}`  | `:double`         | 64-bit floating-point number
   `{:c, 64}`  | `:complex`        | 64-bit complex number, with two 32-bit float components
   `{:c, 128}` | `:complex_double` | 128-bit complex number, with two 64-bit float components

  ## Devices

  PyTorch implements a variety of devices, which can be seen below.

  #{@valid_devices_md_list}
  """
  use Torchx.Macro
  alias Torchx.NIF

  defguard is_tensor(dev, ref) when is_atom(dev) and is_reference(ref)

  @doc """
  Check if device of the given type is available for Torchx.

  You can currently check the availability of:

  * `:cuda`
  * `:mps`
  * `:cpu`

  """
  def device_available?(:cuda), do: NIF.cuda_is_available()
  def device_available?(:mps), do: NIF.mps_is_available()
  def device_available?(:cpu), do: true

  def device_available?(device),
    do: raise(ArgumentError, "Cannot check availability for device #{inspect(device)}.")

  @doc """
  Return devices quantity for the given device type.

  You can check the device count of `:cuda` for now.
  """
  def device_count(:cuda), do: NIF.cuda_device_count()
  def device_count(_), do: raise(ArgumentError, "Only CUDA devices can be counted for now.")

  @doc """
  Returns the default device.

  Here is the priority in the order of availability:

  * `:cuda`
  * `:cpu`

  The default can also be set (albeit not recommended)
  via the application environment by setting the
  `:default_device` option under the `:torchx` application.
  """
  @default_devices [:cuda]
  def default_device do
    case Application.fetch_env(:torchx, :default_device) do
      {:ok, device} ->
        device

      :error ->
        device = Enum.find(@default_devices, {:cpu, -1}, &device_available?/1)
        Application.put_env(:torchx, :default_device, device)
        device
    end
  end

  # LibTorch API bindings

  ## Creation / conversion

  defdevice randint(min, max, shape, type, device)
  defdevice rand(min, max, shape, type, device)
  defdevice normal(mu, sigma, shape, type, device)

  defdevice arange(from, to, step, type, device)
  defdevice arange(from, to, step, type, device, shape)
  defdevice full(shape, scalar, type, device)
  defdevice scalar_tensor(scalar, type, device)
  defdevice ones(shape, type, device)
  def eye(size, type, device), do: eye(size, size, type, device)
  defdevice eye(m, n, type, device)
  defdevice from_blob(blob, shape, type, device)
  defdevice to_device(tensor, device)

  ## Manipulation

  deftensor reshape(tensor, shape)
  deftensor to_type(tensor, type)
  deftensor squeeze(tensor)
  deftensor squeeze(tensor, axis)
  deftensor broadcast_to(tensor, shape)
  deftensor transpose(tensor, dim0, dim1)
  deftensor permute(tensor, dims)
  deftensor split(tensor, split_size)
  deftensor narrow(tensor, dim, start, length)
  deftensor as_strided(tensor, size, strides, offset)
  deftensor concatenate(tensors, axis)
  deftensor gather(tensor_input, tensor_indices, axis)
  deftensor indexed_add(tensor_input, tensor_indices, tensor_updates, axis)
  deftensor indexed_put(tensor_input, tensor_indices, tensor_updates, axis)
  deftensor argsort(tensor, axis, is_descending)
  deftensor flip(tensor, axis)
  deftensor unfold(tensor, dimension, size, step)
  deftensor put(tensor_input, tensor_index, tensor_source)
  deftensor where(tensorA, tensorB, tensorC)

  ## Aggregation

  deftensor sum(tensor, axes, keep_axes)
  deftensor product(tensor)
  deftensor product(tensor, axes, keep_axes)
  deftensor any(tensor)
  deftensor any(tensor, axes, keep_axes)
  deftensor argmax(tensor, axis, keep_axes)
  deftensor argmin(tensor, axis, keep_axes)
  deftensor all(tensor)
  deftensor all(tensor, axes, keep_axes)
  deftensor all_close(tensor_a, tensor_b, rtol, atol, equal_nan)

  deftensor cumulative_sum(tensor, axis)
  deftensor cumulative_product(tensor, axis)
  deftensor cumulative_min(tensor, axis)
  deftensor cumulative_max(tensor, axis)

  ## Binary ops

  deftensor add(tensorA, tensorB)
  deftensor subtract(tensorA, tensorB)
  deftensor multiply(tensorA, tensorB)
  deftensor pow(tensorA, tensorB)
  deftensor remainder(tensorA, tensorB)
  deftensor fmod(tensorA, tensorB)
  deftensor divide(tensorA, tensorB)
  deftensor atan2(tensorA, tensorB)
  deftensor min(tensorA, tensorB)
  deftensor max(tensorA, tensorB)
  deftensor quotient(tensorA, tensorB)

  deftensor left_shift(tensorA, tensorB)
  deftensor right_shift(tensorA, tensorB)

  deftensor equal(tensorA, tensorB)
  deftensor not_equal(tensorA, tensorB)
  deftensor greater(tensorA, tensorB)
  deftensor less(tensorA, tensorB)
  deftensor greater_equal(tensorA, tensorB)
  deftensor less_equal(tensorA, tensorB)

  deftensor logical_and(tensorA, tensorB)
  deftensor logical_or(tensorA, tensorB)
  deftensor logical_xor(tensorA, tensorB)
  deftensor logical_not(tensorA)

  deftensor bitwise_and(tensorA, tensorB)
  deftensor bitwise_or(tensorA, tensorB)
  deftensor bitwise_xor(tensorA, tensorB)

  deftensor amax(tensor, axes, keep_axes)
  deftensor amin(tensor, axes, keep_axes)

  def tensordot(tensorA, tensorB, axesA, axesB),
    do: tensordot(tensorA, tensorB, axesA, [], axesB, [])

  deftensor tensordot(tensorA, tensorB, axesA, batchA, axesB, batchB)
  deftensor matmul(tensorA, tensorB)

  ## Unary ops

  deftensor exp(tensor)
  deftensor expm1(tensor)
  deftensor log(tensor)
  deftensor log1p(tensor)
  deftensor sigmoid(tensor)
  deftensor view_as_real(tensor)
  deftensor conjugate(tensor)
  deftensor cos(tensor)
  deftensor sin(tensor)
  deftensor tan(tensor)
  deftensor cosh(tensor)
  deftensor sinh(tensor)
  deftensor tanh(tensor)
  deftensor acos(tensor)
  deftensor asin(tensor)
  deftensor atan(tensor)
  deftensor acosh(tensor)
  deftensor asinh(tensor)
  deftensor atanh(tensor)
  deftensor sqrt(tensor)
  deftensor rsqrt(tensor)
  deftensor erf(tensor)
  deftensor erfc(tensor)
  deftensor erf_inv(tensor)
  deftensor cbrt(tensor)
  deftensor fft(tensor, length)
  deftensor ifft(tensor, length)

  deftensor abs(tensor)
  deftensor bitwise_not(tensor)
  deftensor ceil(tensor)
  deftensor floor(tensor)
  deftensor negate(tensor)
  deftensor round(tensor)
  deftensor sign(tensor)
  deftensor is_nan(tensor)
  deftensor is_infinity(tensor)

  deftensor pad(tensor, tensor_scalar, config)

  ## LinAlg

  deftensor cholesky(tensor)
  deftensor cholesky(tensor, upper)
  deftensor eigh(tensor)
  deftensor qr(tensor)
  deftensor qr(tensor, reduced)
  deftensor svd(tensor)
  deftensor svd(tensor, full_matrices)
  deftensor lu(tensor)
  deftensor triangular_solve(tensor_a, tensor_b, transpose, upper)
  deftensor determinant(tensor)
  deftensor sort(tensor, axis, descending)
  deftensor top_k(tensor, k)
  deftensor clip(tensor, tensor_min, tensor_max)
  deftensor solve(tensor_a, tensor_b)

  deftensor conv(tensor_input, tensor_kernel, strides, padding, dilation, transposed, groups)
  deftensor max_pool_3d(tensor_input, kernel_size, strides, padding, dilation)

  ## Dirty non-tensor return values

  defvalue to_blob(tensor)
  defvalue to_blob(tensor, limit)
  defvalue delete_tensor(tensor)
  defvalue item(tensor)

  ## Non-dirty non-tensor return values

  def scalar_type({dev, ref}) when is_tensor(dev, ref), do: NIF.scalar_type(ref) |> unwrap!()
  def shape({dev, ref}) when is_tensor(dev, ref), do: NIF.shape(ref) |> unwrap!()
  def nbytes({dev, ref}) when is_tensor(dev, ref), do: NIF.nbytes(ref) |> unwrap!()

  ## Nx

  @doc """
  Gets a Torchx tensor from a Nx tensor.
  """
  def from_nx(tensor) do
    Torchx.Backend.from_nx(tensor)
  end

  @doc """
  Converts a Torchx tensor to a Nx tensor.
  """
  def to_nx(torchx) do
    type = torchx |> scalar_type() |> Torchx.Backend.from_torch_type()
    tensor = Nx.template(shape(torchx), type)
    Torchx.Backend.to_nx(torchx, tensor)
  end

  @doc false
  def __torch__, do: @torch_function

  ## Macro callbacks

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
    xpu: 12,
    mps: 13
  }

  defp normalize_device!({device, index}) when is_atom(device) and is_integer(index),
    do: {device, index}

  defp normalize_device!(device) when is_atom(device),
    do: {device, -1}

  defp normalize_device!(device),
    do: raise(ArgumentError, "expected device to be {atom, index} or atom, got: #{device}")

  defp torch_device!(device, index) do
    id = @devices[device] || raise ArgumentError, "unknown device #{inspect(device)}"
    {id, index}
  end

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, result}), do: result
  defp unwrap!({:error, error}), do: raise("Torchx: " <> List.to_string(error))

  defp unwrap_tensor!(tagged_result, device) do
    case unwrap!(tagged_result) do
      ref when is_reference(ref) ->
        {device, ref}

      list when is_list(list) ->
        Enum.map(list, &{device, &1})

      tuple when is_tuple(tuple) ->
        tuple |> Tuple.to_list() |> Enum.map(&{device, &1}) |> List.to_tuple()
    end
  end

  defp prepare_tensors_list!(tensors_list, dev) do
    tensors =
      Enum.map(tensors_list, fn
        {^dev, ref} when is_tensor(dev, ref) ->
          ref

        {other_dev, ref} when is_tensor(other_dev, ref) ->
          raise ArgumentError, "cannot perform operation across devices #{dev} and #{other_dev}"

        bad_tensor ->
          raise ArgumentError, "expected a Torchx tensor, got: #{inspect(bad_tensor)}"
      end)

    {tensors, dev}
  end

  defp prepare_tensors!(tensors) do
    Enum.map_reduce(tensors, nil, fn
      {dev, ref}, nil when is_tensor(dev, ref) ->
        {ref, dev}

      {dev, ref}, dev when is_tensor(dev, ref) ->
        {ref, dev}

      {dev, ref}, other_dev when is_tensor(dev, ref) ->
        raise ArgumentError, "cannot perform operation across devices #{dev} and #{other_dev}"

      [{dev, ref} | _] = tensors, nil when is_tensor(dev, ref) ->
        prepare_tensors_list!(tensors, dev)

      tensors, dev when is_list(tensors) ->
        prepare_tensors_list!(tensors, dev)

      bad_tensor, _dev ->
        raise ArgumentError, "expected a Torchx tensor, got: #{inspect(bad_tensor)}"
    end)
  end
end
