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
   `{:f, 8}`   | `:float8_e5m2`    | 8-bit floating-point number (E5M2)
   `{:f8_e4m3fn, 8}` | `:float8_e4m3fn` | 8-bit floating-point number (E4M3FN)
   `{:f, 16}`  | `:half`           | 16-bit floating-point number
   `{:f, 32}`  | `:float`          | 32-bit floating-point number
   `{:f, 64}`  | `:double`         | 64-bit floating-point number
   `{:c, 64}`  | `:complex`        | 64-bit complex number, with two 32-bit float components
   `{:c, 128}` | `:complex_double` | 128-bit complex number, with two 64-bit float components

  ## Devices

  PyTorch implements a variety of devices, which can be seen below.

  #{@valid_devices_md_list}

  ## Backend documentation

  Backend-specific behaviour, options, and limitations for Nx linear algebra are
  documented in [Nx.LinAlg](backend_documentation-nx_lin_alg.html). Top-level
  `Nx` blocks and transfers are documented in
  [Nx](backend_documentation-nx.html). See the
  [backend documentation convention](https://hexdocs.pm/nx/backend_documentation-convention.html)
  in the Nx package.
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
  defdevice from_pointer(mode, handle, shape, type, device)

  @torch_function {:to_device, 2}
  def to_device(tensor, device) do
    {[tensor_ref], _current_device} = prepare_tensors!([tensor])
    {user_device, index} = normalize_device!(device)
    target_device_struct = torch_device!(user_device, index)

    case user_device do
      :cpu -> Torchx.NIF.to_device_cpu(tensor_ref, target_device_struct)
      _ -> Torchx.NIF.to_device_io(tensor_ref, target_device_struct)
    end
    |> unwrap_tensor!(user_device)
  end

  ## Manipulation

  # The ops below take a trailing `donate` flag. When it is true and no other
  # tensor shares the buffer of the input, the op reuses that buffer and the
  # input can no longer be used. Otherwise it allocates as usual.

  def reshape(tensor, shape), do: reshape(tensor, shape, false)
  deftensor reshape(tensor, shape, donate)
  def to_type(tensor, type), do: to_type(tensor, type, false)
  deftensor to_type(tensor, type, donate)
  deftensor squeeze(tensor)
  def squeeze(tensor, axis), do: squeeze(tensor, axis, false)
  deftensor squeeze(tensor, axis, donate)
  def broadcast_to(tensor, shape), do: broadcast_to(tensor, shape, nil, false)
  def broadcast_to(tensor, shape, clone), do: broadcast_to(tensor, shape, clone, false)

  deftensor broadcast_to(tensor, shape, clone, donate)
  def transpose(tensor, dim0, dim1), do: transpose(tensor, dim0, dim1, false)
  deftensor transpose(tensor, dim0, dim1, donate)
  def permute(tensor, dims), do: permute(tensor, dims, false)
  deftensor permute(tensor, dims, donate)
  deftensor split(tensor, split_size)
  def slice(tensor, starts, lengths, strides), do: slice(tensor, starts, lengths, strides, false)
  deftensor slice(tensor, starts, lengths, strides, donate)
  deftensor concatenate(tensors, axis)
  deftensor gather(tensor_input, tensor_indices, axis)
  deftensor index(tensor_input, tensor_indices)

  def index_put(tensor_input, tensor_indices, tensor_updates, accumulate),
    do: index_put(tensor_input, tensor_indices, tensor_updates, accumulate, false)

  deftensor index_put(tensor_input, tensor_indices, tensor_updates, accumulate, donate)
  deftensor argsort(tensor, axis, is_descending, stable)
  deftensor flip(tensor, axis)
  def unfold(tensor, dimension, size, step), do: unfold(tensor, dimension, size, step, false)
  deftensor unfold(tensor, dimension, size, step, donate)
  def put(tensor_input, index, tensor_source), do: put(tensor_input, index, tensor_source, false)
  deftensor put(tensor_input, index, tensor_source, donate)
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

  def add(tensorA, tensorB), do: add(tensorA, tensorB, false)
  deftensor add(tensorA, tensorB, donate)
  def subtract(tensorA, tensorB), do: subtract(tensorA, tensorB, false)
  deftensor subtract(tensorA, tensorB, donate)
  def multiply(tensorA, tensorB), do: multiply(tensorA, tensorB, false)
  deftensor multiply(tensorA, tensorB, donate)
  def pow(tensorA, tensorB), do: pow(tensorA, tensorB, false)
  deftensor pow(tensorA, tensorB, donate)
  def remainder(tensorA, tensorB), do: remainder(tensorA, tensorB, false)
  deftensor remainder(tensorA, tensorB, donate)
  def fmod(tensorA, tensorB), do: fmod(tensorA, tensorB, false)
  deftensor fmod(tensorA, tensorB, donate)
  def divide(tensorA, tensorB), do: divide(tensorA, tensorB, false)
  deftensor divide(tensorA, tensorB, donate)
  def atan2(tensorA, tensorB), do: atan2(tensorA, tensorB, false)
  deftensor atan2(tensorA, tensorB, donate)
  def min(tensorA, tensorB), do: min(tensorA, tensorB, false)
  deftensor min(tensorA, tensorB, donate)
  def max(tensorA, tensorB), do: max(tensorA, tensorB, false)
  deftensor max(tensorA, tensorB, donate)
  def quotient(tensorA, tensorB), do: quotient(tensorA, tensorB, false)
  deftensor quotient(tensorA, tensorB, donate)

  def left_shift(tensorA, tensorB), do: left_shift(tensorA, tensorB, false)
  deftensor left_shift(tensorA, tensorB, donate)
  def right_shift(tensorA, tensorB), do: right_shift(tensorA, tensorB, false)
  deftensor right_shift(tensorA, tensorB, donate)

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

  def bitwise_and(tensorA, tensorB), do: bitwise_and(tensorA, tensorB, false)
  deftensor bitwise_and(tensorA, tensorB, donate)
  def bitwise_or(tensorA, tensorB), do: bitwise_or(tensorA, tensorB, false)
  deftensor bitwise_or(tensorA, tensorB, donate)
  def bitwise_xor(tensorA, tensorB), do: bitwise_xor(tensorA, tensorB, false)
  deftensor bitwise_xor(tensorA, tensorB, donate)

  deftensor amax(tensor, axes, keep_axes)
  deftensor amin(tensor, axes, keep_axes)

  def tensordot(tensorA, tensorB, axesA, axesB),
    do: tensordot(tensorA, tensorB, axesA, [], axesB, [])

  deftensor tensordot(tensorA, tensorB, axesA, batchA, axesB, batchB)
  deftensor matmul(tensorA, tensorB)

  ## Unary ops

  def exp(tensor), do: exp(tensor, false)
  deftensor exp(tensor, donate)
  def expm1(tensor), do: expm1(tensor, false)
  deftensor expm1(tensor, donate)
  def log(tensor), do: log(tensor, false)
  deftensor log(tensor, donate)
  def log1p(tensor), do: log1p(tensor, false)
  deftensor log1p(tensor, donate)
  def sigmoid(tensor), do: sigmoid(tensor, false)
  deftensor sigmoid(tensor, donate)
  def view_as_real(tensor), do: view_as_real(tensor, false)
  deftensor view_as_real(tensor, donate)
  def conjugate(tensor), do: conjugate(tensor, false)
  deftensor conjugate(tensor, donate)
  def cos(tensor), do: cos(tensor, false)
  deftensor cos(tensor, donate)
  def sin(tensor), do: sin(tensor, false)
  deftensor sin(tensor, donate)
  def tan(tensor), do: tan(tensor, false)
  deftensor tan(tensor, donate)
  def cosh(tensor), do: cosh(tensor, false)
  deftensor cosh(tensor, donate)
  def sinh(tensor), do: sinh(tensor, false)
  deftensor sinh(tensor, donate)
  def tanh(tensor), do: tanh(tensor, false)
  deftensor tanh(tensor, donate)
  def acos(tensor), do: acos(tensor, false)
  deftensor acos(tensor, donate)
  def asin(tensor), do: asin(tensor, false)
  deftensor asin(tensor, donate)
  def atan(tensor), do: atan(tensor, false)
  deftensor atan(tensor, donate)
  def acosh(tensor), do: acosh(tensor, false)
  deftensor acosh(tensor, donate)
  def asinh(tensor), do: asinh(tensor, false)
  deftensor asinh(tensor, donate)
  def atanh(tensor), do: atanh(tensor, false)
  deftensor atanh(tensor, donate)
  def sqrt(tensor), do: sqrt(tensor, false)
  deftensor sqrt(tensor, donate)
  def rsqrt(tensor), do: rsqrt(tensor, false)
  deftensor rsqrt(tensor, donate)
  def erf(tensor), do: erf(tensor, false)
  deftensor erf(tensor, donate)
  def erfc(tensor), do: erfc(tensor, false)
  deftensor erfc(tensor, donate)
  def erf_inv(tensor), do: erf_inv(tensor, false)
  deftensor erf_inv(tensor, donate)
  deftensor cbrt(tensor)
  deftensor fft(tensor, length, axis)
  deftensor ifft(tensor, length, axis)
  deftensor rfft(tensor, length, axis)
  deftensor irfft(tensor, length, axis)
  deftensor fft2(tensor, lengths, axes)
  deftensor ifft2(tensor, lengths, axes)

  def abs(tensor), do: abs(tensor, false)
  deftensor abs(tensor, donate)
  def bitwise_not(tensor), do: bitwise_not(tensor, false)
  deftensor bitwise_not(tensor, donate)
  def ceil(tensor), do: ceil(tensor, false)
  deftensor ceil(tensor, donate)
  def floor(tensor), do: floor(tensor, false)
  deftensor floor(tensor, donate)
  def negate(tensor), do: negate(tensor, false)
  deftensor negate(tensor, donate)
  def round(tensor), do: round(tensor, false)
  deftensor round(tensor, donate)
  def sign(tensor), do: sign(tensor, false)
  deftensor sign(tensor, donate)
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
  deftensor sort(tensor, axis, descending, stable)
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

  @doc false
  def to_pointer({dev, ref}, mode, permissions)
      when is_tensor(dev, ref) and is_atom(mode) and is_integer(permissions) do
    NIF.to_pointer(ref, mode, permissions) |> unwrap!()
  end

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
          # Auto-transfer tensor to target device
          {^dev, new_ref} = Torchx.to_device({other_dev, ref}, dev)
          new_ref

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
        # Auto-transfer tensor to target device
        {^other_dev, new_ref} = Torchx.to_device({dev, ref}, other_dev)
        {new_ref, other_dev}

      [{dev, ref} | _] = tensors, nil when is_tensor(dev, ref) ->
        prepare_tensors_list!(tensors, dev)

      tensors, dev when is_list(tensors) ->
        prepare_tensors_list!(tensors, dev)

      bad_tensor, _dev ->
        raise ArgumentError, "expected a Torchx tensor, got: #{inspect(bad_tensor)}"
    end)
  end
end
