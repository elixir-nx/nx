defmodule EXLA.Backend do
  @moduledoc """
  A Nx tensor backend for the data kept on the device.

  You can directly transfer to this backend by calling
  `Nx.backend_transfer/2` or `Nx.backend_copy/2`. It
  allows the following options:

    * `:client` - the client to store the data on.
      Defaults to EXLA's default client.

    * `:device_id` - which device to store it on.

  To get the data out of the device backend into a regular
  tensor, call `Nx.backend_transfer/1` (with the device
  tensor as the single argument).

  Note that the `EXLA.Backend` is asynchronous: operations
  on its tensors *may* return immediately, before the tensor
  data is available. The backend will then block only when
  trying to read the data or when passing it to another operation.
  """

  @behaviour Nx.Backend
  @enforce_keys [:buffer]
  defstruct [:buffer]

  alias Nx.Tensor, as: T
  alias EXLA.Backend, as: B

  @impl true
  def constant(out, constant, backend_options) do
    binary_tensor = Nx.BinaryBackend.constant(out, constant, [])
    Nx.BinaryBackend.backend_transfer(binary_tensor, __MODULE__, backend_options)
  end

  @impl true
  def from_binary(%T{shape: shape, type: type} = tensor, binary, backend_options) do
    {client, device_id} = client_and_device_id(backend_options)
    shape = EXLA.Shape.make_shape(type, shape)
    buffer = EXLA.DeviceBuffer.place_on_device(binary, shape, client, device_id)
    put_in(tensor.data, %B{buffer: buffer})
  end

  @impl true
  def backend_copy(tensor, Nx.Tensor, backend_options) do
    backend_copy(tensor, Nx.BinaryBackend, backend_options)
  end

  # TODO: Support direct transfers between EXLA without going through Elixir
  def backend_copy(%T{data: %B{buffer: buffer}} = tensor, backend, backend_options) do
    backend.from_binary(tensor, EXLA.DeviceBuffer.read(buffer), backend_options)
  end

  @impl true
  def backend_transfer(%T{data: %B{buffer: buffer}} = tensor, backend, backend_options) do
    if backend == __MODULE__ and same_client_device?(buffer, backend_options) do
      tensor
    else
      try do
        backend_copy(tensor, backend, backend_options)
      after
        EXLA.DeviceBuffer.deallocate(buffer)
      end
    end
  end

  @impl true
  def backend_deallocate(%T{data: %B{buffer: buffer}}) do
    EXLA.DeviceBuffer.deallocate(buffer)
  end

  @impl true
  def to_batched(out, tensor, opts) do
    leftover = opts[:leftover]

    batch_size = elem(out.shape, 0)
    axis_size = elem(tensor.shape, 0)

    remainder = rem(axis_size, batch_size)
    num_full_batches = div(axis_size, batch_size)

    range =
      if remainder != 0 and leftover == :repeat do
        0..num_full_batches
      else
        0..(num_full_batches - 1)
      end

    Stream.map(range, fn
      ^num_full_batches ->
        expr_fun = fn tensor ->
          Nx.concatenate([
            Nx.slice_along_axis(tensor, num_full_batches * batch_size, remainder),
            Nx.slice_along_axis(tensor, 0, batch_size - remainder)
          ])
        end

        jit(expr_fun, [tensor])

      i ->
        expr_fun = fn tensor, start_idx ->
          Nx.slice_along_axis(tensor, start_idx, batch_size)
        end

        start_idx = i * batch_size
        jit(expr_fun, [tensor, start_idx])
    end)
  end

  @impl true
  def to_binary(%T{data: %B{buffer: buffer}, type: {_, size}}, limit) do
    EXLA.DeviceBuffer.read(buffer, limit * div(size, 8))
  end

  @impl true
  def inspect(%T{} = tensor, inspect_opts) do
    limit = if inspect_opts.limit == :infinity, do: :infinity, else: inspect_opts.limit + 1

    tensor
    |> to_binary(min(limit, Nx.size(tensor)))
    |> then(&Nx.Backend.inspect(tensor, &1, inspect_opts))
    |> maybe_add_signature(tensor)
  end

  if Application.compile_env(:exla, :add_backend_on_inspect, true) do
    defp maybe_add_signature(result, %T{data: %B{buffer: buffer}}) do
      %EXLA.DeviceBuffer{client_name: client_name, device_id: device_id, ref: ref} = buffer
      ~c"#Ref<" ++ rest = :erlang.ref_to_list(ref)
      info = "EXLA.Backend<#{client_name}:#{device_id}, " <> List.to_string(rest)
      Inspect.Algebra.concat([info, Inspect.Algebra.line(), result])
    end
  else
    defp maybe_add_signature(result, _tensor) do
      result
    end
  end

  ## Helpers

  defp client_and_device_id(opts) do
    client = EXLA.Client.fetch!(opts[:client] || EXLA.Client.default_name())
    device_id = opts[:device_id] || client.default_device_id
    {client, device_id}
  end

  defp same_client_device?(buffer, opts) do
    {client, device_id} = client_and_device_id(opts)
    buffer.client_name == client.name and buffer.device_id == device_id
  end

  ## JIT callbacks

  @impl true
  def concatenate(out, tensors, axis) do
    out = Nx.to_template(out)

    expr_fun = fn tensors ->
      Nx.Defn.Expr.concatenate(out, Tuple.to_list(tensors), axis)
    end

    jit(expr_fun, [List.to_tuple(tensors)])
  end

  @impl true
  def slice(out, tensor, start_indices, lengths, strides) do
    out = Nx.to_template(out)

    if Enum.all?(start_indices, &is_integer/1) do
      expr_fun = fn tensor ->
        Nx.Defn.Expr.slice(out, tensor, start_indices, lengths, strides)
      end

      jit(expr_fun, [tensor])
    else
      expr_fun = fn tensor, start_indices ->
        Nx.Defn.Expr.slice(out, tensor, Tuple.to_list(start_indices), lengths, strides)
      end

      jit(expr_fun, [tensor, List.to_tuple(start_indices)])
    end
  end

  @impl true
  def put_slice(out, tensor, start_indices, slice) do
    out = Nx.to_template(out)

    if Enum.all?(start_indices, &is_integer/1) do
      expr_fun = fn tensor, slice ->
        Nx.Defn.Expr.put_slice(out, tensor, start_indices, slice)
      end

      jit(expr_fun, [tensor, slice])
    else
      expr_fun = fn tensor, start_indices, slice ->
        Nx.Defn.Expr.put_slice(out, tensor, Tuple.to_list(start_indices), slice)
      end

      jit(expr_fun, [tensor, List.to_tuple(start_indices), slice])
    end
  end

  @impl true
  def optional(name, args, fun) do
    # Here we take the leading tensor arguments and pass them as JIT arguments
    {tensors, rest} = Enum.split_while(args, &is_struct(&1, Nx.Tensor))

    wrapper_fun = fn tensors ->
      Nx.Defn.Expr.optional(name, Tuple.to_list(tensors) ++ rest, fun)
    end

    jit(wrapper_fun, [List.to_tuple(tensors)])
  end

  binary_ops =
    [:add, :subtract, :multiply, :power, :remainder, :divide, :atan2, :min, :max, :quotient] ++
      [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift] ++
      [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal] ++
      [:logical_and, :logical_or, :logical_xor]

  unary_ops =
    [:exp, :expm1, :log, :log1p, :sigmoid, :cos, :sin, :tan] ++
      [:cosh, :sinh, :tanh, :acos, :asin, :atan, :acosh, :asinh, :atanh] ++
      [:sqrt, :rsqrt, :cbrt, :is_nan, :is_infinity, :erf, :erfc, :erf_inv] ++
      [:abs, :bitwise_not, :ceil, :conjugate, :floor, :negate, :round, :sign] ++
      [:count_leading_zeros, :population_count, :real, :imag]

  callbacks =
    [
      {:eye, [:backend_options], []},
      {:iota, [:axis, :backend_options], []},
      {:random_uniform, [:min, :max, :backend_options], [:min, :max]},
      {:random_normal, [:mu, :sigma, :backend_options], [:mu, :sigma]},
      {:as_type, [:tensor], [:tensor]},
      {:bitcast, [:tensor], [:tensor]},
      {:reshape, [:tensor], [:tensor]},
      {:squeeze, [:tensor, :axes], [:tensor]},
      {:broadcast, [:tensor, :shape, :axes], [:tensor]},
      {:transpose, [:tensor, :axes], [:tensor]},
      {:pad, [:tensor, :pad_value, :padding_config], [:tensor, :pad_value]},
      {:reverse, [:tensor, :axes], [:tensor]},
      {:dot, [:left, :c1, :b1, :right, :c2, :b2], [:left, :right]},
      {:clip, [:tensor, :min, :max], [:tensor, :min, :max]},
      {:take, [:tensor, :indices, :axis], [:tensor, :indices]},
      {:take_along_axis, [:tensor, :indices, :axis], [:tensor, :indices]},
      {:gather, [:input, :indices], [:input, :indices]},
      {:select, [:pred, :on_true, :on_false], [:pred, :on_true, :on_false]},
      {:conv, [:tensor, :kernel, :opts], [:tensor, :kernel]},
      {:all, [:tensor, :opts], [:tensor]},
      {:any, [:tensor, :opts], [:tensor]},
      {:sum, [:tensor, :opts], [:tensor]},
      {:product, [:tensor, :opts], [:tensor]},
      {:reduce_max, [:tensor, :opts], [:tensor]},
      {:reduce_min, [:tensor, :opts], [:tensor]},
      {:argmax, [:tensor, :opts], [:tensor]},
      {:argmin, [:tensor, :opts], [:tensor]},
      {:reduce, [:tensor, :acc, :opts, :fun], [:tensor, :acc]},
      {:window_reduce, [:tensor, :acc, :shape, :opts, :fun], [:tensor, :acc]},
      {:window_sum, [:tensor, :shape, :opts], [:tensor]},
      {:window_product, [:tensor, :shape, :opts], [:tensor]},
      {:window_max, [:tensor, :shape, :opts], [:tensor]},
      {:window_min, [:tensor, :shape, :opts], [:tensor]},
      {:map, [:tensor, :opts, :fun], [:tensor]},
      {:sort, [:tensor, :opts], [:tensor]},
      {:argsort, [:tensor, :opts], [:tensor]},
      {:window_scatter_max, [:tensor, :source, :init_value, :window_dims, :opts],
       [:tensor, :source, :init_value]},
      {:window_scatter_min, [:tensor, :source, :init_value, :window_dims, :opts],
       [:tensor, :source, :init_value]},
      {:indexed_add, [:tensor, :indices, :updates], [:tensor, :indices, :updates]},
      {:indexed_put, [:tensor, :indices, :updates], [:tensor, :indices, :updates]},
      {:cholesky, [:tensor], [:tensor]},
      {:lu, [:tensor, :opts], [:tensor]},
      {:qr, [:tensor, :opts], [:tensor]},
      {:triangular_solve, [:a, :b, :opts], [:a, :b]},
      {:eigh, [:tensor, :opts], [:tensor]},
      {:svd, [:tensor, :opts], [:tensor]},
      {:fft, [:tensor, :opts], [:tensor]},
      {:ifft, [:tensor, :opts], [:tensor]}
    ] ++
      for(op <- binary_ops, do: {op, [:left, :right], [:left, :right]}) ++
      for(op <- unary_ops, do: {op, [:tensor], [:tensor]})

  for {name, args, tensor_args} <- callbacks do
    args = Enum.map(args, &Macro.var(&1, __MODULE__))
    tensor_args = Enum.map(tensor_args, &Macro.var(&1, __MODULE__))

    @impl true
    def unquote(name)(out, unquote_splicing(args)) do
      out = Nx.to_template(out)

      expr_fun = fn unquote_splicing(tensor_args) ->
        Nx.Defn.Expr.unquote(name)(out, unquote_splicing(args))
      end

      jit(expr_fun, [unquote_splicing(tensor_args)])
    end
  end

  defp jit(fun, args), do: EXLA.jit_apply(fun, args, on_conflict: :force)
end
