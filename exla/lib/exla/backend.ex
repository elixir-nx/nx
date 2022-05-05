defmodule EXLA.Backend do
  @moduledoc """
  A Nx tensor backend for the data kept on the device.

  You can directly transfer to this backend by calling
  `Nx.backend_transfer/2` or `Nx.backend_copy/2`. It
  allows the following options:

    * `:client` - the client to store the data on.
      Defaults to the client configured in `Nx.Defn`,
      otherwise uses `:host`.

    * `:device_id` - which device to store it on

  To get the data out of the device backend into a regular
  tensor, call `Nx.backend_transfer/1` (with the device
  tensor as the single argument).
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

  # TODO: Support direct transfers without going through Elixir
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
  def to_batched_list(out, tensor, opts) do
    leftover = opts[:leftover]

    batch_size = elem(out.shape, 0)
    axis_size = elem(tensor.shape, 0)

    remainder = rem(axis_size, batch_size)
    num_full_batches = div(axis_size, batch_size)

    expr_fun = fn tensor, start_idx ->
      Nx.slice_along_axis(tensor, start_idx, batch_size)
    end

    full_batches =
      for i <- 0..(num_full_batches - 1) do
        start_idx = i * batch_size
        EXLA.jit(expr_fun, [tensor, start_idx])
      end

    if remainder != 0 and leftover == :repeat do
      expr_fun = fn tensor ->
        Nx.concatenate([
          Nx.slice_along_axis(tensor, num_full_batches * batch_size, remainder),
          Nx.slice_along_axis(tensor, 0, batch_size - remainder)
        ])
      end

      last_batch = EXLA.jit(expr_fun, [tensor])
      full_batches ++ [last_batch]
    else
      full_batches
    end
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
      '#Ref<' ++ rest = :erlang.ref_to_list(ref)
      info = "EXLA.Backend<#{client_name}:#{device_id}, " <> List.to_string(rest)
      Inspect.Algebra.concat([info, Inspect.Algebra.line(), result])
    end
  else
    defp maybe_add_signature(result, _tensor) do
      result
    end
  end

  ## Helpers

  defp default_client_name do
    opts = Nx.Defn.default_options()

    if opts[:compiler] == EXLA do
      opts[:client] || :host
    else
      :host
    end
  end

  defp client_and_device_id(opts) do
    client = EXLA.Client.fetch!(opts[:client] || default_client_name())
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
    expr_fun = fn tensors ->
      Nx.Defn.Expr.concatenate(out, Tuple.to_list(tensors), axis)
    end

    EXLA.jit(expr_fun, [List.to_tuple(tensors)])
  end

  @impl true
  def optional(_name, args, fun) do
    # Here we take the leading tensor arguments and pass them as JIT arguments
    {tensors, rest} = Enum.split_while(args, &is_struct(&1, Nx.Tensor))

    wrapper_fun = fn tensors ->
      tensors = Tuple.to_list(tensors)
      apply(fun, tensors ++ rest)
    end

    EXLA.jit(wrapper_fun, [List.to_tuple(tensors)])
  end

  binary_ops =
    [:add, :subtract, :multiply, :power, :remainder, :divide, :atan2, :min, :max, :quotient] ++
      [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift] ++
      [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal] ++
      [:logical_and, :logical_or, :logical_xor]

  unary_ops =
    [:exp, :expm1, :log, :log1p, :logistic, :cos, :sin, :tan] ++
      [:cosh, :sinh, :tanh, :acos, :asin, :atan, :acosh, :asinh, :atanh] ++
      [:sqrt, :rsqrt, :cbrt, :erf, :erfc, :erf_inv] ++
      [:abs, :bitwise_not, :ceil, :conjugate, :floor, :negate, :round, :sign] ++
      [:count_leading_zeros, :population_count, :real, :imag]

  callbacks =
    [
      {:eye, [:out, :backend_options], []},
      {:iota, [:out, :axis, :backend_options], []},
      {:random_uniform, [:out, :min, :max, :backend_options], [:min, :max]},
      {:random_normal, [:out, :mu, :sigma, :backend_options], [:mu, :sigma]},
      {:as_type, [:out, :tensor], [:tensor]},
      {:bitcast, [:out, :tensor], [:tensor]},
      {:reshape, [:out, :tensor], [:tensor]},
      {:squeeze, [:out, :tensor, :axes], [:tensor]},
      {:broadcast, [:out, :tensor, :shape, :axes], [:tensor]},
      {:transpose, [:out, :tensor, :axes], [:tensor]},
      {:pad, [:out, :tensor, :pad_value, :padding_config], [:tensor, :pad_value]},
      {:reverse, [:out, :tensor, :axes], [:tensor]},
      {:dot, [:out, :left, :c1, :b1, :right, :c2, :b2], [:left, :right]},
      {:clip, [:out, :tensor, :min, :max], [:tensor, :min, :max]},
      {:slice, [:out, :tensor, :start_indices, :lengths, :strides], [:tensor]},
      {:put_slice, [:out, :tensor, :start, :slice], [:tensor, :slice]},
      {:take, [:out, :tensor, :indices, :axis], [:tensor, :indices]},
      {:take_along_axis, [:out, :tensor, :indices, :axis], [:tensor, :indices]},
      {:gather, [:out, :input, :indices], [:input, :indices]},
      {:select, [:out, :pred, :on_true, :on_false], [:pred, :on_true, :on_false]},
      {:conv, [:out, :tensor, :kernel, :opts], [:tensor, :kernel]},
      {:all, [:out, :tensor, :opts], [:tensor]},
      {:any, [:out, :tensor, :opts], [:tensor]},
      {:sum, [:out, :tensor, :opts], [:tensor]},
      {:product, [:out, :tensor, :opts], [:tensor]},
      {:reduce_max, [:out, :tensor, :opts], [:tensor]},
      {:reduce_min, [:out, :tensor, :opts], [:tensor]},
      {:argmax, [:out, :tensor, :opts], [:tensor]},
      {:argmin, [:out, :tensor, :opts], [:tensor]},
      {:reduce, [:out, :tensor, :acc, :opts, :fun], [:tensor, :acc]},
      {:window_reduce, [:out, :tensor, :acc, :shape, :opts, :fun], [:tensor, :acc]},
      {:window_sum, [:out, :tensor, :shape, :opts], [:tensor]},
      {:window_product, [:out, :tensor, :shape, :opts], [:tensor]},
      {:window_max, [:out, :tensor, :shape, :opts], [:tensor]},
      {:window_min, [:out, :tensor, :shape, :opts], [:tensor]},
      {:map, [:out, :tensor, :opts, :fun], [:tensor]},
      {:sort, [:out, :tensor, :opts], [:tensor]},
      {:argsort, [:out, :tensor, :opts], [:tensor]},
      {:window_scatter_max, [:out, :tensor, :source, :init_value, :window_dims, :opts],
       [:tensor, :source, :init_value]},
      {:window_scatter_min, [:out, :tensor, :source, :init_value, :window_dims, :opts],
       [:tensor, :source, :init_value]},
      {:indexed_add, [:out, :tensor, :indices, :updates], [:indices, :updates]},
      {:cholesky, [:out, :tensor], [:tensor]},
      {:lu, [:out, :tensor, :opts], [:tensor]},
      {:qr, [:out, :tensor, :opts], [:tensor]},
      {:triangular_solve, [:out, :a, :b, :opts], [:a, :b]},
      {:eigh, [:out, :tensor, :opts], [:tensor]},
      {:svd, [:out, :tensor, :opts], [:tensor]}
    ] ++
      for(op <- binary_ops, do: {op, [:out, :left, :right], [:left, :right]}) ++
      for(op <- unary_ops, do: {op, [:out, :tensor], [:tensor]})

  for {name, args, tensor_args} <- callbacks do
    args = Enum.map(args, &Macro.var(&1, __MODULE__))
    tensor_args = Enum.map(tensor_args, &Macro.var(&1, __MODULE__))

    @impl true
    def unquote(name)(unquote_splicing(args)) do
      expr_fun = fn unquote_splicing(tensor_args) ->
        Nx.Defn.Expr.unquote(name)(unquote_splicing(args))
      end

      EXLA.jit(expr_fun, [unquote_splicing(tensor_args)])
    end
  end
end
