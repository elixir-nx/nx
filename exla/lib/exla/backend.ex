defmodule EXLA.Backend do
  @moduledoc """
  EXLA Nx backend implementation for usage of EXLA outside of defn.

  The EXLA backend is designed for rapid prototyping of accelerated Nx programs
  and ensures consistency from prototype to implementation. Generally, you'll want
  to experiment with ideas before organizing them into a project. Backends allow
  you to experiment with Nx in "eager mode" before utilizing the stricter, but
  more performant constructs such as `defn` and JIT compilation.

  While you can also prototype with the BinaryBackend, the EXLA backend offers
  the following advantages:

    1) Performance - The EXLA backend executes functions on CPU/GPU/TPU.

    2) Consistency - You may encounter slight inconsistencies in behavior between
    the BinaryBackend and the EXLA backend which leads to different behavior when
    changing "eager mode" code to compiled code. Using EXLA in "eager mode" ensures
    behavior is consistent when transitioning from eager mode to compiled.
  """

  defstruct [:state]

  @behaviour Nx.Backend

  alias EXLA.Backend, as: B
  import Nx.Shared

  # For now this behaves like the BinaryBackend; however, we can switch this
  # behavior to always make tensors as EXLA Buffers, and all of the backend
  # stuff transfers between them

  @impl true
  def constant(%{type: type, shape: shape} = out, constant, _backend_options) do
    data = :binary.copy(number_to_binary(constant, type), Nx.size(shape))
    from_binary(out, data)
  end

  defp number_to_binary(number, type),
    do: match_types([type], do: <<write!(number, 0)>>)

  @impl true
  def from_binary(t, binary, _backend_options), do: from_binary(t, binary)

  defp from_binary(t, binary) when is_binary(binary), do: %{t | data: %B{state: binary}}
  defp from_binary(t, other), do: %{t | data: %B{state: IO.iodata_to_binary(other)}}

  @impl true
  defdelegate backend_copy(tensor, backend, opts), to: Nx.BinaryBackend

  @impl true
  defdelegate backend_transfer(tensor, backend, opts), to: Nx.BinaryBackend

  @impl true
  defdelegate backend_deallocate(tensor), to: Nx.BinaryBackend

  @impl true
  defdelegate to_binary(t, limit), to: Nx.BinaryBackend

  @impl true
  defdelegate to_batched_list(out, t, x), to: Nx.BinaryBackend

  @impl true
  def eye(%{shape: shape, type: type, names: names}, _) do
    fun = fn ->
      Nx.eye(shape, type: type, names: names, backend: Nx.Defn.Expr)
    end

    EXLA.jit(fun, [])
  end

  @impl true
  def iota(%{shape: shape, type: type, names: names}, axis, _) do
    fun = fn ->
      Nx.iota(shape, type: type, names: names, axis: axis, backend: Nx.Defn.Expr)
    end

    EXLA.jit(fun, [])
  end

  @impl true
  def random_uniform(%{shape: shape, type: type, names: names}, min, max, _) do
    fun = fn ->
      Nx.random_uniform(shape, min, max, type: type, names: names, backend: Nx.Defn.Expr)
    end

    EXLA.jit(fun, [])
  end

  @impl true
  def random_normal(%{shape: shape, type: type, names: names}, mu, sigma, _) do
    fun = fn ->
      Nx.random_normal(shape, mu, sigma, type: type, names: names, backend: Nx.Defn.Expr)
    end

    EXLA.jit(fun, [])
  end

  @impl true
  def inspect(tensor, inspect_opts) do
    limit = inspect_opts.limit
    binary = Nx.to_binary(tensor, if(limit == :infinity, do: [], else: [limit: limit + 1]))
    Nx.Backend.inspect(tensor, binary, inspect_opts)
  end

  @impl true
  def concatenate(out, tensors, axis) do
    expr_fn = fn tensors ->
      apply(Nx.Defn.Expr, :concatenate, [out, Tuple.to_list(tensors), axis])
    end

    EXLA.jit(expr_fn, [List.to_tuple(tensors)])
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
      expr_fn = fn unquote_splicing(tensor_args) ->
        Nx.Defn.Expr.unquote(name)(unquote_splicing(args))
      end

      EXLA.jit(expr_fn, [unquote_splicing(tensor_args)])
    end
  end
end
