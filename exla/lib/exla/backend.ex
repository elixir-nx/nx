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
  def scalar(%{type: type, shape: shape} = out, scalar, _backend_options) do
    data = :binary.copy(number_to_binary(scalar, type), Nx.size(shape))
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


  binary_ops =
    [:add, :subtract, :multiply, :power, :remainder, :divide, :atan2, :min, :max, :quotient] ++
      [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift] ++
      [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal] ++
      [:logical_and, :logical_or, :logical_xor] ++
      [:outer]

  for binary_op <- binary_ops do
    @impl true
    def unquote(binary_op)(out, t1, t2) do
      expr_fn = fn lhs, rhs ->
        apply(Nx.Defn.Expr, unquote(binary_op), [out, lhs, rhs])
      end

      EXLA.jit(expr_fn, [t1, t2])
    end
  end

  unary_ops =
    [:exp, :expm1, :log, :log1p, :logistic, :cos, :sin, :tan] ++
     [:cosh, :sinh, :tanh, :acos, :asin, :atan, :acosh, :asinh] ++
     [:atanh, :sqrt, :rsqrt, :cbrt, :erf, :erfc, :erf_inv] ++
     [:abs, :bitwise_not, :ceil, :floor, :negate, :round, :sign] ++
     [:count_leading_zeros, :population_count]

  for unary_op <- unary_ops do
    @impl true
    def unquote(unary_op)(out, t) do
      expr_fn = fn x ->
        apply(Nx.Defn.Expr, unquote(unary_op), [out, x])
      end

      EXLA.jit(expr_fn, [t])
    end
  end

  window_aggregate_ops = [:window_sum, :window_product, :window_max, :window_min]

  for window_op <- window_aggregate_ops do
    @impl true
    def unquote(window_op)(out, tensor, window_dimensions, opts) do
      expr_fn = fn t ->
        apply(Nx.Defn.Expr, unquote(window_op), [out, t, window_dimensions, opts])
      end

      EXLA.jit(expr_fn, [tensor])
    end
  end

  aggregate_ops = [:all?, :any?, :argmax, :argmin, :sum, :product, :reduce_min, :reduce_max]

  for aggregate_op <- aggregate_ops do
    @impl true
    def unquote(aggregate_op)(out, tensor, opts) do
      expr_fn = fn t ->
        apply(Nx.Defn.Expr, unquote(aggregate_op), [out, t, opts])
      end

      EXLA.jit(expr_fn, [tensor])
    end
  end

  scatter_ops = [:scatter_window_min, :scatter_window_max]

  for scatter_op <- scatter_ops do
    @impl true
    def unquote(scatter_op)(out, tensor, source, dims, opts, init_value) do
      expr_fn = fn tensor, source, init_value ->
        apply(Nx.Defn.Expr, unquote(scatter_op), [out, tensor, source, dims, opts, init_value])
      end

      EXLA.jit(expr_fn, [tensor, source, init_value])
    end
  end

  conversion_ops = [:as_type, :bitcast]

  for conversion_op <- conversion_ops do
    @impl true
    def unquote(conversion_op)(out, tensor) do
      expr_fn = fn tensor ->
        apply(Nx.Defn.Expr, unquote(conversion_op), [out, tensor])
      end

      EXLA.jit(expr_fn, [tensor])
    end
  end

  shape_ops = [:reshape, :squeeze, :transpose, :reverse]

  for shape_op <- shape_ops do
    @impl true
    def unquote(shape_op)(out, tensor, axes_or_shape) do
      expr_fn = fn tensor ->
        apply(Nx.Defn.Expr, unquote(shape_op), [out, tensor, axes_or_shape])
      end

      EXLA.jit(expr_fn, [tensor])
    end
  end

  @impl true
  def broadcast(out, tensor, shape, axes) do
    expr_fn = fn tensor ->
      apply(Nx.Defn.Expr, :broadcast, [out, tensor, shape, axes])
    end

    EXLA.jit(expr_fn, [tensor])
  end

  @impl true
  def pad(out, tensor, pad_value, padding_config) do
    expr_fn = fn tensor, pad_value ->
      apply(Nx.Defn.Expr, :pad, [out, tensor, pad_value, padding_config])
    end

    EXLA.jit(expr_fn, [tensor, pad_value])
  end

  @impl true
  def dot(out, left, c1, b1, right, c2, b2) do
    expr_fn = fn left, right ->
      apply(Nx.Defn.Expr, :dot, [out, left, c1, b1, right, c2, b2])
    end

    EXLA.jit(expr_fn, [left, right])
  end

  @impl true
  def clip(out, tensor, min, max) do
    expr_fn = fn tensor, min, max ->
      apply(Nx.Defn.Expr, :clip, [out, tensor, min, max])
    end

    EXLA.jit(expr_fn, [tensor, min, max])
  end

  @impl true
  def slice(out, tensor, start, limit, strides) do
    expr_fn = fn tensor ->
      apply(Nx.Defn.Expr, :slice, [out, tensor, start, limit, strides])
    end

    EXLA.jit(expr_fn, [tensor])
  end

  @impl true
  def put_slice(out, tensor, put_tensor, location) do
    expr_fn = fn tensor, put_tensor ->
      apply(Nx.Defn.Expr, :put_slice, [out, tensor, put_tensor, location])
    end

    EXLA.jit(expr_fn, [tensor, put_tensor])
  end

  @impl true
  def concatenate(out, tensors, axis) do
    expr_fn = fn tensors ->
      apply(Nx.Defn.Expr, :concatenate, [out, Tuple.to_list(tensors), axis])
    end

    EXLA.jit(expr_fn, [List.to_tuple(tensors)])
  end

  @impl true
  def select(out, pred, on_true, on_false) do
    expr_fn = fn pred, on_true, on_false ->
      apply(Nx.Defn.Expr, :select, [out, pred, on_true, on_false])
    end

    EXLA.jit(expr_fn, [pred, on_true, on_false])
  end

  @impl true
  def conv(out, tensor, kernel, opts) do
    expr_fn = fn tensor, kernel ->
      apply(Nx.Defn.Expr, :conv, [out, tensor, kernel, opts])
    end

    EXLA.jit(expr_fn, [tensor, kernel])
  end

  @impl true
  def reduce(out, tensor, acc, opts, fun) do
    expr_fn = fn tensor, acc ->
      apply(Nx.Defn.Expr, :reduce, [out, tensor, acc, opts, fun])
    end

    EXLA.jit(expr_fn, [tensor, acc])
  end

  @impl true
  def reduce_window(out, tensor, acc, window, opts, fun) do
    expr_fn = fn tensor, acc ->
      apply(Nx.Defn.Expr, :reduce_window, [out, tensor, acc, window, opts, fun])
    end

    EXLA.jit(expr_fn, [tensor, acc])
  end

  @impl true
  def map(out, tensor, fun) do
    expr_fn = fn tensor ->
      apply(Nx.Defn.Expr, :map, [out, tensor, fun])
    end

    EXLA.jit(expr_fn, [tensor])
  end

  @impl true
  def sort(out, tensor, opts) do
    expr_fn = fn tensor ->
      apply(Nx.Defn.Expr, :sort, [out, tensor, opts])
    end

    EXLA.jit(expr_fn, [tensor])
  end

  @impl true
  def argsort(out, tensor, opts) do
    expr_fn = fn tensor ->
      apply(Nx.Defn.Expr, :argsort, [out, tensor, opts])
    end

    EXLA.jit(expr_fn, [tensor])
  end

  @impl true
  def cholesky(out, tensor) do
    expr_fn = fn tensor ->
      apply(Nx.Defn.Expr, :cholesky, [out, tensor])
    end

    EXLA.jit(expr_fn, [tensor])
  end

  @impl true
  def lu(out, tensor, opts) do
    expr_fn = fn tensor ->
      apply(Nx.Defn.Expr, :lu, [out, tensor, opts])
    end

    EXLA.jit(expr_fn, [tensor])
  end

  @impl true
  def qr(out, tensor, opts) do
    expr_fn = fn tensor ->
      apply(Nx.Defn.Expr, :qr, [out, tensor, opts])
    end

    EXLA.jit(expr_fn, [tensor])
  end

  @impl true
  def triangular_solve(out, a, b, opts) do
    expr_fn = fn a, b ->
      apply(Nx.Defn.Expr, :triangular_solve, [out, a, b, opts])
    end

    EXLA.jit(expr_fn, [as_type(out, a), as_type(out, b)])
  end

  @impl true
  def svd(out, tensor, opts) do
    expr_fn = fn tensor ->
      apply(Nx.Defn.Expr, :svd, [out, tensor, opts])
    end

    EXLA.jit(expr_fn, [tensor])
  end
end
