defmodule Nx.LinAlg do
  @moduledoc """
  Nx conveniences for linear algebra.
  """

  import Nx.Shared
  import Nx.Defn, only: [defn: 2, defnp: 2, deftransformp: 2]
  import Nx.Defn.Kernel, only: [keyword!: 2, custom_grad: 2]

  alias Nx.Tensor, as: T

  @default_eps 1.0e-10

  @doc """
  Returns the adjoint of a given tensor.

  If the input tensor is real, it is the same as `Nx.transpose/2`.
  Otherwise, it is the same as `tensor |> Nx.transpose(opts) |> Nx.conjugate()`.

  ## Examples

      iex> Nx.LinAlg.adjoint(Nx.tensor([[1, 2], [3, 4]]))
      #Nx.Tensor<
        s64[2][2]
        [
          [1, 3],
          [2, 4]
        ]
      >

      iex> Nx.LinAlg.adjoint(Nx.tensor([[1, Complex.new(0, 2)], [3, Complex.new(0, -4)]]))
      #Nx.Tensor<
        c64[2][2]
        [
          [1.0+0.0i, 3.0+0.0i],
          [0.0-2.0i, 0.0+4.0i]
        ]
      >
  """
  defn adjoint(t, opts \\ []) do
    case Nx.type(t) do
      {:c, _} ->
        t |> Nx.transpose(opts) |> Nx.conjugate()

      _ ->
        Nx.transpose(t, opts)
    end
  end

  @doc """
  Performs a Cholesky decomposition of a square matrix.

  The matrix must be positive-definite and either Hermitian
  if complex or symmetric if real. An error is raised by the
  default backend if those conditions are not met. Other
  backends may emit undefined behaviour.

  ### Examples

      iex> Nx.LinAlg.cholesky(Nx.tensor([[20.0, 17.6], [17.6, 16.0]]))
      #Nx.Tensor<
        f32[2][2]
        [
          [4.4721360206604, 0.0],
          [3.9354796409606934, 0.7155413031578064]
        ]
      >

      iex> t = Nx.tensor([
      ...>   [6.0, 3.0, 4.0, 8.0],
      ...>   [3.0, 6.0, 5.0, 1.0],
      ...>   [4.0, 5.0, 10.0, 7.0],
      ...>   [8.0, 1.0, 7.0, 25.0]
      ...> ])
      iex> Nx.LinAlg.cholesky(t)
      #Nx.Tensor<
        f32[4][4]
        [
          [2.4494898319244385, 0.0, 0.0, 0.0],
          [1.2247449159622192, 2.1213202476501465, 0.0, 0.0],
          [1.632993221282959, 1.4142135381698608, 2.309401035308838, 0.0],
          [3.265986442565918, -1.4142135381698608, 1.5877132415771484, 3.132491111755371]
        ]
      >

      iex> Nx.LinAlg.cholesky(Nx.tensor([[1.0, Complex.new(0, -2)], [Complex.new(0, 2), 5.0]]))
      #Nx.Tensor<
        c64[2][2]
        [
          [1.0+0.0i, 0.0+0.0i],
          [0.0+2.0i, 1.0+0.0i]
        ]
      >

  ### Error cases

      iex> Nx.LinAlg.cholesky(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]))
      ** (ArgumentError) matrix must be hermitian, a matrix is hermitian iff X = adjoint(X)
  """
  def cholesky(tensor) do
    %T{type: type, shape: shape, names: names} = tensor = Nx.to_tensor(tensor)

    output_type = Nx.Type.to_floating(type)

    {output_shape, output_names} = Nx.Shape.cholesky(shape, names)

    out = %{tensor | type: output_type, shape: output_shape, names: output_names}
    impl!(tensor).cholesky(out, tensor)
  end

  @doc """
  Calculates the p-norm of a tensor.

  For the 0-norm, the norm is the number of non-zero elements in the tensor.

  ## Options

    * `:axes` - defines the axes upon which the norm will be calculated.
      Applies only on 2-norm for 2-D tensors. Default: `nil`.
    * `:ord` - defines which norm will be calculated according to the table below. Default: `2`

  | ord          | 2-D                            | 1-D                               |
  | ------------ | -------------------------------| --------------------------------- |
  | `nil`        | Frobenius norm                 | 2-norm                            |
  | `:nuclear`   | Nuclear norm                   | -                                 |
  | `:frobenius` | Frobenius norm                 | -                                 |
  | `:inf`       | `max(sum(abs(x), axes: [1]))`  | `max(abs(x))`                     |
  | `:neg_inf`   | `min(sum(abs(x), axes: [1]))`  | `min(abs(x))`                     |
  | 0            | -                              | Number of non-zero elements       |
  | 1            | `max(sum(abs(x), axes: [0]))`  | as below                          |
  | -1           | `min(sum(abs(x), axes: [0]))`  | as below                          |
  | 2            | 2-norm                         | as below                          |
  | -2           | smallest singular value        | as below                          |
  | other        | -                              | power(sum(power(abs(x), p)), 1/p) |

  ## Examples

  ### Vector norms

      iex> Nx.LinAlg.norm(Nx.tensor([3, 4]))
      #Nx.Tensor<
        f32
        5.0
      >

      iex> Nx.LinAlg.norm(Nx.tensor([3, 4]), ord: 1)
      #Nx.Tensor<
        f32
        7.0
      >

      iex> Nx.LinAlg.norm(Nx.tensor([3, -4]), ord: :inf)
      #Nx.Tensor<
        f32
        4.0
      >

      iex> Nx.LinAlg.norm(Nx.tensor([3, -4]), ord: :neg_inf)
      #Nx.Tensor<
        f32
        3.0
      >

      iex> Nx.LinAlg.norm(Nx.tensor([3, -4, 0, 0]), ord: 0)
      #Nx.Tensor<
        f32
        2.0
      >

  ### Matrix norms

      iex> Nx.LinAlg.norm(Nx.tensor([[3, -1], [2, -4]]), ord: -1)
      #Nx.Tensor<
        f32
        5.0
      >

      iex> Nx.LinAlg.norm(Nx.tensor([[3, -2], [2, -4]]), ord: 1)
      #Nx.Tensor<
        f32
        6.0
      >

      iex> Nx.LinAlg.norm(Nx.tensor([[3, -2], [2, -4]]), ord: :neg_inf)
      #Nx.Tensor<
        f32
        5.0
      >

      iex> Nx.LinAlg.norm(Nx.tensor([[3, -2], [2, -4]]), ord: :inf)
      #Nx.Tensor<
        f32
        6.0
      >

      iex> Nx.LinAlg.norm(Nx.tensor([[3, 0], [0, -4]]), ord: :frobenius)
      #Nx.Tensor<
        f32
        5.0
      >

      iex> Nx.LinAlg.norm(Nx.tensor([[1, 0, 0], [0, -4, 0], [0, 0, 9]]), ord: :nuclear)
      #Nx.Tensor<
        f32
        14.0
      >

      iex> Nx.LinAlg.norm(Nx.tensor([[1, 0, 0], [0, -4, 0], [0, 0, 9]]), ord: -2)
      #Nx.Tensor<
        f32
        1.0
      >

      iex> Nx.LinAlg.norm(Nx.tensor([[3, 0], [0, -4]]))
      #Nx.Tensor<
        f32
        5.0
      >

      iex> Nx.LinAlg.norm(Nx.tensor([[3, 4], [0, -4]]), axes: [1])
      #Nx.Tensor<
        f32[2]
        [5.0, 4.0]
      >

      iex> Nx.LinAlg.norm(Nx.tensor([[Complex.new(0, 3), 4], [4, 0]]), axes: [0])
      #Nx.Tensor<
        f32[2]
        [5.0, 4.0]
      >

      iex> Nx.LinAlg.norm(Nx.tensor([[Complex.new(0, 3), 0], [4, 0]]), ord: :neg_inf)
      #Nx.Tensor<
        f32
        3.0
      >

  ### Error cases

      iex> Nx.LinAlg.norm(Nx.tensor([3, 4]), ord: :frobenius)
      ** (ArgumentError) expected a 2-D tensor for ord: :frobenius, got a 1-D tensor
  """
  @doc from_backend: false
  defn norm(tensor, opts \\ []) do
    opts = keyword!(opts, [:ord, :axes])
    norm_transform(tensor, opts)
  end

  deftransformp norm_transform(t, opts) do
    rank = Nx.rank(t)

    unless rank == 1 or rank == 2 do
      raise ArgumentError, "expected 1-D or 2-D tensor, got tensor with shape #{inspect(t.shape)}"
    end

    axes_opts = Keyword.take(opts, [:axes])

    case opts[:ord] do
      nil when rank == 1 -> norm_integer(t, 2, axes_opts)
      nil when rank == 2 -> norm_integer(t, 2, axes_opts)
      :frobenius -> norm_frobenius(t, axes_opts)
      :nuclear when rank == 2 -> norm_nuclear(t)
      :nuclear -> raise ArgumentError, "nuclear norm not supported for rank != 2"
      ord when ord in [:inf, :neg_inf] -> norm_inf(t, ord, axes_opts)
      ord when is_integer(ord) -> norm_integer(t, ord, axes_opts)
      ord -> raise ArgumentError, "unknown ord #{inspect(ord)}"
    end
  end

  defp norm_frobenius(%{shape: {_}}, _opts),
    do: raise(ArgumentError, "expected a 2-D tensor for ord: :frobenius, got a 1-D tensor")

  defp norm_frobenius(%{shape: {_, _}} = t, opts), do: norm_integer(t, 2, opts)

  defp norm_nuclear(%{shape: {_, _}} = t) do
    {_u, s, _v} = svd(t)
    Nx.sum(s)
  end

  defp norm_inf(%{shape: shape, type: type} = t, ord, _opts) when ord in [:inf, :neg_inf] do
    output_type = Nx.Type.to_real(type)
    aggregate_axes = if tuple_size(shape) == 2, do: &Nx.sum(&1, axes: [1]), else: & &1
    reduce = if ord == :inf, do: &Nx.reduce_max/1, else: &Nx.reduce_min/1

    t
    |> Nx.abs()
    |> aggregate_axes.()
    |> reduce.()
    |> Nx.as_type(output_type)
  end

  defp norm_integer(%{shape: {_}, type: type} = t, 0, _opts) do
    output_type = Nx.Type.to_real(type)

    t
    |> Nx.not_equal(0)
    |> Nx.sum()
    |> Nx.as_type(output_type)
  end

  defp norm_integer(%{shape: {_, _}, type: type} = t, ord, _opts) when ord in [1, -1] do
    output_type = Nx.Type.to_real(type)
    function = if ord == 1, do: &Nx.reduce_max/1, else: &Nx.reduce_min/1

    t
    |> Nx.abs()
    |> Nx.sum(axes: [0])
    |> function.()
    |> Nx.as_type(output_type)
  end

  defp norm_integer(%{shape: {_, _}}, ord, _opts) when ord not in [-2, -1, 1, 2] do
    raise ArgumentError, "invalid :ord for 2-D tensor, got: #{inspect(ord)}"
  end

  defp norm_integer(%{shape: {_, _}} = t, -2, _opts) do
    {_u, s, _v} = svd(t)
    Nx.reduce_min(s)
  end

  defp norm_integer(%{type: type} = t, ord, opts) when is_integer(ord) do
    output_type = Nx.Type.to_real(type)
    inv_ord = Nx.tensor(1 / ord, type: output_type)

    # We extract this result to a variable because it's used both for
    # getting the normalization coefficient and for the main pipe chain
    abs_t = Nx.abs(t)

    # This coefficient is introduced for better numerical stability
    # The idea is that by dividing the tensor by it, large values of
    # tensor entries and large values of p are reduced, which in turn
    # avoids numerical overflow.
    numerical_stability_coefficient = Nx.reduce_max(abs_t)

    abs_t
    |> Nx.divide(numerical_stability_coefficient)
    |> Nx.power(ord)
    |> Nx.sum(opts)
    |> Nx.power(inv_ord)
    |> Nx.multiply(numerical_stability_coefficient)
  end

  @doc """
  Solve the equation `a x = b` for x, assuming `a` is a triangular matrix.
  Can also solve `x a = b` for x. See the `:left_side` option below.

  `b` must either be a square matrix with the same dimensions as `a` or a 1-D tensor
  with as many rows as `a`.

  ## Options

  The following options are defined in order of precedence

  * `:transform_a` - Defines `op(a)`, depending on its value. Can be one of:
    * `:none` -> `op(a) = a`
    * `:transpose` -> `op(a) = transpose(a)`
    Defaults to `:none`
  * `:lower` - When `true`, defines the `a` matrix as lower triangular. If false, a is upper triangular.
    Defaults to `true`
  * `:left_side` - When `true`, solves the system as `op(A).X = B`. Otherwise, solves `X.op(A) = B`. Defaults to `true`.

  ## Examples

      iex> a = Nx.tensor([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
      iex> Nx.LinAlg.triangular_solve(a, Nx.tensor([4, 2, 4, 2]))
      #Nx.Tensor<
        f32[4]
        [1.3333333730697632, -0.6666666865348816, 2.6666667461395264, -1.3333333730697632]
      >

      iex> a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], type: {:f, 64})
      iex> Nx.LinAlg.triangular_solve(a, Nx.tensor([1, 2, 1]))
      #Nx.Tensor<
        f64[3]
        [1.0, 1.0, -1.0]
      >

      iex> a = Nx.tensor([[1, 0, 0], [1, 1, 0], [0, 1, 1]])
      iex> b = Nx.tensor([[1, 2, 3], [2, 2, 4], [2, 0, 1]])
      iex> Nx.LinAlg.triangular_solve(a, b)
      #Nx.Tensor<
        f32[3][3]
        [
          [1.0, 2.0, 3.0],
          [1.0, 0.0, 1.0],
          [1.0, 0.0, 0.0]
        ]
      >

      iex> a = Nx.tensor([[1, 1, 1, 1], [0, 1, 0, 1], [0, 0, 1, 2], [0, 0, 0, 3]])
      iex> Nx.LinAlg.triangular_solve(a, Nx.tensor([2, 4, 2, 4]), lower: false)
      #Nx.Tensor<
        f32[4]
        [-1.3333333730697632, 2.6666667461395264, -0.6666666865348816, 1.3333333730697632]
      >

      iex> a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 2, 1]])
      iex> b = Nx.tensor([[0, 2, 1], [1, 1, 0], [3, 3, 1]])
      iex> Nx.LinAlg.triangular_solve(a, b, left_side: false)
      #Nx.Tensor<
        f32[3][3]
        [
          [-1.0, 0.0, 1.0],
          [0.0, 1.0, 0.0],
          [1.0, 1.0, 1.0]
        ]
      >

      iex> a = Nx.tensor([[1, 1, 1], [0, 1, 1], [0, 0, 1]], type: {:f, 64})
      iex> Nx.LinAlg.triangular_solve(a, Nx.tensor([1, 2, 1]), transform_a: :transpose, lower: false)
      #Nx.Tensor<
        f64[3]
        [1.0, 1.0, -1.0]
      >

      iex> a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], type: {:f, 64})
      iex> Nx.LinAlg.triangular_solve(a, Nx.tensor([1, 2, 1]), transform_a: :none)
      #Nx.Tensor<
        f64[3]
        [1.0, 1.0, -1.0]
      >

      iex> a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 2, 1]])
      iex> b = Nx.tensor([[0, 1, 3], [2, 1, 3]])
      iex> Nx.LinAlg.triangular_solve(a, b, left_side: false)
      #Nx.Tensor<
        f32[2][3]
        [
          [2.0, -5.0, 3.0],
          [4.0, -5.0, 3.0]
        ]
      >

      iex> a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 2, 1]])
      iex> b = Nx.tensor([[0, 2], [3, 0], [0, 0]])
      iex> Nx.LinAlg.triangular_solve(a, b, left_side: true)
      #Nx.Tensor<
        f32[3][2]
        [
          [0.0, 2.0],
          [3.0, -2.0],
          [-6.0, 2.0]
        ]
      >

      iex> a = Nx.tensor([
      ...> [1, 0, 0],
      ...> [1, Complex.new(0, 1), 0],
      ...> [Complex.new(0, 1), 1, 1]
      ...>])
      iex> b = Nx.tensor([1, -1, Complex.new(3, 3)])
      iex> Nx.LinAlg.triangular_solve(a, b)
      #Nx.Tensor<
        c64[3]
        [1.0+0.0i, 0.0+2.0i, 3.0+0.0i]
      >

  ### Error cases

      iex> Nx.LinAlg.triangular_solve(Nx.tensor([[3, 0, 0, 0], [2, 1, 0, 0]]), Nx.tensor([4, 2, 4, 2]))
      ** (ArgumentError) triangular_solve/3 expected a square tensor, got tensor with shape: {2, 4}

      iex> Nx.LinAlg.triangular_solve(Nx.tensor([[3, 0, 0, 0], [2, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]]), Nx.tensor([4]))
      ** (ArgumentError) incompatible dimensions for a and b on triangular solve

      iex> Nx.LinAlg.triangular_solve(Nx.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]), Nx.tensor([4, 2, 4, 2]))
      ** (ArgumentError) can't solve for singular matrix

      iex> a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], type: {:f, 64})
      iex> Nx.LinAlg.triangular_solve(a, Nx.tensor([1, 2, 1]), transform_a: :conjugate)
      ** (ArgumentError) complex numbers not supported yet

      iex> a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], type: {:f, 64})
      iex> Nx.LinAlg.triangular_solve(a, Nx.tensor([1, 2, 1]), transform_a: :other)
      ** (ArgumentError) invalid value for :transform_a option, expected :none, :transpose, or :conjugate, got: :other

  """
  def triangular_solve(a, b, opts \\ []) do
    opts = keyword!(opts, lower: true, left_side: true, transform_a: :none)
    output_type = binary_type(a, b) |> Nx.Type.to_floating()
    %T{shape: a_shape = {m, _}} = a = Nx.to_tensor(a)
    %T{shape: b_shape} = b = Nx.to_tensor(b)

    case opts[:transform_a] do
      t when t in [:none, :transpose] ->
        nil

      :conjugate ->
        raise ArgumentError, "complex numbers not supported yet"

      t ->
        raise ArgumentError,
              "invalid value for :transform_a option, expected :none, :transpose, or :conjugate, " <>
                "got: #{inspect(t)}"
    end

    case a_shape do
      {n, n} ->
        nil

      other ->
        raise ArgumentError,
              "triangular_solve/3 expected a square tensor, got tensor with shape: #{inspect(other)}"
    end

    left_side = opts[:left_side]

    case b_shape do
      {^m, _} when left_side ->
        nil

      {_, ^m} when not left_side ->
        nil

      {^m} ->
        nil

      _ ->
        raise ArgumentError, "incompatible dimensions for a and b on triangular solve"
    end

    impl!(a, b).triangular_solve(%{b | type: output_type}, a, b, opts)
  end

  @doc """
  Solves the system `AX = B`.

  `A` must have shape `{n, n}` and `B` must have shape `{n, m}` or `{n}`.
  `X` has the same shape as `B`.

  ## Examples

      iex> a = Nx.tensor([[1, 3, 2, 1], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
      iex> Nx.LinAlg.solve(a, Nx.tensor([-3, 0, 4, -2])) |> Nx.round()
      #Nx.Tensor<
        f32[4]
        [1.0, -2.0, 3.0, -4.0]
      >

      iex> a = Nx.tensor([[1, 0, 1], [1, 1, 0], [1, 1, 1]], type: {:f, 64})
      iex> Nx.LinAlg.solve(a, Nx.tensor([0, 2, 1])) |> Nx.round()
      #Nx.Tensor<
        f64[3]
        [1.0, 1.0, -1.0]
      >

      iex> a = Nx.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
      iex> b = Nx.tensor([[2, 2, 3], [2, 2, 4], [2, 0, 1]])
      iex> Nx.LinAlg.solve(a, b) |> Nx.round()
      #Nx.Tensor<
        f32[3][3]
        [
          [1.0, 2.0, 3.0],
          [1.0, 0.0, 1.0],
          [1.0, 0.0, 0.0]
        ]
      >

  If the axes are named, their names are not preserved in the output:

      iex> a = Nx.tensor([[1, 0, 1], [1, 1, 0], [1, 1, 1]], names: [:x, :y])
      iex> Nx.LinAlg.solve(a, Nx.tensor([0, 2, 1], names: [:z])) |> Nx.round()
      #Nx.Tensor<
        f32[3]
        [1.0, 1.0, -1.0]
      >

  ### Error cases

      iex> Nx.LinAlg.solve(Nx.tensor([[1, 0], [0, 1]]), Nx.tensor([4, 2, 4, 2]))
      ** (ArgumentError) `b` tensor has incompatible dimensions, expected {2, 2} or {2}, got: {4}

      iex> Nx.LinAlg.solve(Nx.tensor([[3, 0, 0, 0], [2, 1, 0, 0], [1, 1, 1, 1]]), Nx.tensor([4]))
      ** (ArgumentError) `a` tensor has incompatible dimensions, expected a 2-D tensor with as many rows as columns, got: {3, 4}
  """
  # IMPORTANT: This function cannot be a defn because
  # optional needs to work on the actual backend.
  @doc from_backend: false
  def solve(a, b) do
    %T{shape: a_shape, type: a_type} = a = Nx.to_tensor(a)
    %T{shape: b_shape, type: b_type} = b = Nx.to_tensor(b)

    output_shape = Nx.Shape.solve(a_shape, b_shape)
    output_type = a_type |> Nx.Type.merge(b_type) |> Nx.Type.to_floating()
    output = Nx.template(output_shape, output_type)

    Nx.Shared.optional(:solve, [a, b], output, fn a, b ->
      # Since we have triangular solve, which accepts upper
      # triangular matrices with the `lower: false` option,
      # we can solve a system as follows:

      # A.X = B -> QR.X = B -> R.X = adjoint(Q).B

      {q, r} = Nx.LinAlg.qr(a)

      triangular_solve(r, Nx.dot(adjoint(q), b), lower: false)
    end)
  end

  @doc """
  Inverts a square 2-D tensor.

  For non-square tensors, use `svd/2` for pseudo-inverse calculations.

  ## Examples

      iex> a = Nx.tensor([[1, 2, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0 , 0, 0, 1]])
      iex> a_inv = Nx.LinAlg.invert(a)
      #Nx.Tensor<
        f32[4][4]
        [
          [1.0, -2.0, -1.0, 2.0],
          [0.0, 1.0, 0.0, -1.0],
          [0.0, 0.0, 1.0, -1.0],
          [0.0, 0.0, 0.0, 1.0]
        ]
      >
      iex> Nx.dot(a, a_inv)
      #Nx.Tensor<
        f32[4][4]
        [
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 1.0]
        ]
      >
      iex> Nx.dot(a_inv, a)
      #Nx.Tensor<
        f32[4][4]
        [
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 1.0]
        ]
      >

  ### Error cases

      iex> Nx.LinAlg.invert(Nx.tensor([[3, 0, 0, 0], [2, 1, 0, 0]]))
      ** (ArgumentError) invert/1 expects a square tensor, got tensor with shape: {2, 4}

      iex> Nx.LinAlg.invert(Nx.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]))
      ** (ArgumentError) can't solve for singular matrix

  """
  @doc from_backend: false
  defn invert(tensor) do
    case Nx.shape(tensor) do
      {n, n} ->
        :ok

      shape ->
        raise ArgumentError,
              "invert/1 expects a square tensor, got tensor with shape: #{inspect(shape)}"
    end

    tensor
    |> invert_tensor()
    |> custom_grad(fn ans, g ->
      # As defined in https://juliadiff.org/ChainRulesCore.jl/stable/maths/arrays.html#Matrix-inversion-2
      ans_h = adjoint(ans)

      [{tensor, ans_h |> Nx.negate() |> Nx.dot(g) |> Nx.dot(ans_h)}]
    end)
  end

  defnp invert_tensor(tensor) do
    identity = Nx.eye(tensor)
    Nx.LinAlg.solve(tensor, identity)
  end

  @doc """
  Calculates the QR decomposition of a tensor with shape `{..., M, N}`.

  ## Options

    * `:mode` - Can be one of `:reduced`, `:complete`. Defaults to `:reduced`
      For the following, `K = min(M, N)`

      * `:reduced` - returns `q` and `r` with shapes `{..., M, K}` and `{..., K, N}`
      * `:complete` - returns `q` and `r` with shapes `{..., M, M}` and `{..., M, N}`

    * `:eps` - Rounding error threshold that can be applied during the triangularization

  ## Examples

      iex> {q, r} = Nx.LinAlg.qr(Nx.tensor([[-3, 2, 1], [0, 1, 1], [0, 0, -1]]))
      iex> q
      #Nx.Tensor<
        f32[3][3]
        [
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]
        ]
      >
      iex> r
      #Nx.Tensor<
        f32[3][3]
        [
          [-3.0, 2.0, 1.0],
          [0.0, 1.0, 1.0],
          [0.0, 0.0, -1.0]
        ]
      >

      iex> t = Nx.tensor([[3, 2, 1], [0, 1, 1], [0, 0, 1]])
      iex> {q, r} = Nx.LinAlg.qr(t)
      iex> q
      #Nx.Tensor<
        f32[3][3]
        [
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]
        ]
      >
      iex> r
      #Nx.Tensor<
        f32[3][3]
        [
          [3.0, 2.0, 1.0],
          [0.0, 1.0, 1.0],
          [0.0, 0.0, 1.0]
        ]
      >

      iex> {qs, rs} = Nx.LinAlg.qr(Nx.tensor([[[-3, 2, 1], [0, 1, 1], [0, 0, -1]],[[3, 2, 1], [0, 1, 1], [0, 0, 1]]]))
      iex> qs
      #Nx.Tensor<
        f32[2][3][3]
        [
          [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
          ],
          [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
          ]
        ]
      >
      iex> rs
      #Nx.Tensor<
        f32[2][3][3]
        [
          [
            [-3.0, 2.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, -1.0]
          ],
          [
            [3.0, 2.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0]
          ]
        ]
      >

      iex> t = Nx.tensor([[3, 2, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1]], type: {:f, 32})
      iex> {q, r} = Nx.LinAlg.qr(t, mode: :reduced)
      iex> q
      #Nx.Tensor<
        f32[4][3]
        [
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 0.7071067690849304],
          [0.0, 0.0, 0.7071067690849304]
        ]
      >
      iex> r
      #Nx.Tensor<
        f32[3][3]
        [
          [3.0, 2.0, 1.0],
          [0.0, 1.0, 1.0],
          [0.0, 0.0, 1.4142135381698608]
        ]
      >

      iex> t = Nx.tensor([[3, 2, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]], type: {:f, 32})
      iex> {q, r} = Nx.LinAlg.qr(t, mode: :complete)
      iex> q
      #Nx.Tensor<
        f32[4][4]
        [
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 1.0]
        ]
      >
      iex> r
      #Nx.Tensor<
        f32[4][3]
        [
          [3.0, 2.0, 1.0],
          [0.0, 1.0, 1.0],
          [0.0, 0.0, 1.0],
          [0.0, 0.0, 0.0]
        ]
      >

  ## Error cases

      iex> Nx.LinAlg.qr(Nx.tensor([[[1, 1, 1, 1], [-1, 4, 4, -1], [4, -2, 2, 0]]]))
      ** (ArgumentError) tensor must have at least as many rows as columns in the last two axes, got 3 rows and 4 columns

      iex> Nx.LinAlg.qr(Nx.tensor([1, 2, 3, 4, 5]))
      ** (ArgumentError) tensor must have at least rank 2, got rank 1 with shape {5}
  """
  def qr(tensor, opts \\ []) do
    opts = keyword!(opts, mode: :reduced, eps: @default_eps)
    %T{type: type, shape: shape} = tensor = Nx.to_tensor(tensor)

    mode = opts[:mode]
    valid_modes = [:reduced, :complete]

    unless mode in valid_modes do
      raise ArgumentError,
            "invalid :mode received. Expected one of #{valid_modes}, received: #{mode}"
    end

    output_type = Nx.Type.to_floating(type)
    {q_shape, r_shape} = Nx.Shape.qr(shape, opts)

    impl!(tensor).qr(
      {%{
         tensor
         | type: output_type,
           shape: q_shape,
           names: List.duplicate(nil, tuple_size(q_shape))
       },
       %{
         tensor
         | type: output_type,
           shape: r_shape,
           names: List.duplicate(nil, tuple_size(r_shape))
       }},
      tensor,
      opts
    )
  end

  @doc """
  Calculates the Eigenvalues and Eigenvectors of symmetric 2-D tensors.

  It returns `{eigenvals, eigenvecs}`.

  ## Options

    * `:max_iter` - `integer`. Defaults to `50_000`
      Number of maximum iterations before stopping the decomposition

    * `:eps` - `float`. Defaults to 1.0e-10
      Tolerance applied during the decomposition

  Note not all options apply to all backends, as backends may have
  specific optimizations that render these mechanisms unnecessary.

  ## Examples

      iex> {eigenvals, eigenvecs} = Nx.LinAlg.eigh(Nx.tensor([[1, 0], [0, 2]]))
      iex> Nx.round(eigenvals)
      #Nx.Tensor<
        f32[2]
        [1.0, 2.0]
      >
      iex> eigenvecs
      #Nx.Tensor<
        f32[2][2]
        [
          [1.0, 0.0],
          [0.0, 1.0]
        ]
      >

      iex> {eigenvals, eigenvecs} = Nx.LinAlg.eigh(Nx.tensor([[0, 1, 2], [1, 0, 2], [2, 2, 3]]))
      iex> Nx.round(eigenvals)
      #Nx.Tensor<
        f32[3]
        [5.0, -1.0, -1.0]
      >
      iex> eigenvecs
      #Nx.Tensor<
        f32[3][3]
        [
          [0.4082472324371338, 0.9128734469413757, 0.0],
          [0.40824851393699646, -0.18257413804531097, 0.8944271802902222],
          [0.8164970278739929, -0.36514827609062195, -0.4472135901451111]
        ]
      >

  ## Error cases

      iex> Nx.LinAlg.eigh(Nx.tensor([[1, 2, 3], [4, 5, 6]]))
      ** (ArgumentError) eigh/2 expects a square tensor, got tensor with shape: {2, 3}

      iex> Nx.LinAlg.eigh(Nx.tensor([[1, 2], [3, 4]]))
      ** (ArgumentError) input tensor must be symmetric
  """
  def eigh(tensor, opts \\ []) do
    opts = keyword!(opts, max_iter: 50_000, eps: @default_eps)
    %T{type: type, shape: shape} = tensor = Nx.to_tensor(tensor)

    Nx.Shared.raise_complex_not_implemented_yet(type, "LinAlg.eigh", 2)

    output_type = Nx.Type.to_floating(type)

    {eigenvals_shape, eigenvecs_shape} =
      case shape do
        {n, n} ->
          {{n}, {n, n}}

        shape ->
          raise ArgumentError,
                "eigh/2 expects a square tensor, got tensor with shape: #{inspect(shape)}"
      end

    impl!(tensor).eigh(
      {%{tensor | names: [nil], type: output_type, shape: eigenvals_shape},
       %{tensor | names: [nil, nil], type: output_type, shape: eigenvecs_shape}},
      tensor,
      opts
    )
  end

  @doc """
  Calculates the Singular Value Decomposition of 2-D tensors.

  It returns `{u, s, vt}` where the elements of `s` are sorted
  from highest to lowest.

  ## Options

    * `:max_iter` - `integer`. Defaults to `1000`
      Number of maximum iterations before stopping the decomposition

    * `:eps` - `float`. Defaults to 1.0e-12
      Tolerance applied during the decomposition

  Note not all options apply to all backends, as backends may have
  specific optimizations that render these mechanisms unnecessary.

  ## Examples

      iex> {u, s, v} = Nx.LinAlg.svd(Nx.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
      iex> u
      #Nx.Tensor<
        f32[3][3]
        [
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]
        ]
      >
      iex> s
      #Nx.Tensor<
        f32[3]
        [1.0, 1.0, 1.0]
      >
      iex> v
      #Nx.Tensor<
        f32[3][3]
        [
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, -1.0]
        ]
      >

      iex> {u, s, vt} = Nx.LinAlg.svd(Nx.tensor([[2, 0, 0], [0, 3, 0], [0, 0, -1], [0, 0, 0]]))
      iex> u
      #Nx.Tensor<
        f32[4][4]
        [
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 1.0]
        ]
      >
      iex> s
      #Nx.Tensor<
        f32[3]
        [3.0, 2.0, 1.0]
      >
      iex> vt
      #Nx.Tensor<
        f32[3][3]
        [
          [0.0, 1.0, 0.0],
          [1.0, 0.0, 0.0],
          [0.0, 0.0, -1.0]
        ]
      >

  """
  def svd(tensor, opts \\ []) do
    opts = keyword!(opts, [:max_iter, eps: @default_eps])
    %T{type: type, shape: shape} = tensor = Nx.to_tensor(tensor)

    Nx.Shared.raise_complex_not_implemented_yet(type, "LinAlg.svd", 2)

    output_type = Nx.Type.to_floating(type)
    {u_shape, s_shape, v_shape} = Nx.Shape.svd(shape)

    impl!(tensor).svd(
      {%{tensor | names: [nil, nil], type: output_type, shape: u_shape},
       %{tensor | names: [nil], type: output_type, shape: s_shape},
       %{tensor | names: [nil, nil], type: output_type, shape: v_shape}},
      tensor,
      opts
    )
  end

  @doc """
  Calculates the A = PLU decomposition of a 2-D tensor A with shape `{N, N}`.

  ## Options

    * `:eps` - Rounding error threshold that can be applied during the factorization

  ## Examples

      iex> {p, l, u} = Nx.LinAlg.lu(Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
      iex> p
      #Nx.Tensor<
        s64[3][3]
        [
          [0, 0, 1],
          [0, 1, 0],
          [1, 0, 0]
        ]
      >
      iex> l
      #Nx.Tensor<
        f32[3][3]
        [
          [1.0, 0.0, 0.0],
          [0.5714285969734192, 1.0, 0.0],
          [0.1428571492433548, 2.0, 1.0]
        ]
      >
      iex> u
      #Nx.Tensor<
        f32[3][3]
        [
          [7.0, 8.0, 9.0],
          [0.0, 0.4285714328289032, 0.8571428656578064],
          [0.0, 0.0, 0.0]
        ]
      >
      iex> p |> Nx.dot(l) |> Nx.dot(u)
      #Nx.Tensor<
        f32[3][3]
        [
          [1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0],
          [7.0, 8.0, 9.0]
        ]
      >

      iex> {p, l, u} = Nx.LinAlg.lu(Nx.tensor([[1, 0, 1], [-1, 0, -1], [1, 1, 1]]))
      iex> p
      #Nx.Tensor<
        s64[3][3]
        [
          [1, 0, 0],
          [0, 0, 1],
          [0, 1, 0]
        ]
      >
      iex> l
      #Nx.Tensor<
        f32[3][3]
        [
          [1.0, 0.0, 0.0],
          [1.0, 1.0, 0.0],
          [-1.0, 0.0, 1.0]
        ]
      >
      iex> u
      #Nx.Tensor<
        f32[3][3]
        [
          [1.0, 0.0, 1.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0]
        ]
      >
      iex> p |> Nx.dot(l) |> Nx.dot(u)
      #Nx.Tensor<
        f32[3][3]
        [
          [1.0, 0.0, 1.0],
          [-1.0, 0.0, -1.0],
          [1.0, 1.0, 1.0]
        ]
      >

  ## Error cases

      iex> Nx.LinAlg.lu(Nx.tensor([[1, 1, 1, 1], [-1, 4, 4, -1], [4, -2, 2, 0]]))
      ** (ArgumentError) tensor must have as many rows as columns, got shape: {3, 4}
  """
  def lu(tensor, opts \\ []) do
    opts = keyword!(opts, eps: @default_eps)
    %T{type: type, shape: shape} = tensor = Nx.to_tensor(tensor)

    output_type = Nx.Type.to_floating(type)
    {p_shape, l_shape, u_shape} = Nx.Shape.lu(shape)
    names = [nil, nil]

    impl!(tensor).lu(
      {%{tensor | type: type, shape: p_shape, names: names},
       %{tensor | type: output_type, shape: l_shape, names: names},
       %{tensor | type: output_type, shape: u_shape, names: names}},
      tensor,
      opts
    )
  end

  @doc """
  Produces the tensor taken to the given power by dot-product.

  The input is always a square tensor and a non-negative integer,
  and the output is a square tensor of the same dimensions as the input tensor.

  The dot-products are unrolled inside `defn`.

  ## Examples

      iex> Nx.LinAlg.matrix_power(Nx.tensor([[1, 2], [3, 4]]), 0)
      #Nx.Tensor<
        s64[2][2]
        [
          [1, 0],
          [0, 1]
        ]
      >

      iex> Nx.LinAlg.matrix_power(Nx.tensor([[1, 2], [3, 4]]), 6)
      #Nx.Tensor<
        s64[2][2]
        [
          [5743, 8370],
          [12555, 18298]
        ]
      >

      iex> Nx.LinAlg.matrix_power(Nx.eye(3), 65535)
      #Nx.Tensor<
        s64[3][3]
        [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]
        ]
      >

      iex> Nx.LinAlg.matrix_power(Nx.tensor([[1, 2], [3, 4]]), -1)
      #Nx.Tensor<
        f32[2][2]
        [
          [-2.0, 1.0],
          [1.5, -0.5]
        ]
      >

      iex> Nx.LinAlg.matrix_power(Nx.tensor([[1, 2], [3, 4], [5, 6]]), 1)
      ** (ArgumentError) matrix_power/2 expects a square tensor, got tensor with shape: {3, 2}
  """
  @doc from_backend: false
  def matrix_power(tensor, power) when is_integer(power) and power < 0 do
    matrix_power(invert(tensor), abs(power))
  end

  # We need a special-case for 0 since the code below
  # is optimized to not compute an initial eye.
  def matrix_power(tensor, 0) do
    case Nx.shape(tensor) do
      {n, n} ->
        :ok

      shape ->
        raise ArgumentError,
              "matrix_power/2 expects a square tensor, got tensor with shape: #{inspect(shape)}"
    end

    Nx.eye(tensor)
  end

  def matrix_power(tensor, power) when is_integer(power) do
    case Nx.shape(tensor) do
      {n, n} ->
        :ok

      shape ->
        raise ArgumentError,
              "matrix_power/2 expects a square tensor, got tensor with shape: #{inspect(shape)}"
    end

    power
    |> Integer.digits(2)
    |> tl()
    |> Enum.reverse()
    |> Enum.reduce({nil, tensor}, fn
      1, {nil, exp_tensor} ->
        {exp_tensor, Nx.dot(exp_tensor, exp_tensor)}

      1, {result_tensor, exp_tensor} ->
        {Nx.dot(result_tensor, exp_tensor), Nx.dot(exp_tensor, exp_tensor)}

      0, {result_tensor, exp_tensor} ->
        {result_tensor, Nx.dot(exp_tensor, exp_tensor)}
    end)
    |> then(fn
      {nil, exp_tensor} -> exp_tensor
      {result, exp_tensor} -> Nx.dot(result, exp_tensor)
    end)
  end

  @doc """
  Calculates the determinant of a square 2D tensor.

  ### Examples

  For 2x2 and 3x3, the results are given by the closed formulas:

      iex> Nx.LinAlg.determinant(Nx.tensor([[1, 2], [3, 4]]))
      #Nx.Tensor<
        f32
        -2.0
      >

      iex> Nx.LinAlg.determinant(Nx.tensor([[1.0, 2.0, 3.0], [1.0, -2.0, 3.0], [7.0, 8.0, 9.0]]))
      #Nx.Tensor<
        f32
        48.0
      >

  When there are linearly dependent rows or columns, the determinant is 0:

      iex> Nx.LinAlg.determinant(Nx.tensor([[1.0, 0.0], [3.0, 0.0]]))
      #Nx.Tensor<
        f32
        0.0
      >

      iex> Nx.LinAlg.determinant(Nx.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]]))
      #Nx.Tensor<
        f32
        0.0
      >

  The determinant can also be calculated when the axes are bigger than 3:

      iex> Nx.LinAlg.determinant(Nx.tensor([
      ...> [1, 0, 0, 0],
      ...> [0, 1, 2, 3],
      ...> [0, 1, -2, 3],
      ...> [0, 7, 8, 9.0]
      ...> ]))
      #Nx.Tensor<
        f32
        48.0
      >

      iex> Nx.LinAlg.determinant(Nx.tensor([
      ...> [0, 0, 0, 0, -1],
      ...> [0, 1, 2, 3, 0],
      ...> [0, 1, -2, 3, 0],
      ...> [0, 7, 8, 9, 0],
      ...> [1, 0, 0, 0, 0]
      ...> ]))
      #Nx.Tensor<
        f32
        48.0
      >

  If the axes are named, their names are not preserved in the output:

      iex> two_by_two = Nx.tensor([[1, 2], [3, 4]], names: [:x, :y])
      iex> Nx.LinAlg.determinant(two_by_two)
      #Nx.Tensor<
        f32
        -2.0
      >

      iex> three_by_three = Nx.tensor([[1.0, 2.0, 3.0], [1.0, -2.0, 3.0], [7.0, 8.0, 9.0]], names: [:x, :y])
      iex> Nx.LinAlg.determinant(three_by_three)
      #Nx.Tensor<
        f32
        48.0
      >

  Also supports complex inputs:

      iex> t = Nx.tensor([[1, 0, 0], [0, Complex.new(0, 2), 0], [0, 0, 3]])
      iex> Nx.LinAlg.determinant(t)
      #Nx.Tensor<
        c64
        0.0+6.0i
      >

      iex> t = Nx.tensor([[0, 0, 0, 1], [0, Complex.new(0, 2), 0, 0], [0, 0, 3, 0], [1, 0, 0, 0]])
      iex> Nx.LinAlg.determinant(t)
      #Nx.Tensor<
        c64
        0.0-6.0i
      >

  """
  # IMPORTANT: This function cannot be a defn because
  # optional needs to work on the actual backend.
  def determinant(tensor) do
    tensor = Nx.to_tensor(tensor)
    output = Nx.template({}, Nx.Type.to_floating(tensor.type))

    case Nx.shape(tensor) do
      {n, n} ->
        :ok

      shape ->
        raise ArgumentError,
              "determinant/1 expects a square tensor, got tensor with shape: #{inspect(shape)}"
    end

    Nx.Shared.optional(:determinant, [tensor], output, fn tensor ->
      case Nx.shape(tensor) do
        {2, 2} ->
          determinant_2by2(tensor)

        {3, 3} ->
          determinant_3by3(tensor)

        {n, n} ->
          determinant_NbyN(tensor)
      end
    end)
  end

  defnp determinant_2by2(t) do
    t = Nx.tile(t, [1, 2])

    result = diagonal_product(t, 0) - diagonal_product(t, 1)

    # Ensure floating point result
    result * 1.0
  end

  defnp determinant_3by3(t) do
    pos_t = Nx.tile(t, [1, 2])

    neg_t = Nx.reverse(pos_t, axes: [1])

    result =
      diagonal_product(pos_t, 0) +
        diagonal_product(pos_t, 1) +
        diagonal_product(pos_t, 2) -
        diagonal_product(neg_t, 0) -
        diagonal_product(neg_t, 1) -
        diagonal_product(neg_t, 2)

    # Ensure floating point result
    result * 1.0
  end

  defnp determinant_NbyN(t) do
    nxn = {n, _} = Nx.shape(t)

    # Taken from slogdet at https://github.com/google/jax/blob/a3a6afcd5b8bf3d60aba94054bb0001c0fcc50d7/jax/_src/numpy/linalg.py#L134
    {p, l, u} = Nx.LinAlg.lu(t)

    diag = Nx.take_diagonal(l) * Nx.take_diagonal(u)
    is_zero = Nx.any(diag == 0)
    transitions = p |> Nx.real() |> Nx.dot(Nx.iota({n}))

    upper_tri_mask = Nx.iota(nxn, axis: 0) |> Nx.less(Nx.iota(nxn, axis: 1))

    parity =
      transitions
      |> Nx.broadcast(nxn, axes: [0])
      |> Nx.greater(transitions)
      |> Nx.multiply(upper_tri_mask)
      |> Nx.sum()

    sign =
      if is_zero do
        0
      else
        -2 * rem(parity, 2) + 1
      end

    if is_zero do
      0
    else
      sign * Nx.product(diag)
    end
  end

  defnp diagonal_product(t, offset) do
    t
    |> Nx.take_diagonal(offset: offset)
    |> Nx.product()
  end
end
