defmodule Nx.LinAlg do
  @moduledoc """
  Nx conveniences for linear algebra.
  """

  import Nx.Shared
  import Nx.Defn.Kernel, only: [keyword!: 2]

  alias Nx.Tensor, as: T

  @default_eps 1.0e-10

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
          [3.9354796409606934, 0.7155417203903198]
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
          [1.2247447967529297, 2.1213204860687256, 0.0, 0.0],
          [1.6329931020736694, 1.4142135381698608, 2.309401035308838, 0.0],
          [3.265986204147339, -1.4142132997512817, 1.5877132415771484, 3.13249135017395]
        ]
      >

  ### Error cases

      iex> Nx.LinAlg.cholesky(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]))
      ** (ArgumentError) matrix must be symmetric, a matrix is symmetric iff X = X.T
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

  ### Error cases

      iex> Nx.LinAlg.norm(Nx.tensor([3, 4]), ord: :frobenius)
      ** (ArgumentError) expected a 2-D tensor for ord: :frobenius, got a 1-D tensor
  """
  @doc from_backend: false
  def norm(tensor, opts \\ []) when is_list(opts) do
    %{shape: s} = t = Nx.to_tensor(tensor)
    opts = keyword!(opts, [:ord, :axes])
    rank = Nx.rank(t)

    unless rank == 1 or rank == 2,
      do: raise(ArgumentError, "expected 1-D or 2-D tensor, got tensor with shape #{inspect(s)}")

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
    output_type = Nx.Type.to_floating(type)
    aggregate_axes = if tuple_size(shape) == 2, do: &Nx.sum(&1, axes: [1]), else: & &1
    reduce = if ord == :inf, do: &Nx.reduce_max/1, else: &Nx.reduce_min/1

    t
    |> Nx.abs()
    |> aggregate_axes.()
    |> reduce.()
    |> Nx.as_type(output_type)
  end

  defp norm_integer(%{shape: {_}, type: type} = t, 0, _opts) do
    output_type = Nx.Type.to_floating(type)

    t
    |> Nx.not_equal(0)
    |> Nx.sum()
    |> Nx.as_type(output_type)
  end

  defp norm_integer(%{shape: {_, _}, type: type} = t, ord, _opts) when ord in [1, -1] do
    output_type = Nx.Type.to_floating(type)
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
    output_type = Nx.Type.to_floating(type)
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

  ### Error cases

      iex> Nx.LinAlg.triangular_solve(Nx.tensor([[3, 0, 0, 0], [2, 1, 0, 0]]), Nx.tensor([4, 2, 4, 2]))
      ** (ArgumentError) expected a square matrix, got matrix with shape: {2, 4}

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
        raise ArgumentError, "expected a square matrix, got matrix with shape: #{inspect(other)}"
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

    ### Error cases

      iex> Nx.LinAlg.solve(Nx.tensor([[1, 0], [0, 1]]), Nx.tensor([4, 2, 4, 2]))
      ** (ArgumentError) `b` tensor has incompatible dimensions, expected {2, 2} or {2}, got: {4}

      iex> Nx.LinAlg.solve(Nx.tensor([[3, 0, 0, 0], [2, 1, 0, 0], [1, 1, 1, 1]]), Nx.tensor([4]))
      ** (ArgumentError) `a` tensor has incompatible dimensions, expected a 2-D tensor with as many rows as columns, got: {3, 4}
  """
  @doc from_backend: false
  def solve(a, b) do
    %T{shape: a_shape} = a = Nx.to_tensor(a)
    %T{shape: b_shape} = b = Nx.to_tensor(b)

    Nx.Shape.solve(a_shape, b_shape)

    # We need to achieve an LQ decomposition for `a` (henceforth called A)
    # because triangular_solve only accepts lower_triangular matrices
    # Therefore, we can use the fact that if we have M = Q'R' -> transpose(M) = LQ.
    # If we set M = transpose(A), we can get A = LQ through transpose(qr(transpose(A)))

    # Given A = LQ, we can then solve AX = B as shown below
    # AX = B -> L(QX) = B -> LY = B, where Y = QX -> X = transpose(Q)Y

    # Finally, it can be shown that when B is a matrix, X can be
    # calculated by solving for each corresponding column in B
    {qt, r_prime} = a |> Nx.transpose() |> qr()
    l = Nx.transpose(r_prime)
    y = triangular_solve(l, b)
    Nx.dot(qt, y)
  end

  @doc """
  Inverts a square 2-D tensor.

  For non-square tensors, use `svd/2` for pseudo-inverse calculations.

  ## Examples

      iex> a = Nx.tensor([[1, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
      iex> a_inv = Nx.LinAlg.invert(a)
      #Nx.Tensor<
        f32[4][4]
        [
          [1.0, 0.0, 0.0, 0.0],
          [-2.0, 1.0, 0.0, 0.0],
          [-1.0, 0.0, 1.0, 0.0],
          [2.0, -1.0, -1.0, 1.0]
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
      ** (ArgumentError) can only invert square matrices, got: {2, 4}

      iex> Nx.LinAlg.invert(Nx.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]))
      ** (ArgumentError) can't solve for singular matrix

  """
  @doc from_backend: false
  def invert(tensor) do
    %T{shape: {m, n}} = tensor = Nx.to_tensor(tensor)

    if m != n do
      raise ArgumentError,
            "can only invert square matrices, got: {#{m}, #{n}}"
    end

    identity = Nx.eye(tensor)
    solve(tensor, identity)
  end

  @doc """
  Calculates the QR decomposition of a 2-D tensor with shape `{M, N}`.

  ## Options

    * `:mode` - Can be one of `:reduced`, `:complete`. Defaults to `:reduced`
      For the following, `K = min(M, N)`

      * `:reduced` - returns `q` and `r` with shapes `{M, K}` and `{K, N}`
      * `:complete` - returns `q` and `r` with shapes `{M, M}` and `{M, N}`

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

      iex> t = Nx.tensor([[3, 2, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1]], type: {:f, 32})
      iex> {q, r} = Nx.LinAlg.qr(t, mode: :reduced)
      iex> q
      #Nx.Tensor<
        f32[4][3]
        [
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0],
          [0.0, 0.0, 0.0]
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

      iex> Nx.LinAlg.qr(Nx.tensor([[1, 1, 1, 1], [-1, 4, 4, -1], [4, -2, 2, 0]]))
      ** (ArgumentError) tensor must have at least as many rows as columns, got shape: {3, 4}
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
      {%{tensor | type: output_type, shape: q_shape, names: [nil, nil]},
       %{tensor | type: output_type, shape: r_shape, names: [nil, nil]}},
      tensor,
      opts
    )
  end

  @doc """
  Calculates the Singular Value Decomposition of 2-D tensors.

  It returns `{u, s, vt}` where the elemens of `s` are sorted
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
end
