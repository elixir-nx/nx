defmodule Nx.Random do
  @moduledoc """
  Pseudo-random number generators.

  Unlike the stateful pseudo-random number generators (PRNGs)
  that users of most programming languages and numerical libraries
  may be accustomed to, Nx random functions require an explicit
  PRNG key to be passed as a first argument (see [below](#module-the-key-to-understanding-nx-random-keys) for more info). That key is defined by
  an `Nx.Tensor` composed of 2 unsigned 32-bit integers, usually
  generated by the `Nx.Random.key/1` function:

      iex> Nx.Random.key(12)
      #Nx.Tensor<
        u32[2]
        [0, 12]
      >

  Or for example:

      iex> Nx.Random.key(System.os_time())

  This key can then be used in any of Nx’s random number generation
  routines:

      iex> key = Nx.Random.key(12)
      iex> {uniform, _new_key} = Nx.Random.uniform(key)
      iex> uniform
      #Nx.Tensor<
        f32
        0.7691127061843872
      >

  Now, when generating a new random number, you pass the `new_key`
  to get a different number.

  The function in this module also have a `*_split` variant, which
  is used when the key has been split before hand.

  ## Design and Context

  In short, Nx's PRNGs are based on a Threefry counter PRNG
  associated to a functional array-oriented splitting model.
  To summarize, among other requirements, Nx's PRNG aims to:

  1. Ensure reproducibility

  2. Parallelize well, both in terms of vectorization
     (generating array values) and multi-replica, multi-core
     computation. In particular it should not use sequencing
     constraints between random function calls.

  > ### The key to understanding Nx.Random keys {: .info}
  >
  > Most Elixir users might be used to not having to keep
  > track of the PRNG state while their code executes.
  >
  > While this works fine when we're dealing with the CPU,
  > we can think of keeping track of the `Nx.Random` key as a way to
  > isolate multiple GPU users, much like the PRNG on different
  > BEAM nodes is isolated. Each key gets updated in its own
  > isolated sequence of calls, and thus we don't get different
  > results for each process using the same PRNG as we would
  > in the normal situation.
  >
  > The fact that the key is a parameter for the functions also
  > helps with the caching and operator fusion of the computational
  > graphs. Because the PRNG functions themselves are stateless,
  > compilers can take advantage of this to further improve execution times.

  """

  import Nx.Defn, only: [deftransform: 2, deftransformp: 2, defn: 2, defnp: 2]

  @nbits 32

  @doc """
  Create a pseudo-random number generator (PRNG) key given an integer seed.

  ## Examples

      iex> Nx.Random.key(12)
      #Nx.Tensor<
        u32[2]
        [0, 12]
      >

      iex> Nx.Random.key(999999999999)
      #Nx.Tensor<
        u32[2]
        [232, 3567587327]
      >
  """
  deftransform key(seed) do
    seed =
      case seed do
        seed when is_integer(seed) ->
          Nx.u64(seed)

        %Nx.Tensor{} = seed when seed.type == {:u, 64} ->
          seed

        %Nx.Tensor{} = seed when seed.type == {:s, 64} ->
          Nx.bitcast(seed, {:u, 64})

        other ->
          raise ArgumentError,
                "expected seed to be an integer, u64 tensor or s64 tensor, got: #{inspect(other)}"
      end

    key_n(seed)
  end

  defnp key_n(seed) do
    k1 = Nx.right_shift(seed, 32)
    k2 = Nx.bitwise_and(seed, Nx.u64(0xFFFFFFFF))

    Nx.stack([k1, k2])
    |> Nx.as_type(:u32)
    |> stop_grad()
  end

  @doc """
  Splits a PRNG key into `num` new keys by adding a leading axis.

  ## Examples

      iex> key = Nx.Random.key(1701)
      iex> Nx.Random.split(key)
      #Nx.Tensor<
        u32[2][2]
        [
          [56197195, 1801093307],
          [961309823, 1704866707]
        ]
      >

      iex> key = Nx.Random.key(1701)
      iex> Nx.Random.split(key, parts: 4)
      #Nx.Tensor<
        u32[4][2]
        [
          [4000152724, 2030591747],
          [2287976877, 2598630646],
          [2426625787, 580268518],
          [3136765380, 433355682]
        ]
      >
  """
  defn split(key, opts \\ []) do
    assert_key!(key)
    opts = keyword!(opts, parts: 2)
    threefry2x32(key, {opts[:parts], 2}) |> stop_grad()
  end

  @doc """
  Folds in new data to a PRNG key.

  ## Examples

      iex> key = Nx.Random.key(42)
      iex> Nx.Random.fold_in(key, 99)
      #Nx.Tensor<
        u32[2]
        [2015327502, 1351855566]
      >

      iex> key = Nx.Random.key(42)
      iex> Nx.Random.fold_in(key, 1234)
      #Nx.Tensor<
        u32[2]
        [1356445167, 2917756949]
      >

      iex> key = Nx.Random.key(42)
      iex> Nx.Random.fold_in(key, Nx.tensor([[1, 99], [1234, 13]]))
      #Nx.Tensor<
        u32[2][2][2]
        [
          [
            [64467757, 2916123636],
            [2015327502, 1351855566]
          ],
          [
            [1356445167, 2917756949],
            [3514951389, 229662949]
          ]
        ]
      >
  """
  defn fold_in(key, data) do
    assert_key!(key)

    data = Nx.as_type(data, :u64)

    k1 = Nx.right_shift(data, 32)
    k2 = Nx.bitwise_and(data, Nx.u64(0xFFFFFFFF))

    {x1, x2} =
      Nx.stack([k1, k2])
      |> Nx.reshape({2, :auto})
      |> Nx.as_type(:u32)
      |> threefry2x32_20_pair(key)

    [x1, x2]
    |> Nx.stack(axis: -1)
    |> Nx.reshape(fold_shape(Nx.shape(data)))
    |> stop_grad()
  end

  deftransformp fold_shape(shape) do
    Tuple.insert_at(shape, tuple_size(shape), 2)
  end

  defnp threefry2x32(key, shape) do
    case shape |> Nx.size() |> rem(2) do
      0 ->
        Nx.iota({2, div(Nx.size(shape), 2)}, type: :u32)
        |> threefry2x32_20_concat(key)
        |> Nx.reshape(shape)

      1 ->
        Nx.concatenate([Nx.iota({Nx.size(shape)}, type: :u32), Nx.tensor([0], type: :u32)])
        |> Nx.reshape({2, :auto})
        |> threefry2x32_20_concat(key)
        |> Access.get(0..-2//1)
        |> Nx.reshape(shape)
    end
  end

  defn threefry2x32_20_concat(xs, ks) do
    {nx1, nx2} = threefry2x32_20_pair(xs, ks)
    Nx.concatenate([nx1, nx2], axis: 0)
  end

  defnp threefry2x32_20_pair(xs, ks) do
    rotations = {Nx.tensor([13, 15, 26, 6], type: :u8), Nx.tensor([17, 29, 16, 24], type: :u8)}

    key1 = ks[0]
    key2 = ks[1]
    xs = {xs[0] + key1, xs[1] + key2}

    ks = {
      key2,
      Nx.bitwise_xor(key1, key2)
      |> Nx.bitwise_xor(0x1BD11BDA),
      key1
    }

    state = {xs, ks, rotations}

    {_, {{nx1, nx2}, _, _}} =
      while {x = Nx.tensor(1, type: :u32), state}, x < 6 do
        {x + Nx.tensor(1, type: :u32), rolled_loop_step(x, state)}
      end

    {nx1, nx2}
  end

  defnp apply_round({xs1, xs2}, rot) do
    y1 = xs1 + xs2

    y2 =
      rotate_left(xs2, rot)
      |> Nx.bitwise_xor(y1)

    {y1, y2}
  end

  defnp rolled_loop_step(i, {{_xs1, _xs2} = xs, {k1, k2, k3}, {r1, r2}}) do
    {xs1, xs2} =
      while xs, r <- r1 do
        apply_round(xs, r)
      end

    xs1 = k1 + xs1
    xs2 = k2 + i + xs2

    new_xs = {xs1, xs2}
    new_ks = {k2, k3, k1}
    new_rs = {r2, r1}

    {new_xs, new_ks, new_rs}
  end

  defnp rotate_left(x, rot) do
    x <<< rot ||| x >>> (Nx.tensor(@nbits, type: :u32) - rot)
  end

  defnp random_bits(key, opts \\ []) do
    assert_key!(key)
    opts = keyword!(opts, shape: {}, bit_width: 32)
    shape = opts[:shape]
    bit_width = opts[:bit_width]

    case bit_width do
      64 ->
        bits =
          threefry2x32(key, {2, Nx.size(shape)})
          |> Nx.as_type({:u, 64})

        bits = bits[0] <<< 32 ||| bits[1]
        Nx.reshape(bits, shape)

      32 ->
        threefry2x32(key, shape)

      _ ->
        threefry2x32(key, shape)
        |> Nx.as_type({:u, bit_width})
    end
    |> stop_grad()
  end

  deftransformp mantissa_shift(nbits, type) do
    mantissa =
      case type do
        {:f, 8} -> 2
        {:bf, 16} -> 7
        {:f, 16} -> 10
        {:f, 32} -> 23
        {:f, 64} -> 52
      end

    Nx.tensor(nbits - mantissa, type: {:u, nbits})
  end

  @doc """
  Sample uniform random integer values in the semi-open open interval `[min_value, max_value)`.

  ## Options

    * `:type` - the integer type for the returned tensor
    * `:shape` - shape of the returned tensor
    * `:names` - the names of the returned tensor

  ## Examples

      iex> key = Nx.Random.key(1701)
      iex> {randint, _new_key} = Nx.Random.randint(key, 1, 100)
      iex> randint
      #Nx.Tensor<
        s32
        91
      >

      iex> key = Nx.Random.key(1701)
      iex> {randint, _new_key} = Nx.Random.randint(key, 1, 100, shape: {3, 2}, type: :u32)
      iex> randint
      #Nx.Tensor<
        u32[3][2]
        [
          [9, 20],
          [19, 6],
          [71, 15]
        ]
      >

  """
  defn randint(key, min_val, max_val, opts \\ []) do
    keys = split(key)
    {randint_split(keys[1], min_val, max_val, opts), keys[0]}
  end

  @doc """
  Same as `randint/4` but assumes the key has already been split.
  """
  defn randint_split(key, min_val, max_val, opts \\ []) do
    opts = keyword!(opts, [:names, :type, shape: {}])
    assert_key!(key)

    shape = opts[:shape]
    type = {_, nbits} = infer_type(min_val, max_val, opts)

    case type do
      {:u, _} -> :ok
      {:s, _} -> :ok
      _ -> raise ArgumentError, "expected integer type, got type #{inspect(type)}"
    end

    min_val = Nx.broadcast(min_val, shape)
    max_val = Nx.broadcast(max_val, shape)

    random_bits = random_bits(key, shape: randint_random_bits_shape(shape), bit_width: nbits)

    higher_bits = random_bits[0]
    lower_bits = random_bits[1]

    span = Nx.as_type(max_val - min_val, {:u, nbits})

    multiplier =
      Nx.pow(Nx.u64(2), Nx.quotient(nbits, 2))
      |> Nx.remainder(span)
      |> Nx.pow(2)
      |> Nx.remainder(span)

    offset =
      higher_bits
      |> Nx.remainder(span)
      |> Nx.multiply(multiplier)
      |> Nx.add(Nx.remainder(lower_bits, span))
      |> Nx.remainder(span)

    (min_val + offset)
    |> Nx.as_type(type)
    |> Nx.reshape(shape, take_names(opts))
    |> stop_grad()
  end

  deftransformp randint_random_bits_shape(shape), do: Tuple.insert_at(shape, 0, 2)

  @doc """
  Shortcut for `uniform(key, 0.0, 1.0, opts)`.
  """
  defn uniform(key, opts \\ []) do
    uniform(key, 0.0, 1.0, opts)
  end

  @doc """
  Sample uniform float values in the semi-open interval `[min_val, max_val)`.

  ## Options

    * `:type` - a float type for the returned tensor

    * `:shape` - shape of the returned tensor

    * `:names` - the names of the returned tensor

  ## Examples

      iex> key = Nx.Random.key(1701)
      iex> {uniform, _new_key} = Nx.Random.uniform(key)
      iex> uniform
      #Nx.Tensor<
        f32
        0.9728643894195557
      >

      iex> key = Nx.Random.key(1701)
      iex> {uniform, _new_key} = Nx.Random.uniform(key, shape: {3, 2}, type: :f16)
      iex> uniform
      #Nx.Tensor<
        f16[3][2]
        [
          [0.75390625, 0.6484375],
          [0.7294921875, 0.21484375],
          [0.09765625, 0.0693359375]
        ]
      >

      iex> key = Nx.Random.key(1701)
      iex> {uniform, _new_key} = Nx.Random.uniform(key, shape: {2, 2}, type: :c64)
      iex> uniform
      #Nx.Tensor<
        c64[2][2]
        [
          [0.18404805660247803+0.6546461582183838i, 0.5525915622711182+0.11568140983581543i],
          [0.6074584722518921+0.8104375600814819i, 0.247686505317688+0.21975469589233398i]
        ]
      >
  """
  defn uniform(key, min_val, max_val, opts \\ []) do
    keys = split(key)
    {uniform_split(keys[1], min_val, max_val, opts), keys[0]}
  end

  @doc """
  Same as `uniform/4` but assumes the key has already been split.
  """
  defn uniform_split(key, min_value, max_value, opts \\ []) do
    assert_key!(key)
    opts = keyword!(opts, [:names, :type, shape: {}])
    type = infer_float_type(min_value, max_value, opts)

    float_or_complex(key, type, opts[:shape], fn key, {_type, nbits} = type, shape ->
      u_one = Nx.tensor(1.0, type: type) |> Nx.bitcast({:u, nbits})

      min_value = Nx.as_type(min_value, type)
      max_value = Nx.as_type(max_value, type)

      random_bits(key, shape: shape, bit_width: nbits)
      |> Nx.right_shift(mantissa_shift(nbits, type))
      |> Nx.bitwise_or(u_one)
      |> Nx.bitcast(type)
      |> Nx.subtract(Nx.tensor(1.0, type: type))
      |> Nx.multiply(max_value - min_value)
      |> Nx.add(min_value)
      |> Nx.max(min_value)
      |> Nx.reshape(shape, take_names(opts))
    end)
    |> stop_grad()
  end

  @doc """
  Shortcut for `normal(key, 0.0, 1.0, opts)`.
  """
  defn normal(key, opts \\ []) do
    normal(key, 0.0, 1.0, opts)
  end

  @doc """
  Returns a normal distribution with the given `mean` and `standard_deviation`.

  ## Options

    * `:type` - a float or complex type for the returned tensor

    * `:shape` - shape of the returned tensor

    * `:names` - the names of the returned tensor

  ## Examples

      iex> key = Nx.Random.key(42)
      iex> {normal, _new_key} = Nx.Random.normal(key)
      iex> normal
      #Nx.Tensor<
        f32
        1.3694695234298706
      >

      iex> key = Nx.Random.key(42)
      iex> {normal, _new_key} = Nx.Random.normal(key, 0, 1, shape: {3, 2}, type: :f16)
      iex> normal
      #Nx.Tensor<
        f16[3][2]
        [
          [-0.32568359375, -0.77197265625],
          [0.39208984375, 0.5341796875],
          [0.270751953125, -2.080078125]
        ]
      >

      iex> key = Nx.Random.key(42)
      iex> {normal, _new_key} = Nx.Random.normal(key, 0, 1, shape: {2, 2}, type: :c64)
      iex> normal
      #Nx.Tensor<
        c64[2][2]
        [
          [-0.7632761001586914+0.8661127686500549i, -0.14282889664173126-0.7384796142578125i],
          [0.678461492061615+0.4118310809135437i, -2.269538402557373-0.3689095079898834i]
        ]
      >

      iex> key = Nx.Random.key(1337)
      iex> {normal, _new_key} = Nx.Random.normal(key, 10, 5, shape: {1_000})
      iex> Nx.mean(normal)
      #Nx.Tensor<
        f32
        9.70022201538086
      >
      iex> Nx.standard_deviation(normal)
      #Nx.Tensor<
        f32
        5.051416397094727
      >
  """
  defn normal(key, mean, standard_deviation, opts \\ []) do
    keys = split(key)
    {normal_split(keys[1], mean, standard_deviation, opts), keys[0]}
  end

  @doc """
  Same as `normal/4` but assumes the key has already been split.
  """
  defn normal_split(key, mean, standard_deviation, opts \\ []) do
    assert_key!(key)
    opts = keyword!(opts, [:names, :type, shape: {}])
    type = infer_float_type(mean, standard_deviation, opts)

    float_or_complex(key, type, opts[:shape], fn key, type, shape ->
      min_value = next_after_minus_1(type)
      u = uniform_split(key, min_value, 1, opts |> put_type(type) |> put_shape(shape))

      normal = Nx.sqrt(Nx.tensor(2, type: type)) * Nx.erf_inv(u)
      Nx.as_type(standard_deviation, type) * normal + Nx.as_type(mean, type)
    end)
    |> stop_grad()
  end

  @doc """
  Returns a sample from a multivariate normal distribution with given `mean` and `covariance` (matrix).
  The function assumes that the covariance is a positive semi-definite matrix.
  Otherwise, the result will not be normally distributed.

  ## Options

    * `:type` - a float type for the returned tensor

    * `:shape` - batch shape of the returned tensor, i.e. the prefix of the result shape excluding the last axis

    * `:names` - the names of the returned tensor

    * `:method` - a decomposition method used for the covariance. Must be one of :svd, :eigh, and :cholesky.
      Defaults to :cholesky. For singular covariance matrices, use :svd or :eigh.

  ## Examples

      iex> key = Nx.Random.key(12)
      iex> {multivariate_normal, _new_key} = Nx.Random.multivariate_normal(key, Nx.tensor([0]), Nx.tensor([[1]]))
      iex> multivariate_normal
      #Nx.Tensor<
        f32[1]
        [0.735927939414978]
      >

      iex> key = Nx.Random.key(12)
      iex> {multivariate_normal, _new_key} = Nx.Random.multivariate_normal(key, Nx.tensor([0, 0]), Nx.tensor([[1, 0], [0, 1]]))
      iex> multivariate_normal
      #Nx.Tensor<
        f32[2]
        [-1.3425945043563843, -0.40812060236930847]
      >

      iex> key = Nx.Random.key(12)
      iex> {multivariate_normal, _new_key} = Nx.Random.multivariate_normal(key, Nx.tensor([0]), Nx.tensor([[1]]), shape: {3, 2}, type: :f16)
      iex> multivariate_normal
      #Nx.Tensor<
        f16[3][2][1]
        [
          [
            [0.326904296875],
            [0.2176513671875]
          ],
          [
            [0.316650390625],
            [0.1109619140625]
          ],
          [
            [0.53955078125],
            [-0.8857421875]
          ]
        ]
      >

      iex> key = Nx.Random.key(12)
      iex> {multivariate_normal, _new_key} = Nx.Random.multivariate_normal(key, Nx.tensor([0, 0]), Nx.tensor([[1, 0], [0, 1]]), shape: {3, 2})
      iex> multivariate_normal
      #Nx.Tensor<
        f32[3][2][2]
        [
          [
            [0.9891449809074402, 1.0795185565948486],
            [-0.9467806220054626, 1.47813880443573]
          ],
          [
            [2.2095863819122314, -1.529456377029419],
            [-0.7933920621871948, 1.121195673942566]
          ],
          [
            [0.10976295918226242, -0.9959557056427002],
            [0.4754556119441986, 1.1413804292678833]
          ]
        ]
      >
  """
  defn multivariate_normal(key, mean, covariance, opts \\ []) do
    keys = split(key)
    {multivariate_normal_split(keys[1], mean, covariance, opts), keys[0]}
  end

  @doc """
  Same as `multivariate_normal/4` but assumes the key has already been split.
  """
  defn multivariate_normal_split(key, mean, covariance, opts \\ []) do
    assert_key!(key)
    opts = keyword!(opts, [:names, :type, :shape, method: :cholesky])
    {type, shape} = validate_multivariate_normal_args(mean, covariance, opts)
    mean = Nx.as_type(mean, type)
    covariance = Nx.as_type(covariance, type)

    a =
      case opts[:method] do
        :svd ->
          {u, s, _} = Nx.LinAlg.svd(covariance)
          u * Nx.sqrt(s)

        :eigh ->
          {s, u} = Nx.LinAlg.eigh(covariance)
          u * Nx.sqrt(s)

        :cholesky ->
          Nx.LinAlg.cholesky(covariance)
      end

    z =
      normal_split(key, 0.0, 1.0, shape: shape, type: type)
      |> Nx.reshape({:auto, Nx.size(mean)})

    # x = mean + z * a^T
    (mean + Nx.dot(z, [1], a, [1]))
    |> Nx.reshape(shape, names: opts[:names])
    |> stop_grad()
  end

  deftransformp validate_multivariate_normal_args(mean, covariance, opts) do
    if opts[:method] not in [:svd, :eigh, :cholesky] do
      raise ArgumentError,
            """
            method must be one of :svd, :eigh, and :cholesky
            """
    end

    type = infer_float_type(mean, covariance, opts)

    case type do
      {:f, _} -> nil
      {:bf, _} -> nil
      {:c, _} -> Nx.Shared.raise_complex_not_supported(:multivariate_normal_split, 4)
      _ -> raise ArgumentError, "expected float or complex type, got type #{inspect(type)}"
    end

    if Nx.rank(mean) != 1 do
      raise ArgumentError,
            """
            expected mean to have rank 1, got tensor with rank #{Nx.rank(mean)}
            """
    end

    dim = Nx.size(mean)

    if Nx.rank(covariance) != 2 do
      raise ArgumentError,
            """
            expected covariance to have rank 2, got tensor with rank #{Nx.rank(covariance)}
            """
    end

    if Nx.axis_size(covariance, 0) != Nx.axis_size(covariance, 1) do
      raise ArgumentError,
            """
            expected covariance to be a square matrix, got tensor with shape #{Nx.shape(covariance)}
            """
    end

    if Nx.axis_size(covariance, 0) != dim do
      raise ArgumentError,
            """
            expected mean and covariance to have the same dimensions, got #{dim} and #{Nx.axis_size(covariance, 0)}
            """
    end

    shape =
      case opts[:shape] do
        nil ->
          {dim}

        dims when is_tuple(dims) ->
          Tuple.append(dims, dim)

        _ ->
          raise ArgumentError,
                """
                shape must be a tuple of integers
                """
      end

    {type, shape}
  end

  @doc """
  Randomly shuffles tensor elements along an axis.

  ## Options

    * `:axis` - the axis along which to shuffle. Defaults to `0`

    * `:independent` - a boolean that indicates whether the permutations
      are independent along the given axis. Defaults to `false`

  ## Examples

      iex> key = Nx.Random.key(42)
      iex> {shuffled, _new_key} = Nx.Random.shuffle(key, Nx.iota({3, 4}, axis: 0))
      iex> shuffled
      #Nx.Tensor<
        s32[3][4]
        [
          [2, 2, 2, 2],
          [0, 0, 0, 0],
          [1, 1, 1, 1]
        ]
      >

      iex> key = Nx.Random.key(10)
      iex> {shuffled, _new_key} = Nx.Random.shuffle(key, Nx.iota({3, 4}, axis: 1), independent: true, axis: 1)
      iex> shuffled
      #Nx.Tensor<
        s32[3][4]
        [
          [2, 1, 3, 0],
          [3, 0, 1, 2],
          [2, 3, 0, 1]
        ]
      >
  """
  defn shuffle(key, tensor, opts \\ []) do
    opts = keyword!(opts, axis: 0, independent: false)
    axis = opts[:axis]

    if opts[:independent] do
      shuffle_independent(key, tensor, axis: axis)
    else
      {idx, key} = shuffle_independent(key, Nx.iota({Nx.axis_size(tensor, axis)}), axis: 0)
      {Nx.take(tensor, idx, axis: axis), key}
    end
  end

  defnp shuffle_independent(key, tensor, opts) do
    axis = opts[:axis]

    # reference: https://github.com/google/jax/blob/838bc454895ed2086563301936fb0d6d852fd198/jax/_src/random.py#L437
    exponent = 3
    uint32max = Nx.Constants.max_finite(:u32)

    num_rounds =
      Nx.ceil(exponent * Nx.log(Nx.size(tensor)) / Nx.log(uint32max))
      |> Nx.as_type(:u32)

    {_, out, key} =
      while {i = 0, tensor, key}, i < num_rounds do
        keys = split(key)
        sort_keys = random_bits(keys[1], shape: tensor.shape)
        tensor = sort_key_val(tensor, sort_keys, axis: axis)
        {i + 1, tensor, keys[0]}
      end

    {out, key}
  end

  defnp sort_key_val(tensor, sort_keys, opts \\ []) do
    idx = Nx.argsort(sort_keys, axis: opts[:axis])
    Nx.take(tensor, idx, axis: opts[:axis])
  end

  @choice_options """
  ## Options

    * `:samples` - The number of samples to take

    * `:axis` - The axis along which to take samples.
      If `nil`, the tensor is flattened beforehand.

    * `:replace` - a boolean that specifies if samples will
      be taken with or without replacement. Defaults to `true`.
  """
  @doc """
  Generates random samples from a tensor.

  #{@choice_options}

  ## Examples

      iex> k = Nx.Random.key(1)
      iex> t = Nx.iota({4, 3})
      iex> {result, _key} = Nx.Random.choice(k, t, samples: 4, axis: 0)
      iex> result
      #Nx.Tensor<
        s32[4][3]
        [
          [6, 7, 8],
          [9, 10, 11],
          [6, 7, 8],
          [0, 1, 2]
        ]
      >
      iex> {result, _key} = Nx.Random.choice(k, t, samples: 4, axis: 0, replace: false)
      iex> result
      #Nx.Tensor<
        s32[4][3]
        [
          [3, 4, 5],
          [9, 10, 11],
          [6, 7, 8],
          [0, 1, 2]
        ]
      >

  If no axis is specified, the tensor is flattened:

      iex> k = Nx.Random.key(2)
      iex> t = Nx.iota({3, 2})
      iex> {result, _key} = Nx.Random.choice(k, t)
      iex> result
      #Nx.Tensor<
        s32[1]
        [4]
      >
      iex> {result, _key} = Nx.Random.choice(k, t, samples: 6, replace: false)
      iex> result
      #Nx.Tensor<
        s32[6]
        [2, 0, 4, 5, 1, 3]
      >
  """
  deftransform choice(key, tensor, opts \\ []) do
    if is_list(opts) do
      choice_no_probabilities(key, tensor, opts)
    else
      choice(key, tensor, opts, [])
    end
  end

  defnp choice_no_probabilities(key, tensor, opts) do
    {tensor_shape, n_inputs, n_draws, axis, replace} = validate_choice_opts(tensor, opts)
    tensor = Nx.reshape(tensor, tensor_shape)

    if replace do
      {idx, key} = randint(key, 0, n_inputs, shape: {n_draws}, type: :u32)
      result = Nx.take(tensor, idx, axis: axis)
      {result, key}
    else
      {shuffled, key} = shuffle(key, tensor, axis: axis)
      result = Nx.slice_along_axis(shuffled, 0, n_draws, axis: axis)
      {result, key}
    end
  end

  @doc """
  Generates random samples from a tensor with specified probabilities.

  The probabilities tensor must have the same size as the axis along
  which the samples are being taken. If no axis is given, the size
  must be equal to the input tensor's size.

  #{@choice_options}

  ## Examples

      iex> k = Nx.Random.key(1)
      iex> t = Nx.iota({4, 3})
      iex> p = Nx.tensor([0.1, 0.7, 0.2])
      iex> {result, _key} = Nx.Random.choice(k, t, p, samples: 3, axis: 1)
      iex> result
      #Nx.Tensor<
        s32[4][3]
        [
          [1, 0, 1],
          [4, 3, 4],
          [7, 6, 7],
          [10, 9, 10]
        ]
      >
      iex> {result, _key} = Nx.Random.choice(k, t, p, samples: 3, axis: 1, replace: false)
      iex> result
      #Nx.Tensor<
        s32[4][3]
        [
          [1, 2, 0],
          [4, 5, 3],
          [7, 8, 6],
          [10, 11, 9]
        ]
      >

  If no axis is specified, the tensor is flattened.
  Notice that in the first case we get a higher occurence
  of the entries with bigger probabilities, while in the
  second case, without replacements, we get those samples
  first.

      iex> k = Nx.Random.key(2)
      iex> t = Nx.iota({2, 3})
      iex> p = Nx.tensor([0.01, 0.1, 0.19, 0.6, 0.05, 0.05])
      iex> {result, _key} = Nx.Random.choice(k, t, p)
      iex> result
      #Nx.Tensor<
        s32[1]
        [3]
      >
      iex> {result, _key} = Nx.Random.choice(k, t, p, samples: 6)
      iex> result
      #Nx.Tensor<
        s32[6]
        [3, 3, 3, 0, 3, 3]
      >
      iex> {result, _key} = Nx.Random.choice(k, t, p, samples: 6, replace: false)
      iex> result
      #Nx.Tensor<
        s32[6]
        [3, 1, 2, 5, 4, 0]
      >
  """
  defn choice(key, tensor, p, opts) do
    {tensor_shape, n_inputs, n_draws, axis, replace} = validate_choice_opts(tensor, opts)
    tensor = Nx.reshape(tensor, tensor_shape)

    case Nx.rank(p) do
      1 -> :ok
      r -> raise ArgumentError, "probability tensor must have rank 1, got: #{r}"
    end

    case {Nx.size(p), Nx.axis_size(tensor, axis)} do
      {n, n} ->
        :ok

      {p_size, a_size} ->
        raise ArgumentError,
              "probability tensor of size #{p_size} is expected to have size #{a_size}"
    end

    if replace do
      p_cumulative = Nx.cumulative_sum(p)
      {uniform, key} = uniform(key, shape: {n_draws}, type: Nx.type(p_cumulative))
      r = p_cumulative[-1] * (1 - uniform)

      # naïve implementation of jax.numpy.searchsorted
      p_cumulative = Nx.new_axis(p_cumulative, 0)
      r = Nx.new_axis(r, 1)
      idx = Nx.argmin(p_cumulative <= r, tie_break: :low, axis: 1)

      result = Nx.take(tensor, idx, axis: axis)
      {result, key}
    else
      {g, k} = gumbel(key, shape: {n_inputs}, type: Nx.type(p))
      g = -g - Nx.log(p)
      idx = g |> Nx.argsort() |> Nx.slice_along_axis(0, n_draws, axis: 0)

      result = Nx.take(tensor, idx, axis: axis)
      {result, k}
    end
  end

  deftransformp validate_choice_opts(tensor, opts) do
    opts = Keyword.validate!(opts, [:axis, samples: 1, replace: true])

    {axis, tensor_shape} =
      case opts[:axis] do
        nil ->
          {0, {Tuple.product(tensor.shape)}}

        axis ->
          {Nx.Shape.normalize_axis(tensor.shape, axis, tensor.names), tensor.shape}
      end

    if Nx.rank(tensor) < 1 do
      raise ArgumentError, "tensor must have rank 1 or greater"
    end

    n_draws = opts[:samples]

    if n_draws < 1 do
      raise "must take at least one sample, got samples=#{n_draws}"
    end

    n_inputs =
      case opts[:axis] do
        nil -> Nx.size(tensor)
        _ -> Nx.axis_size(tensor, axis)
      end

    replace = opts[:replace]

    if not replace and n_draws > n_inputs do
      raise ArgumentError, "cannot take more samples than the input size when replace: false"
    end

    {tensor_shape, n_inputs, n_draws, axis, replace}
  end

  @doc """
  Sample Gumbel random values with given shape and float dtype.

  ## Options

    * `:shape` - the shape of the output tensor containing the
      random samples. Defaults to `{}`

    * `:type` - the floating-point output type. Defaults to `{:f, 32}`

  ## Examples

      iex> {result, _key} = Nx.Random.gumbel(Nx.Random.key(1))
      iex> result
      #Nx.Tensor<
        f32
        -0.7294610142707825
      >

      iex> {result, _key} = Nx.Random.gumbel(Nx.Random.key(1), shape: {2, 3})
      iex> result
      #Nx.Tensor<
        f32[2][3]
        [
          [0.6247938275337219, -0.21740718185901642, 0.7678327560424805],
          [0.7778404355049133, 4.0895304679870605, 0.3029090166091919]
        ]
      >
  """
  defn gumbel(key, opts \\ []) do
    keys = split(key)
    {gumbel_split(keys[1], opts), keys[0]}
  end

  @doc """
  Same as `gumbel/2`, but assumes the key has been split beforehand.
  """
  defn gumbel_split(key, opts \\ []) do
    opts = keyword!(opts, shape: {}, type: {:f, 32})
    type = opts[:type]
    shape = opts[:shape]

    if not Nx.Type.float?(type) do
      raise ArgumentError, "output type must be floating-point, got: #{inspect(type)}"
    end

    u =
      uniform_split(key, Nx.Constants.smallest_positive_normal(type), 1, shape: shape, type: type)

    stop_grad(-Nx.log(-Nx.log(u)))
  end

  deftransformp next_after_minus_1({_, bits}) do
    # Get the floating point representation of -1 and
    # convert it to a big integer so the precision comes last (after exponent)
    <<x::size(bits)-big>> = <<-1::float-size(bits)-big>>

    # Decrement the precision by one (decrement because the sign is separate)
    # and convert it back to a float
    <<f::float-size(bits)-big>> = <<x - 1::integer-size(bits)-big>>

    f
  end

  deftransformp float_or_complex(key, type, shape, fun) do
    case type do
      {:c, _} ->
        type = Nx.Type.to_real(type)
        data = fun.(key, type, Tuple.append(shape, 2))
        to_complex = Nx.stack([1, Nx.Constants.i()])
        Nx.dot(data, to_complex)

      {t, _} when t == :f or t == :bf ->
        fun.(key, type, shape)

      _ ->
        raise ArgumentError, "expected float or complex type, got type #{inspect(type)}"
    end
  end

  deftransformp take_names(opts), do: Keyword.take(opts, [:names])

  deftransformp infer_type(left, right, opts) do
    if type = opts[:type] do
      Nx.Type.normalize!(type)
    else
      Nx.Type.merge(Nx.type(left), Nx.type(right))
    end
  end

  deftransformp infer_float_type(left, right, opts) do
    if type = opts[:type] do
      Nx.Type.normalize!(type)
    else
      Nx.Type.to_floating(Nx.Type.merge(Nx.type(left), Nx.type(right)))
    end
  end

  deftransformp put_type(opts, type), do: Keyword.put(opts, :type, type)
  deftransformp put_shape(opts, shape), do: Keyword.put(opts, :shape, shape)

  defnp assert_key!(tensor) do
    %{shape: shape, type: type} = tensor

    case shape do
      {2} ->
        :ok

      _ ->
        raise ArgumentError,
              "expected key to have shape {2}, got tensor with shape #{inspect(shape)}"
    end

    case type do
      {:u, 32} ->
        :ok

      _ ->
        raise ArgumentError,
              "expected key with 32-bit unsigned integer type, got key with type #{inspect(type)}"
    end
  end
end
