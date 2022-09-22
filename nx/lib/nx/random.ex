defmodule Nx.Random do
  @moduledoc """
  Pseudo-random number generators.

  Unlike the stateful pseudo-random number generators (PRNGs)
  that users of most programming languages and numerical libraries
  may be accustomed to, Nx random functions require an explicit
  PRNG key to be passed as a first argument. That key is defined by
  an `Nx.Tensor` composed of 2 unsigned 32-bit integers, usually
  generated by the `Nx.Random.key/1` function:

      iex> Nx.Random.key(12)
      #Nx.Tensor<
        u32[2]
        [0, 12]
      >

  This key can then be used in any of Nx’s random number generation
  routines:

      iex> key = Nx.Random.key(12)
      iex> Nx.Random.uniform(key)
      #Nx.Tensor<
        f32
        0.3037703037261963
      >

  Note that using a key does not modify it, so reusing the same key
  will lead to the same result:

      iex> key = Nx.Random.key(12)
      iex> Nx.Random.uniform(key)
      #Nx.Tensor<
        f32
        0.3037703037261963
      >

  If you need a new random number, you can use `Nx.Random.split/2` to
  generate new subkeys:

      iex> key = Nx.Random.key(12)
      iex> keys = Nx.Random.split(key)
      iex> Nx.Random.uniform(keys[0])
      #Nx.Tensor<
        f32
        0.1531379222869873
      >
      iex> Nx.Random.uniform(keys[1])
      #Nx.Tensor<
        f32
        0.7691127061843872
      >

  ## Design and Context

  In short, Nx's PRNGs are based on a Threefry counter PRNG
  associated to a functional array-oriented splitting model.
  To summarize, among other requirements, Nx's PRNG aims to:

  1. Ensure reproducibility

  2. Parallelize well, both in terms of vectorization
     (generating array values) and multi-replica, multi-core
     computation. In particular it should not use sequencing
     constraints between random function calls.
  """

  import Nx.Defn, only: [deftransformp: 2, defn: 2, defnp: 2]

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
  defn key(seed) do
    k1 = Nx.right_shift(seed, 32)
    k2 = Nx.bitwise_and(seed, 0xFFFFFFFF)

    Nx.stack([k1, k2])
    |> Nx.as_type({:u, 32})
  end

  @doc """
  Splits a PRNG key into `num` new keys by adding a leading axis.

  ## Examples

      iex> key = Nx.Random.key(1701)
      iex> Nx.Random.split(key, 2)
      #Nx.Tensor<
        u32[2][2]
        [
          [56197195, 1801093307],
          [961309823, 1704866707]
        ]
      >

      iex> key = Nx.Random.key(999999999999)
      iex> Nx.Random.split(key, 4)
      #Nx.Tensor<
        u32[4][2]
        [
          [3959978897, 4079927650],
          [3769699049, 3585271160],
          [3182829676, 333122445],
          [3185556048, 1258545461]
        ]
      >
  """
  defn split(key, num \\ 2) do
    assert_key!(key)

    key
    |> threefry2x32(Nx.iota({num, 2}))
    |> Nx.as_type({:u, 32})
  end

  defnp threefry2x32(key, count) do
    padding =
      Nx.size(count)
      |> rem(2)

    Nx.flatten(count)
    |> Nx.pad(0, [{0, padding, 0}])
    |> Nx.reshape({2, :auto})
    |> Nx.as_type({:u, 32})
    |> threefry2x32_20(key)
    |> Nx.flatten()
    |> Nx.pad(0, [{0, -padding, 0}])
    |> Nx.reshape(count)
  end

  defnp threefry2x32_20(xs, ks) do
    rotations =
      {Nx.tensor([13, 15, 26, 6], type: {:u, 8}), Nx.tensor([17, 29, 16, 24], type: {:u, 8})}

    key1 = ks[0]
    key2 = ks[1]

    xs1 = xs[0] + key1
    xs2 = xs[1] + key2
    xs = {xs1, xs2}

    ks = {
      key2,
      Nx.bitwise_xor(key1, key2)
      |> Nx.bitwise_xor(0x1BD11BDA),
      key1
    }

    state = {xs, ks, rotations}

    {_, {{nx1, nx2}, _, _}} =
      while {x = Nx.tensor(0, type: :u32), state}, x < 5 do
        {x + Nx.tensor(1, type: :u32), rolled_loop_step(x, state)}
      end

    Nx.stack([nx1, nx2])
  end

  defnp apply_round({xs1, xs2}, rot) do
    y1 = xs1 + xs2

    y2 =
      rotate_left(xs2, rot)
      |> Nx.bitwise_xor(y1)

    {y1, y2}
  end

  defnp rolled_loop_step(i, {{_xs1, _xs2} = xs, {k1, k2, k3}, {r1, r2}}) do
    {_, {xs1, xs2}, _} =
      while {x = Nx.tensor(0, type: :u32), xs, r1}, x < 4 do
        {x + Nx.tensor(1, type: :u32), apply_round(xs, r1[x]), r1}
      end

    xs1 = k1 + xs1
    xs2 = k2 + i + Nx.tensor(1, type: :u32) + xs2

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
          threefry2x32(key, Nx.iota({Nx.size(shape) * 2}))
          |> Nx.reshape({2, :auto})
          |> Nx.as_type({:u, 64})

        bits = bits[0] <<< 32 ||| bits[1]
        Nx.reshape(bits, shape)

      32 ->
        threefry2x32(key, Nx.iota(shape, type: {:s, 64}))
        |> Nx.as_type({:u, 32})

      _ ->
        threefry2x32(key, Nx.iota(shape, type: {:s, 64}))
        |> Nx.as_type({:u, bit_width})
    end
  end

  defnp mantissa(type) do
    case type do
      {:bf, 16} -> 7
      {:f, 16} -> 10
      {:f, 32} -> 23
      {:f, 64} -> 52
    end
  end

  @doc """
  Sample uniform random integer values in `[min_value, max_value)`.

  ## Options

    * `:type` - the integer type for the returned tensor
    * `:shape` - shape of the returned tensor
    * `:names` - the names of the returned tensor

  ## Examples

      iex> key = Nx.Random.key(1701)
      iex> Nx.Random.randint(key, 1, 100)
      #Nx.Tensor<
        s64
        10
      >

      iex> key = Nx.Random.key(1701)
      iex> Nx.Random.randint(key, 1, 100, shape: {3,3,2}, type: :u32)
      #Nx.Tensor<
        u32[3][3][2]
        [
          [
            [77, 54],
            [50, 77],
            [83, 53]
          ],
          [
            [65, 68],
            [98, 69],
            [27, 95]
          ],
          [
            [50, 18],
            [32, 60],
            [83, 50]
          ]
        ]
      >

  """
  defn randint(key, min_val, max_val, opts \\ []) do
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

    span = max_val - min_val

    multiplier =
      Nx.power(2, Nx.quotient(nbits, 2))
      |> Nx.remainder(span)
      |> Nx.power(2)
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
  end

  deftransformp randint_random_bits_shape(shape), do: Tuple.insert_at(shape, 0, 2)

  @doc """
  Shortcut for `uniform(key, 0.0, 1.0, opts)`.
  """
  defn uniform(key, opts \\ []) do
    uniform(key, 0.0, 1.0, opts)
  end

  @doc """
  Sample uniform float values in `[min_val, max_val)`.

  ## Options

    * `:type` - a float type for the returned tensor

    * `:shape` - shape of the returned tensor

    * `:names` - the names of the returned tensor

  ## Examples

      iex> key = Nx.Random.key(1701)
      iex> Nx.Random.uniform(key)
      #Nx.Tensor<
        f32
        0.1725379228591919
      >

      iex> key = Nx.Random.key(1701)
      iex> Nx.Random.uniform(key, shape: {3, 2}, type: :f16)
      #Nx.Tensor<
        f16[3][2]
        [
          [0.076171875, 0.18359375],
          [0.8125, 0.65625],
          [0.53125, 0.30078125]
        ]
      >

      iex> key = Nx.Random.key(1701)
      iex> Nx.Random.uniform(key, shape: {2, 2}, type: :c64)
      #Nx.Tensor<
        c64[2][2]
        [
          [0.9313580989837646+0.4727839231491089i, 0.5327110290527344+0.6050407886505127i],
          [0.5649927854537964+0.13510417938232422i, 0.730334997177124+0.1008983850479126i]
        ]
      >
  """
  defn uniform(key, min_value, max_value, opts \\ []) do
    assert_key!(key)
    opts = keyword!(opts, [:names, :type, shape: {}])
    type = infer_float_type(min_value, max_value, opts)

    float_or_complex(key, type, opts[:shape], fn key, {_type, nbits} = type, shape ->
      u_one = Nx.tensor(1.0, type: type) |> Nx.bitcast({:u, nbits})

      min_value = Nx.as_type(min_value, type)
      max_value = Nx.as_type(max_value, type)

      random_bits(key, shape: shape, bit_width: nbits)
      |> Nx.right_shift(Nx.tensor(nbits - mantissa(type), type: {:u, nbits}))
      |> Nx.bitwise_or(u_one)
      |> Nx.bitcast(type)
      |> Nx.subtract(Nx.tensor(1.0, type: type))
      |> Nx.multiply(max_value - min_value)
      |> Nx.add(min_value)
      |> Nx.max(min_value)
      |> Nx.reshape(shape, take_names(opts))
    end)
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
      iex> Nx.Random.normal(key)
      #Nx.Tensor<
        f32
        -0.18471182882785797
      >

      iex> key = Nx.Random.key(42)
      iex> Nx.Random.normal(key, 0, 1, shape: {3, 3, 2}, type: :f16)
      #Nx.Tensor<
        f16[3][3][2]
        [
          [
            [-0.6201171875, -1.017578125],
            [-0.1424560546875, 0.10052490234375],
            [-0.513671875, 0.308349609375]
          ],
          [
            [-1.423828125, -1.9873046875],
            [-0.59912109375, 0.662109375],
            [-0.54150390625, -2.3359375]
          ],
          [
            [-0.1448974609375, -0.4560546875],
            [0.2802734375, 0.2548828125],
            [-1.1044921875, -1.359375]
          ]
        ]
      >

      iex> key = Nx.Random.key(42)
      iex> Nx.Random.normal(key, 0, 1, shape: {2, 2}, type: :c64)
      #Nx.Tensor<
        c64[2][2]
        [
          [0.48962485790252686+0.25225749611854553i, -0.6273903846740723-0.7026026844978333i],
          [0.715733528137207-0.5684993863105774i, -1.7461833953857422-2.725870370864868i]
        ]
      >

      iex> key = Nx.Random.key(1337)
      iex> normal = Nx.Random.normal(key, 10, 5, shape: {1_000})
      iex> Nx.mean(normal)
      #Nx.Tensor<
        f32
        9.897998809814453
      >
      iex> Nx.standard_deviation(normal)
      #Nx.Tensor<
        f32
        4.988009929656982
      >
  """
  defn normal(key, mean, standard_deviation, opts \\ []) do
    assert_key!(key)
    opts = keyword!(opts, [:names, :type, shape: {}])
    type = infer_float_type(mean, standard_deviation, opts)

    float_or_complex(key, type, opts[:shape], fn key, type, shape ->
      min_value = -1 + Nx.Constants.smallest_positive_normal_number(type)
      u = uniform(key, min_value, 1, opts |> put_type(type) |> put_shape(shape))

      normal = Nx.sqrt(Nx.tensor(2, type: type)) * Nx.erf_inv(u)
      Nx.as_type(standard_deviation, type) * normal + Nx.as_type(mean, type)
    end)
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
        raise(
          ArgumentError,
          "expected key to have shape {2}, got tensor with shape #{inspect(shape)}"
        )
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
