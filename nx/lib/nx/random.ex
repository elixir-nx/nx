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
      while {x = 0, state}, x < 5 do
        {x + 1, rolled_loop_step(x, state)}
      end

    Nx.stack([nx1, nx2])
  end

  defnp apply_round({xs1, xs2}, rot) do
    y1 = xs1 + xs2

    y2 =
      rotate_left(xs2, rot)
      |> Nx.bitwise_xor(y1)

    # losing precision on purpose due to upcasts
    {y1 |> Nx.as_type({:u, 32}), y2 |> Nx.as_type({:u, 32})}
  end

  defnp rolled_loop_step(i, {{_xs1, _xs2} = xs, {k1, k2, k3}, {r1, r2}}) do
    {_, {xs1, xs2}, _} =
      while {x = 0, xs, r1}, x < 4 do
        {x + 1, apply_round(xs, r1[x]), r1}
      end

    xs1 = k1 + xs1

    xs2 = k2 + i + 1 + xs2

    new_xs = {xs1 |> Nx.as_type({:u, 32}), xs2 |> Nx.as_type({:u, 32})}

    new_ks = {k2, k3, k1}
    new_rs = {r2, r1}

    {new_xs, new_ks, new_rs}
  end

  defnp rotate_left(x, rot) do
    x <<< rot ||| x >>> (@nbits - rot)
  end

  defnp float_info(type) do
    case type do
      {:bf, 16} -> [exp: 8, mantissa: 7]
      {:f, 16} -> [exp: 6, mantissa: 10]
      {:f, 32} -> [exp: 8, mantissa: 23]
      {:f, 64} -> [exp: 11, mantissa: 52]
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
        91
      >

      iex> key = Nx.Random.key(1701)
      iex> Nx.Random.randint(key, 1, 100, shape: {3,3,2}, type: :u32)
      #Nx.Tensor<
        u32[3][3][2]
        [
          [
            [68, 4],
            [76, 75],
            [11, 75]
          ],
          [
            [6, 85],
            [17, 42],
            [16, 33]
          ],
          [
            [15, 12],
            [3, 88],
            [16, 92]
          ]
        ]
      >

  """
  defn randint(key, min_val, max_val, opts \\ []) do
    opts = keyword!(opts, [:names, shape: {}, type: {:s, 64}])
    assert_key!(key)

    shape = Nx.shape(opts[:shape])
    type = {_, nbits} = normalize(opts[:type])

    case type do
      {:u, _} ->
        :ok

      {:s, _} ->
        :ok

      _ ->
        raise ArgumentError,
              "expected integer type, got type #{inspect(type)}"
    end

    keys = split(key)

    min_val = Nx.broadcast(min_val, shape)
    max_val = Nx.broadcast(max_val, shape)

    higher_bits = random_bits(keys[0], shape: shape, bit_width: nbits)
    lower_bits = random_bits(keys[1], shape: shape, bit_width: nbits)
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
    iex> Nx.Random.uniform(key, shape: {3,3,2}, type: :f16)
    #Nx.Tensor<
      f16[3][3][2]
      [
        [
          [0.5712890625, 0.318359375],
          [0.744140625, 0.576171875],
          [0.2412109375, 0.9833984375]
        ],
        [
          [0.0556640625, 0.42578125],
          [0.0263671875, 0.0634765625],
          [0.12890625, 0.9306640625]
        ],
        [
          [0.46484375, 0.087890625],
          [0.3857421875, 0.169921875],
          [0.0419921875, 0.53125]
        ]
      ]
    >
  """
  defn uniform(key, min_value, max_value, opts \\ []) do
    opts = keyword!(opts, [:names, shape: {}, type: {:f, 32}])
    assert_key!(key)

    type = {_type, nbits} = normalize(opts[:type])

    case type do
      {:f, _} ->
        :ok

      {:bf, _} ->
        :ok

      _ ->
        raise ArgumentError,
              "expected float type, got type #{inspect(type)}"
    end

    info = float_info(type)
    shape = Nx.shape(opts[:shape])
    u_one = Nx.tensor(1.0, type: type) |> Nx.bitcast({:u, nbits})

    random_bits(key, shape: shape, bit_width: nbits)
    |> Nx.as_type({:u, nbits})
    |> Nx.right_shift(Nx.tensor(nbits - info[:mantissa], type: {:u, nbits}))
    |> Nx.bitwise_or(u_one)
    |> Nx.bitcast(type)
    |> Nx.subtract(Nx.tensor(1.0, type: type))
    |> Nx.multiply(max_value - min_value)
    |> Nx.add(min_value)
    |> Nx.max(min_value)
    |> Nx.reshape(shape, take_names(opts))
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
          [-0.1424560546875, 0.1005859375],
          [-0.513671875, 0.308349609375]
        ],
        [
          [-1.423828125, -1.9873046875],
          [-0.599609375, 0.662109375],
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
    iex> Nx.Random.normal(key, 0, 1, shape: {3,3}, type: :c64)
    #Nx.Tensor<
      c64[3][3]
      [
        [-0.7446164488792419-1.6652092933654785i, -1.2271071672439575+0.23443973064422607i, -0.053599901497364044-0.24498997628688812i],
        [0.9805877208709717+0.4470720589160919i, 0.44665536284446716-0.3771430552005768i, 0.7519879341125488+0.2825981676578522i],
        [0.4686059355735779-0.11017023772001266i, 0.4970967769622803+0.5699526071548462i, 0.15884320437908173+0.3396047353744507i]
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
  defn normal(key, mean \\ 0, standard_deviation \\ 1, opts \\ []) do
    opts = keyword!(opts, [:names, shape: {}, type: {:f, 32}])
    assert_key!(key)

    type = normalize(opts[:type])

    case type do
      {:c, _} ->
        k = split(key, 2)
        opts = as_real_type(opts)
        real = normal_real(k[0], mean, standard_deviation, opts)
        imag = normal_real(k[1], mean, standard_deviation, opts)
        real + Nx.Constants.i() * imag

      {t, _} when t == :f or t == :bf ->
        normal_real(key, mean, standard_deviation, opts)

      _ ->
        raise ArgumentError,
              "expected float or complex type, got type #{inspect(type)}"
    end
  end

  defnp normal_real(key, mean, standard_deviation, opts \\ []) do
    u = uniform(key, -1, 1, opts)

    normal = Nx.sqrt(Nx.type(2, type: opts[:type]) * Nx.erf_inv(u)
    standard_deviation * normal + mean
  end

  deftransformp as_real_type(opts) do
    real = Nx.Type.to_real(opts[:type])
    Keyword.put(opts, :type, real)
  end

  deftransformp take_names(opts), do: Keyword.take(opts, [:names])
  deftransformp normalize(type), do: Nx.Type.normalize!(type)

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
