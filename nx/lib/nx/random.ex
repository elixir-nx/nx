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
    |> Nx.as_type(:u32)
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

      iex> key = Nx.Random.key(999999999999)
      iex> Nx.Random.split(key, parts: 4)
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
  defn split(key, opts \\ []) do
    assert_key!(key)
    opts = keyword!(opts, parts: 2)
    threefry2x32(key, {opts[:parts], 2})
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

    k1 = Nx.right_shift(data, 32)
    k2 = Nx.bitwise_and(data, 0xFFFFFFFF)

    {x1, x2} =
      Nx.stack([k1, k2])
      |> Nx.reshape({2, :auto})
      |> Nx.as_type(:u32)
      |> threefry2x32_20_pair(key)

    [x1, x2]
    |> Nx.stack(axis: -1)
    |> Nx.reshape(fold_shape(Nx.shape(data)))
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
      iex> {randint, _new_key} = Nx.Random.randint(key, 1, 100)
      iex> randint
      #Nx.Tensor<
        s64
        66
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
