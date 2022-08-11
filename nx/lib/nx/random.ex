defmodule Nx.Random do
  @moduledoc """

  ## PRNG
  (Pseudo-Random Number Generator)

  Unlike the stateful pseudo-random number
  generators (PRNGs) that users of most programming
  languages and numerical libraries
  may be accustomed to, Nx random functions all
  require an explicit PRNG state to be passed as
  a first argument. The random state is described by
  an `Nx.Tensor` composed by 2 unsigned 32-bit integers
  that we call a key, usually generated by the `Nx.Random.key/1`
  function:

      iex> Nx.Random.key(12)
      #Nx.Tensor<
        u32[2]
        [0, 12]
      >

  This key can then be used in any of
  Nx’s random number generation routines:

      iex> key = Nx.Random.key(12)
      iex> Nx.Random.uniform(key)
      #Nx.Tensor<
        f32[1]
        [0.3037703037261963]
      >

  Note that using a key does not modify it, so
  reusing the same key will lead to the same result:

      iex> key = Nx.Random.key(12)
      iex> Nx.Random.uniform(key)
      #Nx.Tensor<
        f32[1]
        [0.3037703037261963]
      >

  If you need a new random number, you
  can use `Nx.Random.split/2` to generate new subkeys:

      iex> key = Nx.Random.key(12)
      iex> keys = Nx.Random.split(key)
      iex> Nx.Random.uniform(keys[0])
      #Nx.Tensor<
        f32[1]
        [0.1531379222869873]
      >
      iex> Nx.Random.uniform(keys[1])
      #Nx.Tensor<
        f32[1]
        [0.7691127061843872]
      >

  ## Advanced

  Design and Context

   In short, Nx's PRNGs are based on a Threefry counter PRNG
   associated to a functional array-oriented splitting model

  To summarize, among other requirements, Nx's PRNG aims to:

  1. Ensure reproducibility;

  2. Parallelize well, both in terms of vectorization (generating array values)
     and multi-replica, multi-core computation. In particular it should not use
     sequencing constraints between random function calls.
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
    assert_key(key)

    impl(key, Nx.iota({num, 2}))
  end

  defn fold_in(key, data) do
    assert_key(key)

    impl(key, key(data))
  end

  deftransformp max_count_size(shape, bit_width) do
    [shape: {rem(Nx.size(shape) * bit_width, 32)
      |> Nx.any()
      |> Nx.add(Nx.size(shape) * bit_width / 32)}]
  end

  defn random_bits(key, opts \\ []) do
    assert_key(key)
    opts = keyword!(opts, shape: {1}, bit_width: 32)
    shape = opts[:shape]
    bit_width = opts[:bit_width]

    # {_, key} =
    #   while {x = 0, key}, x <  Nx.rank(opts[:shape]) do
    #     {x + 1, fold_in(key, x)}
    #   end


    #bits = impl(key, Nx.iota(max_count_size(shape, bit_width)[:shape]))

    bits = case bit_width do
      64 -> bits =
            impl(key, Nx.concatenate([Nx.iota(shape), Nx.iota(shape)+Nx.size(shape)]))
            |> Nx.reshape({2, :auto})
            |> Nx.as_type({:u, bit_width})
            (bits[0] <<< 32) ||| bits[1]
      32 -> impl(key, Nx.iota(shape))
      _ ->  impl(key, Nx.iota(shape))
        # |> Nx.reshape({1, :auto})
        # |> Nx.right_shift(Nx.multiply(
        #   Nx.tensor(bit_width, type: {:u, 32}),
        #   Nx.iota({2})
        #   |> Nx.as_type({:u, 32})
        # ))
        # |> Nx.bitwise_and(Nx.Constants.max_finite({:u, bit_width}))
        |> Nx.as_type({:u, bit_width})
    end

    Nx.reshape(bits, shape)
  end

  defnp impl(key, count) do
    shape = Nx.shape(count)

    reshaped_key = Nx.reshape(key, {2, 1})

    reshaped_count =
      Nx.reshape(count, {:auto})
      |> Nx.as_type({:u, 32})

    threefry2x32(reshaped_key, reshaped_count)
    |> Nx.reshape(shape)
    |> Nx.as_type({:u, 32})
  end

  # Check count
  defn threefry2x32(key, count) do
    padding = Nx.size(count)
      |> rem(2)

    Nx.pad(count, 0, [{0,padding,0}])
    |> Nx.reshape({2, :auto})
    |> Nx.as_type({:u, 32})
    |> threefry2x32_20(key)
    |> Nx.reshape({:auto})
    |> Nx.pad(0, [{0,-padding,0}])
  end

  defnp threefry2x32_20(xs, ks) do
    rotations = Nx.tensor([[13, 15, 26, 6], [17, 29, 16, 24]], type: {:u, 8})
    key1 = ks[0]
    key2 = ks[1]

    xs = Nx.add(ks, xs)

    ks =
      Nx.stack([
        key2,
        Nx.bitwise_xor(key1, key2)
        |> Nx.bitwise_xor(0x1BD11BDA),
        key1
      ])
      |> Nx.as_type({:u, 32})
    #we lose precision on purpose

    state = {xs, ks, rotations}

    {_, {nxs, _, _}} =
      while {x = 0, state}, Nx.less(x, 5) do
        {x + 1, rolled_loop_step(x, state)}
      end

    nxs
  end

  defnp apply_round(xs, rot) do
    y1 = Nx.sum(xs, axes: [0])

    y2 =
      rotate_left(xs[1], rot)
      |> Nx.bitwise_xor(y1)

    #losing precision on purpose due to upcasts
    Nx.stack([y1, y2])
    |> Nx.as_type({:u, 32})
  end

  defnp rolled_loop_step(i, {xs, ks, rs}) do
    k1 = ks[0]
    k2 = ks[1]
    k3 = ks[2]
    r1 = rs[0]
    r2 = rs[1]

    {_, xs, _} =
      while {x = 0, xs, rs}, x < 4 do
        {x + 1, apply_round(xs, rs[0][x]), rs}
      end

    xs1 = k1 + xs[0]

    xs2 = k2 + i + 1 + xs[1]

    new_xs =
      Nx.stack([xs1, xs2])
      |> Nx.as_type({:u, 32})

    new_ks = Nx.stack([k2, k3, k1])
    new_rs = Nx.stack([r2, r1])

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
  Sample uniform random integer values in [min_val, max_val).

  ## Options

    * `:type` - an int type for the returned tensor

    * `:shape` - shape of the returned tensor

  """
  defn randint(key, min_val, max_val, opts \\ []) do
    opts = keyword!(opts, shape: {1}, type: {:s, 32})
    assert_key(key)

    shape = opts[:shape]
    type = {_, nbits} = normalize(opts[:type])
    case type do
      {:u, _} -> :ok
      {:s, _} -> :ok
      _ -> raise ArgumentError,
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

    Nx.as_type(min_val + offset, type)
  end

  @doc """
  Sample uniform float values in [min_val, max_val).

  ## Options

    * `:type` - a float type for the returned tensor

    * `:shape` - shape of the returned tensor

    * `:min_val` - minimum value, default is 0.0

    * `:max_val` - maximum value, default is 1.0

  """
  defn uniform(key, opts \\ []) do
    opts = keyword!(opts, shape: {1}, type: {:f, 32}, min_val: 0.0, max_val: 1.0)

    assert_key(key)

    type = {_type, nbits} = normalize(opts[:type])
    case type do
      {:f, _} -> :ok
      {:bf, _} -> :ok

      _ -> raise ArgumentError,
            "expected float type, got type #{inspect(type)}"
    end

    info = float_info(type)

    shape = opts[:shape]
    min_val = opts[:min_val]
    max_val = opts[:max_val]

    u_one = Nx.tensor(1.0, type: type) |> Nx.bitcast({:u, nbits})

    random_bits(key, shape: shape, bit_width: nbits)
    |> Nx.as_type({:u, nbits})
    |> Nx.right_shift(Nx.tensor(nbits - info[:mantissa], type: {:u, nbits}))
    |> Nx.bitwise_or(u_one)
    |> Nx.bitcast(type)
    |> Nx.subtract(Nx.tensor(1.0, type: type))
    |> Nx.multiply(max_val - min_val)
    |> Nx.add(min_val)
    |> Nx.reshape(shape)
    |> Nx.max(min_val)
  end

  deftransformp normalize(type), do: Nx.Type.normalize!(type)

  defnp assert_key(tensor) do
    %{shape: shape, type: type} = tensor
    #shape = Nx.shape(tensor)
    case shape do
      {2} -> :ok

      _ -> raise(
        ArgumentError,
        "expected key to have shape {2}, got tensor with shape #{inspect(shape)}"
      )
    end

    #type = Nx.type(tensor)
    case type do
      {:u, _} -> :ok
      {:s, _} -> :ok
      _ -> raise ArgumentError,
            "expected key with integer type, got key with type #{inspect(type)}"
    end
  end
end
