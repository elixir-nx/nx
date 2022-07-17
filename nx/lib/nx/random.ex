defmodule Nx.Random do
  @moduledoc """
  
  ## PRNG
  
  Unlike the stateful pseudorandom number
  generators (PRNGs) that users of most programming
  languages and numerical libraries
  may be accustomed to, Nx random functions all
  require an explicit PRNG state to be passed as
  a first argument. The random state is described by
  two unsigned 32-bit integers that we call a key,
  usually generated by the `Nx.Random.prng_key/1`
  function:
  
      iex> alias Nx.Random
      iex> key = Random.prng_key(12)
      iex> key
      [0, 12]
  
  This key can then be used in any of
  Nx’s random number generation routines:
  
      iex> Random.uniform(key)
      [0.3037703037261963]
  
  Note that using a key does not modify it, so
  reusing the same key will lead to the same result:
  
      iex> Random.uniform(key)
      [0.3037703037261963]
  
  If you need a new random number, you
  can use `Nx.Random.split/2` to generate new subkeys:
  
      iex> key, subkey = Random.split(key)
      iex> Random.uniform(subkey)
      [0.10536897]
  
  ## Advanced
  
  Design and Context
  
  TLDR: Nx PRNG = Threefry counter PRNG + a functional array-oriented splitting model
  
  To summarize, among other requirements, the Nx PRNG aims to:
  
  1. ensure reproducibility,
  
  2. parallelize well, both in terms of vectorization (generating array values)
     and multi-replica, multi-core computation. In particular it should not use
     sequencing constraints between random function calls.
  """

  import Nx.Defn, only: [defn: 2, defnp: 2]
  import Nx.Defn.Kernel, only: [assert_shape: 2, keyword!: 2]

  defp assert_key(tensor) do
    assert_shape(tensor, {2})
    type = Nx.type(tensor)

    if not Nx.Type.integer?(type) do
      raise ArgumentError,
            "expected key with integer type, got key with type #{inspect(type)}"
    end
  end

  @doc """
  Create a pseudo-random number generator (PRNG) key given an integer seed.
  """
  def key(seed) do
    k1 = Nx.right_shift(seed, 32) |> Nx.reshape({1})
    k2 = Nx.bitwise_and(seed, 0xFFFFFFFF) |> Nx.reshape({1})

    Nx.concatenate([k1, k2])
    |> Nx.reshape({2})
    |> Nx.as_type({:u, 32})
  end

  @doc """
  Splits a PRNG key into num new keys by adding a leading axis.
  """
  def split(key, num) when num > 1 do
    assert_key(key)

    impl(key, Nx.iota({num, 2}))
  end

  def fold_in(key, data) when is_integer(data) do
    assert_key(key)

    impl(key, key(data))
  end

  def random_bits(key, shape \\ {1}) do
    assert_key(key)

    impl(key, Nx.iota(shape))
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
  defnp threefry2x32(key, count) do
    transform(count, &threefry2x32_in_transform(&1))
    |> Nx.reshape({2, :auto})
    |> Nx.as_type({:u, 32})
    |> threefry2x32_20(key)
    |> transform(&threefry2x32_out_transform(&1, count))
  end

  defp threefry2x32_in_transform(count) do
    count_size = Nx.axis_size(count, 0)
    even? = rem(count_size, 2) == 0

    if even? do
      count
    else
      Nx.concatenate([count, Nx.tensor([0])])
    end
  end

  defp threefry2x32_out_transform(count, original) do
    count_size = Nx.axis_size(original, 0)
    even? = rem(count_size, 2) == 0

    if even? do
      count
    else
      count
      |> Nx.flatten()
      |> Nx.slice_along_axis(0, count_size, axis: 0)
    end
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

    state = {xs, ks, rotations}

    {_, {nxs, _, _}} =
      while {x = 0, state}, Nx.less(x, 5) do
        {x + 1, rolled_loop_step(x, state)}
      end

    nxs
  end

  defnp apply_round(xs, rot) do
    y1 = Nx.add(xs[0], xs[1])

    y2 =
      rotate_left(xs[1], rot)
      |> Nx.bitwise_xor(y1)

    Nx.stack([y1, y2])
    |> Nx.as_type({:u, 32})
  end

  defnp rolled_loop_step(i, {xs, ks, rs}) do
    {k1, k2, k3} = {ks[0], ks[1], ks[2]}
    {r1, r2} = {rs[0], rs[1]}

    {_, xs, _} =
      while {x = 0, xs, rs}, Nx.less(x, 4) do
        {x + 1, apply_round(xs, rs[0][x]), rs}
      end

    xs1 =
      Nx.broadcast(k1, xs[0])
      |> Nx.add(xs[0])

    xs2 =
      Nx.broadcast(k2 + i + 1, xs[1])
      |> Nx.add(xs[1])

    new_xs =
      Nx.stack([xs1, xs2])
      |> Nx.as_type({:u, 32})

    new_ks = Nx.stack([k2, k3, k1])
    new_rs = Nx.stack([r2, r1])

    {new_xs, new_ks, new_rs}
  end

  defnp rotate_left(x, rot) do
    nbits = 32
    x <<< rot ||| x >>> (nbits - rot)
  end

  defp finfo(type) do
    case type do
      {:bf, 16} -> [exp: 8, mant: 7]
      {:f, 16} -> [exp: 5, mant: 11]
      {:f, 32} -> [exp: 8, mant: 23]
      {:f, 64} -> [exp: 11, mant: 52]
    end
  end

  defn uniform(key, opts \\ []) do
    opts = keyword!(opts, shape: {1}, type: {:f, 32}, minval: 0.0, maxval: 1.0)

    transform(key, &uniform_transform(&1, opts))

    info = transform(opts[:type], &finfo(&1))

    shape = opts[:shape]
    type = {_dtype, nbits} = opts[:type]
    minval = opts[:minval]
    maxval = opts[:maxval]

    u_one = Nx.tensor(1.0, type: type) |> Nx.bitcast({:u, nbits})

    transform(key, &random_bits(&1, opts[:shape]))
    |> Nx.as_type({:u, nbits})
    |> Nx.right_shift(Nx.tensor(nbits - info[:mant], type: {:u, nbits}))
    |> Nx.bitwise_or(u_one)
    |> Nx.bitcast(type)
    |> Nx.subtract(Nx.tensor(1.0, type: type))
    |> Nx.multiply(maxval - minval)
    |> Nx.add(minval)
    |> Nx.reshape(shape)
    |> Nx.max(minval)
  end

  defp uniform_transform(key, opts) do
    assert_key(key)

    type = opts[:type]

    if not Nx.Type.float?(type) do
      raise ArgumentError,
            "expected float type, got type #{inspect(type)}"
    end
  end
end
