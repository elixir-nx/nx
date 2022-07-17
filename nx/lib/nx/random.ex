defmodule Nx.Random do
  @moduledoc """
  Nx conveniences for random number generator.
  """

  import Nx.Defn, only: [defn: 2, defnp: 2]
  import Nx.Defn.Kernel, only: [assert_shape: 2]

  defp assert_key(tensor) do
    assert_shape(tensor, {2})
    type = Nx.type(tensor)

    if not Nx.Type.integer?(type) do
      raise ArgumentError,
            "expected tensor with integer type, got tensor with type #{inspect(type)}"
    end
  end

  def key(seed) do
    k1 = Nx.right_shift(seed, 32) |> Nx.reshape({1})
    k2 = Nx.bitwise_and(seed, 0xFFFFFFFF) |> Nx.reshape({1})

    Nx.concatenate([k1, k2])
    |> Nx.reshape({2})
    |> Nx.as_type({:u, 32})
  end

  def split!(key, num \\ 2) when num > 1 do
    assert_key(key)

    split(key, num)
  end

  def split(key, num \\ 2) do
    impl(key, Nx.iota({num, 2}))
  end

  def fold_in!(key, data) when is_integer(data) do
    assert_key(key)

    fold_in(key, data)
  end

  def fold_in(key, data) do
    impl(key, key(data))
  end

  def random_bits!(key, shape \\ {1}) do
    assert_key(key)

    random_bits(key, shape)
  end

  def random_bits(key, shape \\ {1}) do
    impl(key, Nx.iota(shape))
  end

  def impl(key, count) do
    shape = Nx.shape(count)

    reshaped_key = Nx.reshape(key, {2, 1})

    reshaped_count =
      Nx.reshape(count, {:auto})
      |> Nx.as_type({:u, 32})

    threefry2x32(reshaped_key, reshaped_count)
    |> Nx.reshape(shape)
    |> Nx.as_type({:u, 32})
  end

  def threefry2x32(key, count) do
    count_size = Nx.axis_size(count, 0)
    even? = rem(count_size, 2) == 0

    if even? do
      count
    else
      Nx.concatenate([count, Nx.tensor([0])])
    end
    |> Nx.reshape({2, :auto})
    |> Nx.as_type({:u, 32})
    |> threefry2x32_20(key)
    |> then(fn output ->
      if even? do
        output
      else
        output
        |> Nx.flatten()
        |> Nx.slice_along_axis(0, count_size, axis: 0)
      end
    end)
  end

  defn threefry2x32_20(xs, ks) do
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

  def uniform(key, shape \\ {1}, type \\ {:f, 32}, minval \\ 0.0, maxval \\ 1.0) do
    bits = random_bits(key, shape)

    if not Nx.Type.float?(type) do
      raise ArgumentError,
            "expected float type, got type #{inspect(type)}"
    end

    info = finfo(type)

    _uniform(key, shape, type, minval, maxval, bits, info)
  end

  defp _uniform(_key, shape, type = {_dtype, nbits}, minval, maxval, bits, info) do
    u_one = Nx.tensor(1.0, type: type) |> Nx.bitcast({:u, nbits})

    bits
    |> Nx.right_shift(Nx.tensor(nbits - info[:mant], type: {:u, nbits}))
    |> Nx.bitwise_or(u_one)
    |> Nx.bitcast(type)
    |> Nx.subtract(Nx.tensor(1.0, type: type))
    |> Nx.multiply(maxval - minval)
    |> Nx.add(minval)
    |> Nx.reshape(shape)
    |> Nx.max(minval)
  end
end
