defmodule Nx.Random do
  @moduledoc """
  Nx conveniences for random number generator.
  """

  import Nx.Defn, only: [defnp: 2]
  import Nx.Defn.Kernel, only: [assert_shape: 2]

  alias Nx.Tensor, as: T

  defp assert_key(tensor) do
    assert_shape(tensor, {1, 2})
    type = Nx.type(tensor)
    if not Nx.Type.integer?(type) do
      raise ArgumentError,
              "expected tensor with integer type, got tensor with type #{inspect(type)}"
    end
  end

  @spec threefry_seed(integer()) :: T
  def threefry_seed(seed) when is_integer(seed) do
    k1 = Nx.right_shift(seed, 32) |> Nx.reshape({1})
    k2 = Nx.bitwise_and(seed, 0xFFFFFFFF) |> Nx.reshape({1})
    Nx.concatenate([k1, k2])
    |> Nx.reshape({1, 2})
    |> Nx.as_type({:u, 32})
  end

  @spec threefry_split(T, pos_integer()) :: T
  def threefry_split(key, num \\ 2) when num > 1 do
    assert_key(key)

    impl(key, Nx.iota({num, 2}))
  end

  #Check data requirements
  @spec fold_in(T, integer()) :: T
  def fold_in(key, data) when is_integer(data) do
    assert_key(key)

    impl(key, threefry_seed(data))
  end

  @spec random_bits(T, T) :: T
  def random_bits(key, shape \\ {1}) do
    assert_key(key)

    impl(key, Nx.iota(shape))
  end

  defp impl(key, count, shape \\ {}) do
    assert_key(key)

    shape =
    if shape == {} do
      Nx.shape(count)
    else
      shape
    end

    reshaped_key = Nx.reshape(key, {2, 1})
    reshaped_count =
      Nx.reshape(count, {:auto})
      |> Nx.as_type({:u, 32})

    threefry2x32(reshaped_key, reshaped_count)
    |> Nx.reshape(shape)
  end

  #Check count
  defp threefry2x32(key, count) do

    even? = rem(Nx.axis_size(count, 0), 2) == 0

    if even? do
      count
    else
      Nx.concatenate([Nx.tensor([0]), Nx.flatten(count)])
    end
    |> Nx.reshape({2, :auto})
    |> Nx.as_type({:u, 32})
    |> threefry2x32_20(key)
    |> then(fn output ->
      if even?, do: output, else:
      output
      |> Nx.to_flat_list()
      |> tl()
      |> Nx.tensor()
    end)
  end



  defnp threefry2x32_20(xs, ks) do
    rotations = Nx.tensor([[13, 15, 26, 6], [17, 29, 16, 24]], type: {:u, 8})
    key1 = ks[0]
    key2 = ks[1]

    xs = Nx.add(ks, xs)

    ks = Nx.stack(
      [
      key2,
      Nx.bitwise_xor(key1, key2)
      |> Nx.bitwise_xor(0x1BD11BDA),
      key1
      ]
    )
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

    new_x =
      Nx.stack([xs1, xs2])
      |> Nx.as_type({:u, 32})
    new_k = Nx.stack([k2, k3, k1])
    new_r = Nx.stack([r2, r1])

    {new_x, new_k, new_r}
  end

  defnp rotate_left(x, rot) do
    nbits = 32
    x <<< rot ||| x >>> (nbits - rot)
  end
end
