defmodule Nx.Random do
  @moduledoc """
  Nx conveniences for random number generator.
  """

  import Nx.Shared
  import Nx.Defn, only: [defn: 2, defnp: 2]
  import Nx.Defn.Kernel, only: [assert_shape: 2]

  alias Nx.Tensor, as: T

  #defguardp is_threefry_key(tensor)
  #         when assert_shape(tensor, {2}) and Nx.Type.integer?(Nx.type(tensor))

  defp assert_key(tensor) do
    assert_shape(tensor, {2, 1})
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
    Nx.stack([k1, k2])
    |> Nx.as_type({:u, 32})
  end

  @spec threefry_split(T, pos_integer()) :: [T]
  def threefry_split(key, num) do
    assert_key(key)

    key
    |> threefry2x32(Nx.iota({num * 2}))
    |> Nx.reshape({:auto, 2})
    #|> Enum.chunk_every(2)
  end

  #Check data requirements
  @spec threefry_fold_in(PRNG.prng_key(), integer()) :: PRNG.prng_key()
  def threefry_fold_in(key, data) do
    assert_key(key)


    threefry2x32(key, threefry_seed(data))
  end

  @spec threefry_random_bits(T, pos_integer()) :: [T]
  def threefry_random_bits(key, num \\ 1) when num >= 1 do
    assert_key(key)

    threefry2x32(key, iota(num))
  end

  #Check count
  defp threefry2x32(key, count) do
    assert_key(key)

    even? = rem(Nx.axis_size(count, 0), 2) == 0
    IO.inspect(even?)
    if even? do
      Nx.flatten(count)
    else
      Nx.concatenate([Nx.tensor(0), Nx.flatten(count)])
    end
    |> Nx.reshape({2, :auto})
    |> Nx.as_type({:u, 32})
    |> threefry2x32_20(key)
    |> then(fn output ->
      if even?, do: output, else:
      output
      |> Nx.to_flat_list()
      |> tl()
      |> Nx.to_tensor()
    end)
  end

  defnp threefry2x32_20(xs, ks) do
    rotations = Nx.tensor([[13, 15, 26, 6], [17, 29, 16, 24]], type: {:u, 8})
    key1 = ks[0][0]
    key2 = ks[1][0]
    #[key1, key2] = Nx.reshape(ks, {1,2}) |> Nx.to_flat_list()
    xs = Nx.broadcast(ks, xs)
    |> Nx.add(xs)
    #|> Nx.as_type({:u, 32})

    #xs = Enum.zip_with([xs, ks], fn [x, k] -> add_to_list(x, k) end)
    ks = Nx.stack(
      [
      key2,
      Nx.bitwise_xor(key1, key2)
      |> Nx.bitwise_xor(0x1BD11BDA),
      key1
      #|> Nx.to_number(),
      ]
    )
    |> Nx.as_type({:u, 32})

    state = {xs, ks, rotations}
    #state = Nx.concatenate([Nx.stack([xs[0], xs[1],Nx.tensor([0,0])]), ks, rotations], axis: 1)
    #IO.inspect(state)

    #transform(Nx.shape(state), &IO.inspect/1)

    {_, {nxs, _, _}} =
    while {x = 0, state}, Nx.less(x, 5) do
      {x + 1, rolled_loop_step(x, state)}
    end
    nxs
    # iota(5)
    # |> Enum.reduce(state, &rolled_loop_step/2)
    # |> hd()
    # |> Nx.flatten()
    # |> Nx.as_type({:u, 32})
    #transform(state, &IO.inspect/1)

  end

  defnp apply_round(xs, rot) do
    y1 =
      Nx.add(xs[0], xs[1])
      |> Nx.as_type({:u, 32})



    y2 =
      rotate_left(xs[1], rot)
      |> Nx.bitwise_xor(y1)
      |> Nx.as_type({:u, 32})

    #IO.inspect([y1, y2])
    Nx.stack([y1, y2])
  end
#[xs, _ks = [k1, k2, k3], _rotations = [r1, r2]]
  defnp rolled_loop_step(i, {xs, ks, rs}) do
    #xs = state[0]
    #ks = state[1]
    #rs = state[2]

    #transform(Nx.shape(state), &IO.inspect/1)

    {k1, k2, k3} = {ks[0], ks[1], ks[2]}
    {r1, r2} = {rs[0], rs[1]}


    #IO.inspect(xs)
    #xs = Enum.reduce(r1, xs, &apply_round(&2, &1))
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
    new_k =
      Nx.stack([k2, k3, k1])
      |> Nx.as_type({:u, 32})
    new_r =
      Nx.stack([r2, r1])
      |> Nx.as_type({:u, 8})

    {new_x, new_k, new_r}
  end

  defnp rotate_left(x, rot) do
    nbits = 32
    x <<< rot ||| x >>> (nbits - rot)
  end

  defp iota(num) do
    Enum.to_list(0..(num - 1))
  end


end
