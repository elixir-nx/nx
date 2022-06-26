defmodule LinReg do
  import Nx.Defn

  # y = mx + b
  defn init_random_params do
    m = Nx.random_normal({1, 1}, 0.0, 0.1)
    b = Nx.random_normal({1}, 0.0, 0.1)
    {m, b}
  end

  defn predict({m, b}, inp) do
    inp
    |> Nx.dot(m)
    |> Nx.add(b)
  end

  # MSE Loss
  defn loss({m, b}, inp, tar) do
    preds = predict({m, b}, inp)
    Nx.mean(Nx.power(tar - preds, 2))
  end

  defn update({m, b} = params, inp, tar, step) do
    {grad_m, grad_b} = grad(params, &loss(&1, inp, tar))
    {m - grad_m * step, b - grad_b * step}
  end

  def train(params, epochs, lin_fn) do
    data =
      Stream.repeatedly(fn -> for _ <- 1..32, do: :rand.uniform() * 10 end)
      |> Stream.map(fn x -> Enum.zip(x, Enum.map(x, lin_fn)) end)

    for _ <- 1..epochs, reduce: params do
      acc ->
        data
        |> Enum.take(200)
        |> Enum.reduce(
          acc,
          fn batch, cur_params ->
            {inp, tar} = Enum.unzip(batch)
            x = Nx.reshape(Nx.tensor(inp), {32, 1})
            y = Nx.reshape(Nx.tensor(tar), {1, 32})
            update(cur_params, x, y, 0.001)
          end
        )
    end
  end
end

params = LinReg.init_random_params()
m = :rand.normal(0.0, 10.0)
b = :rand.normal(0.0, 5.0)
IO.puts("Target m: #{m} Target b: #{b}\n")

lin_fn = fn x -> m * x + b end
epochs = 100

# These will be very close to the above coefficients
IO.inspect(EXLA.jit(&LinReg.train/3).(params, epochs, lin_fn))
