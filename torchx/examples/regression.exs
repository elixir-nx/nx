defmodule LinReg do
  import Nx.Defn

  # y = mx + b
  defn init_random_params do
    key = Nx.Random.key(42)
    {m, new_key} = Nx.Random.normal(key, 0.0, 0.1, shape: {1, 1})
    {b, _new_key} = Nx.Random.normal(new_key, 0.0, 0.1, shape: {1})
    {m, b}
  end

  defn predict({m, b}, inp) do
    Nx.dot(inp, m) + b
  end

  # MSE Loss
  defn loss({m, b}, inp, tar) do
    preds = predict({m, b}, inp)
    Nx.mean(Nx.pow(tar - preds, 2))
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
            y = Nx.reshape(Nx.tensor(tar), {32, 1})
            update(cur_params, x, y, 0.001)
          end
        )
    end
  end
end

Nx.default_backend(Torchx.Backend)

params = LinReg.init_random_params()
m = :rand.normal(0.0, 10.0)
b = :rand.normal(0.0, 5.0)
IO.puts("Target m: #{m} Target b: #{b}\n")

lin_fn = fn x -> m * x + b end
epochs = 100

# These will be very close to the above coefficients
{time, {trained_m, trained_b}} = :timer.tc(LinReg, :train, [params, epochs, lin_fn])

trained_m =
  trained_m
  |> Nx.squeeze()
  |> Nx.backend_transfer()
  |> Nx.to_number()

trained_b =
  trained_b
  |> Nx.squeeze()
  |> Nx.backend_transfer()
  |> Nx.to_number()

IO.puts("Trained in #{time / 1_000_000} sec.")
IO.puts("Trained m: #{trained_m} Trained b: #{trained_b}\n")
IO.puts("Accuracy m: #{m - trained_m} Accuracy b: #{b - trained_b}")
