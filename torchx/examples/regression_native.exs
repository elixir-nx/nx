defmodule Torchx.LinReg do
  import Nx.Defn

  # Data size
  @n 32

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

  defn update({m, b}, inp, tar, step) do
    preds = predict({m, b}, inp)

    errors = Nx.subtract(tar, preds)

    # Derivative m
    grad_m = Nx.mean(Nx.multiply(inp, errors))
    # Derivative b
    grad_b = Nx.mean(errors)

    {Nx.subtract(m, Nx.multiply(grad_m, Nx.tensor(-2 * step))),
     Nx.subtract(b, Nx.multiply(grad_b, Nx.tensor(-2 * step)))}
  end

  def train(params, epochs, lin_fn) do
    data =
      Stream.repeatedly(fn -> for _ <- 1..@n, do: :rand.uniform() * 10 end)
      |> Stream.map(fn x -> Enum.zip(x, Enum.map(x, lin_fn)) end)

    for _ <- 1..epochs, reduce: params do
      acc ->
        data
        |> Enum.take(200)
        |> Enum.reduce(
          acc,
          fn batch, cur_params ->
            {inp, tar} = Enum.unzip(batch)
            x = Nx.reshape(Nx.tensor(inp), {@n, 1})
            y = Nx.reshape(Nx.tensor(tar), {@n, 1})
            update(cur_params, x, y, 0.001)
          end
        )
    end
  end

  def run(epochs \\ 100) do
    Nx.default_backend(Torchx.Backend)
    params = init_random_params()
    m = :rand.normal(0.0, 10.0)
    b = :rand.normal(0.0, 5.0)
    IO.puts("Target m: #{m} Target b: #{b}\n")

    lin_fn = fn x -> m * x + b end

    # These will be very close to the above coefficients
    {trained_m, trained_b} = train(params, epochs, lin_fn)

    trained_m =
      trained_m
      |> Nx.squeeze()
      |> Nx.backend_transfer()
      |> Nx.to_scalar()

    trained_b =
      trained_b
      |> Nx.squeeze()
      |> Nx.backend_transfer()
      |> Nx.to_scalar()

    IO.puts("Trained m: #{trained_m} Trained b: #{trained_b}\n")
  end
end


Torchx.LinReg.run()
