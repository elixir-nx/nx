defmodule LinReg do
  # Data size
  @n 32

  # y = mx + b
  def init_random_params do
    m = Nx.random_normal({1, 1}, 0.0, 0.1)
    b = Nx.random_normal({1}, 0.0, 0.1)
    {m, b}
  end

  def predict({m, b}, inp) do
    inp
    |> Nx.dot(m)
    |> Nx.add(b)
  end

  def update({m, b}, inp, tar, step) do
    preds = predict({m, b}, inp)

    errors = Nx.subtract(tar, preds)

    # Derivative wrt m
    grad_m = Nx.mean(Nx.multiply(inp, errors))
    # Derivative wrt b
    grad_b = Nx.mean(errors)

    {Nx.subtract(m, Nx.multiply(grad_m, -2 * step)),
     Nx.subtract(b, Nx.multiply(grad_b, -2 * step))}
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
    params = LinReg.init_random_params()
    m = :rand.normal(0.0, 10.0)
    b = :rand.normal(0.0, 5.0)
    IO.puts("Target m: #{m} Target b: #{b}\n")

    lin_fn = fn x -> m * x + b end

    # These will be very close to the above coefficients
    IO.inspect(LinReg.train(params, epochs, lin_fn))
  end
end
