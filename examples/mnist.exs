defmodule MNIST do
  import Nx.Defn

  @default_defn_compiler Exla

  defn normalize_batch(batch) do
    Nx.divide(batch, 255)
  end

  defn init_random_params do
    w1 = Nx.random_normal({784, 128}, 0.0, 0.1)
    b1 = Nx.random_normal({128}, 0.0, 0.1)
    w2 = Nx.random_normal({128, 10}, 0.0, 0.1)
    b2 = Nx.random_normal({10}, 0.0, 0.1)
    {w1, b1, w2, b2}
  end

  defn logsumexp(logits) do
    logits
    |> Nx.exp()
    |> Nx.sum(axis: 0)
    |> Nx.log()
  end

  defn predict({w1, b1, w2, b2}, batch) do
    logits =
      batch
      |> Nx.dot(w1)
      |> Nx.add(b1)
      |> Nx.dot(w2)
      |> Nx.add(b2)

    logits - logsumexp(logits)
  end

  defn loss({w1, b1, w2, b2}, batch_images, batch_labels) do
    preds = predict({w1, b1, w2, b2}, batch_images)
    -Nx.divide(Nx.sum(preds * batch_labels, axis: 1), 32)
  end

  defn grad_b1(batch_labels) do
    1 / Nx.log(10.0)
  end

  defn grad_b2(w2, batch_labels) do
    1 / Nx.log(10.0)
  end

  defn grad_w1(w2, batch_images, batch_labels) do
    Nx.dot(Nx.dot(Nx.transpose(batch_images), batch_labels), Nx.transpose(w2)) / Nx.log(10.0)
  end

  defn grad_w2(w1, b1, batch_images, batch_labels) do
    Nx.dot(Nx.transpose(Nx.dot(batch_images, w1) + b1), batch_labels) / Nx.log(10.0)
  end

  defn gradient({w1, b1, w2, b2}, batch_images, batch_labels) do
    grad_b1 = grad_b1(batch_labels)
    grad_b2 = grad_b2(w2, batch_labels)
    grad_w1 = grad_w1(w2, batch_images, batch_labels)
    grad_w2 = grad_w2(w1, b1, batch_images, batch_labels)
    {grad_w1, grad_b1, grad_w2, grad_b2}
  end

  defn update_param(param, grad_val, step) do
    param - step * grad_val
  end

  defn update({w1, b1, w2, b2}, batch_images, batch_labels, step) do
    {grad_w1, grad_b1, grad_w2, grad_b2} = gradient({w1, b1, w2, b2}, batch_images, batch_labels)
    {
      update_param(w1, grad_w1, step),
      update_param(b1, grad_b1, step),
      update_param(w2, grad_w2, step),
      update_param(b2, grad_b2, step)
    }
  end

  def normalize_images(images) do
    images
    |> Enum.map(&Nx.tensor(&1, type: {:u, 8}))
    |> Enum.map(&normalize_batch/1)
  end

  def normalize_labels(labels) do
    labels
    |> Enum.map(&Nx.tensor(&1, type: {:u, 8}))
  end

  def to_one_hot(x) do
    for i <- 0..9, do: if x == i, do: 1, else: 0
  end

  def download(images, labels) do
    base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

    :inets.start()
    :ssl.start()

    {:ok, {_status, _response, train_image_data}} = :httpc.request(:get, {base_url++images, []}, [], [])
    {:ok, {_status, _response, train_label_data}} = :httpc.request(:get, {base_url++labels, []}, [], [])

    <<mw::32, n_numbers::32, n_rows::32, n_cols::32, images::bitstring>> = train_image_data |> :zlib.gunzip()
    train_images =
      images
      |> :binary.bin_to_list()
      |> Enum.chunk_every(784)
      |> Enum.chunk_every(32)

    IO.inspect Enum.count(train_images)

    train_labels =
      train_label_data
      |> :zlib.gunzip()
      |> :binary.bin_to_list()
      |> Enum.drop(8)
      |> Enum.map(&to_one_hot/1)
      |> Enum.chunk_every(32)

    {train_images, train_labels}
  end

  def train(imgs, labels, params, opts \\ []) do
    epochs = opts[:epochs] || 5
    for epoch <- 1..epochs do
      {{w1, b1, w2, b2}, 0} =
        imgs
        |> Enum.zip(labels)
        |> Enum.shuffle()
        |> Enum.reduce({params, 0} fn {imgs, tar}, {cur_params, batch} ->
            IO.puts("Batch: #{batch}")
            IO.inspect loss(cur_params, imgs, tar)
            {update(cur_params, imgs, tar, 0.001), batch+1}
          end
          )

      {val_img, val_lab} = imgs |> Enum.zip(labels) |> Enum.random()
      val_loss = loss({w1, b1, w2, b2}, val_img, val_lab)
      IO.puts("Epoch: #{epoch}\tLoss: #{val_loss}\n")
      val_loss
    end
  end
end

IO.puts("Downloading dataset...\n")
{train_images, train_labels} = MNIST.download('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')

IO.puts("Normalizing images and labels...\n")
{train_images, train_labels} = {MNIST.normalize_images(train_images), MNIST.normalize_labels(train_labels)}

IO.inspect train_labels

IO.inspect Enum.take(train_images, 1)
IO.puts("Initializing parameters...\n")
params = MNIST.init_random_params()

IO.inspect params

IO.puts("Training MNIST for 10 epochs...\n\n")
IO.inspect MNIST.train(train_images, train_labels, params)