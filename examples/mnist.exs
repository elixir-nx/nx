defmodule MNIST do
  import Nx.Defn

  @default_defn_compiler Exla

  defn normalize_batch(batch) do
    batch / 255.0
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
    |> Nx.sum(axis: 1)
    |> Nx.log()
  end

  defn predict({w1, b1, w2, b2}, batch) do
    logits =
      batch
      |> Nx.dot(w1)
      |> Nx.add(b1)
      |> Nx.tanh()
      |> Nx.dot(w2)
      |> Nx.add(b2)
      |> Nx.tanh()

    logits - logsumexp(logits)
  end

  defn accuracy({w1, b1, w2, b2}, batch_images, batch_labels) do
    targets = Nx.argmax(batch_labels, axis: 1)
    preds = Nx.argmax(predict({w1, b1, w2, b2}, batch_images), axis: 1)
    Nx.mean(Nx.equal(targets, preds))
  end

  defn loss({w1, b1, w2, b2}, batch_images, batch_labels) do
    preds = predict({w1, b1, w2, b2}, batch_images)
    -Nx.mean(Nx.sum(preds * batch_labels, axis: 1))
  end

  defn update_param(param, grad_val, step) do
    param - step * grad_val
  end

  defn update({w1, b1, w2, b2}, batch_images, batch_labels, step) do
    {grad_w1, grad_b1, grad_w2, grad_b2} = grad({w1, b1, w2, b2}, loss({w1, b1, w2, b2}, batch_images, batch_labels))
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

    # Download train images
    {:ok, {_status, _response, train_image_data}} = :httpc.request(:get, {base_url++images, []}, [], [])

    <<_::32, n_images::32, n_rows::32, n_cols::32, images::bitstring>> = train_image_data |> :zlib.gunzip()
    IO.puts("Downloaded #{n_images} #{n_rows}x#{n_cols} images\n")

    train_images =
      images
      |> :binary.bin_to_list()
      |> Enum.chunk_every(784)
      |> Enum.chunk_every(32)

    # Download train labels
    {:ok, {_status, _response, train_label_data}} = :httpc.request(:get, {base_url++labels, []}, [], [])

    <<_::32, n_labels::32, labels::bitstring>> = train_label_data |> :zlib.gunzip()
    IO.puts("Downloaded #{n_labels} labels")

    train_labels =
      labels
      |> :binary.bin_to_list()
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
        |> Enum.reduce({params, 0}, fn {imgs, tar}, {cur_params, batch} ->
            IO.puts("Batch: #{batch}\n")
            IO.puts("Loss: ")
            IO.inspect loss(cur_params, imgs, tar)
            IO.puts("Accuracy: ")
            IO.inspect accuracy(cur_params, imgs, tar)

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

IO.puts("Initializing parameters...\n")
params = MNIST.init_random_params()

IO.puts("Training MNIST for 10 epochs...\n\n")
IO.inspect MNIST.train(train_images, train_labels, params)