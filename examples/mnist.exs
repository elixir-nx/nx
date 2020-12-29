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

  defn softmax(logits) do
    Nx.exp(logits) / Nx.reshape(Nx.sum(Nx.exp(logits), axes: [1]), {32, 1})
  end

  defn predict({w1, b1, w2, b2}, batch) do
    batch
    |> Nx.dot(w1)
    |> Nx.add(b1)
    |> Nx.logistic()
    |> Nx.dot(w2)
    |> Nx.add(b2)
    |> softmax()
  end

  defn accuracy({w1, b1, w2, b2}, batch_images, batch_labels) do
    # Nx.mean(Nx.equal(Nx.argmax(batch_labels, axis: 1), Nx.argmax(predict({w1, b1, w2, b2}, batch_images), axis: 1)))
    {Nx.argmax(predict({w1, b1, w2, b2}, batch_images), axis: 1), Nx.argmax(batch_labels, axis: 1)}
  end

  defn loss({w1, b1, w2, b2}, batch_images, batch_labels) do
    preds = predict({w1, b1, w2, b2}, batch_images)
    -Nx.mean(Nx.sum(Nx.log(preds) * batch_labels, axes: [1]))
  end

  defn update({w1, b1, w2, b2}, batch_images, batch_labels, step) do
    {grad_w1, grad_b1, grad_w2, grad_b2} = grad({w1, b1, w2, b2}, loss({w1, b1, w2, b2}, batch_images, batch_labels))
    {
      w1 - (grad_w1 * step),
      b1 - (grad_b1 * step),
      w2 - (grad_w2 * step),
      b2 - (grad_b2 * step)
    }
  end

  defn average(cur_avg, batch, total) do
    cur_avg + (batch / total)
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
    for epoch <- 1..epochs, reduce: {params, Nx.tensor(0.0), Nx.tensor(0.0)} do
      acc ->
        total_batches = Enum.count(imgs)
        {new_params, epoch_avg_loss, epoch_avg_acc} =
          imgs
          |> Enum.zip(labels)
          |> Enum.reduce(acc, fn {imgs, tar}, {cur_params, avg_loss, avg_accuracy} ->
              batch_loss = loss(cur_params, imgs, tar)
              batch_accuracy = accuracy(cur_params, imgs, tar)
              IO.inspect batch_accuracy
              IO.inspect Nx.argmax(predict(cur_params, imgs), axis: 1)
              IO.inspect Nx.argmax(tar, axis: 1)
              avg_loss = average(avg_loss, batch_loss, total_batches)
              avg_accuracy = average(avg_accuracy, batch_accuracy, total_batches)
              {update(cur_params, imgs, tar, 0.01), avg_loss, avg_accuracy}
            end
            )

        IO.puts("Epoch #{epoch} average loss: #{inspect(epoch_avg_loss)}")
        IO.puts("Epoch #{epoch} average accuracy: #{inspect(epoch_avg_acc)}")
        {new_params, Nx.tensor(0.0), Nx.tensor(0.0)}
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
{final_params, _, _} = MNIST.train(train_images, train_labels, params, epochs: 10)

IO.inspect final_params