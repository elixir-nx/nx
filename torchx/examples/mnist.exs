defmodule MNIST do
  import Nx.Defn

  defn init_random_params do
    w1 = Nx.random_normal({784, 128}, 0.0, 0.1, names: [:input, :layer])
    b1 = Nx.random_normal({128}, 0.0, 0.1, names: [:layer])
    w2 = Nx.random_normal({128, 10}, 0.0, 0.1, names: [:layer, :output])
    b2 = Nx.random_normal({10}, 0.0, 0.1, names: [:output])
    {w1, b1, w2, b2}
  end

  defn softmax(logits) do
    exp = Nx.exp(logits)
    Nx.divide(exp, Nx.sum(exp, axes: [:output], keep_axes: true))
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
    Nx.mean(
      Nx.equal(
        Nx.argmax(batch_labels, axis: :output),
        Nx.argmax(predict({w1, b1, w2, b2}, batch_images), axis: :output)
      )
      |> Nx.as_type({:s, 8})
    )
  end

  defn loss({w1, b1, w2, b2}, batch_images, batch_labels) do
    preds = predict({w1, b1, w2, b2}, batch_images)
    -Nx.sum(Nx.mean(Nx.log(preds) * batch_labels, axes: [:output]))
  end

  defn update({w1, b1, w2, b2} = params, batch_images, batch_labels, step) do
    {grad_w1, grad_b1, grad_w2, grad_b2} = grad(params, loss(params, batch_images, batch_labels))

    {
      w1 - grad_w1 * step,
      b1 - grad_b1 * step,
      w2 - grad_w2 * step,
      b2 - grad_b2 * step
    }
  end

  defn update_with_averages({_, _, _, _} = cur_params, imgs, tar, avg_loss, avg_accuracy, total) do
    batch_loss = loss(cur_params, imgs, tar)
    batch_accuracy = accuracy(cur_params, imgs, tar)
    avg_loss = avg_loss + batch_loss / total
    avg_accuracy = avg_accuracy + batch_accuracy / total
    {update(cur_params, imgs, tar, 0.01), avg_loss, avg_accuracy}
  end

  defp unzip_cache_or_download(zip) do
    base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    path = Path.join("tmp", zip)

    data =
      if File.exists?(path) do
        IO.puts("Using #{zip} from tmp/\n")
        File.read!(path)
      else
        IO.puts("Fetching #{zip} from https://storage.googleapis.com/cvdf-datasets/mnist/\n")
        :inets.start()
        :ssl.start()

        {:ok, {_status, _response, data}} = :httpc.request(:get, {base_url ++ zip, []}, [], [])
        File.mkdir_p!("tmp")
        File.write!(path, data)

        data
      end

    :zlib.gunzip(data)
  end

  def download(images, labels) do
    <<_::32, n_images::32, n_rows::32, n_cols::32, images::binary>> =
      unzip_cache_or_download(images)

    train_images =
      images
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n_images, n_rows * n_cols}, names: [:batch, :input])
      |> Nx.divide(255)
      |> Nx.to_batched_list(30)

    IO.puts("#{n_images} #{n_rows}x#{n_cols} images\n")

    <<_::32, n_labels::32, labels::binary>> = unzip_cache_or_download(labels)

    train_labels =
      labels
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n_labels, 1}, names: [:batch, :output])
      |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
      |> Nx.as_type({:s, 8})
      |> Nx.to_batched_list(30)

    IO.puts("#{n_labels} labels\n")

    {train_images, train_labels}
  end

  def train_epoch(cur_params, imgs, labels) do
    total_batches = Enum.count(imgs)

    imgs
    |> Enum.zip(labels)
    |> Enum.reduce({cur_params, Nx.tensor(0.0), Nx.tensor(0.0)}, fn
      {imgs, tar}, {cur_params, avg_loss, avg_accuracy} ->
        update_with_averages(cur_params, imgs, tar, avg_loss, avg_accuracy, total_batches)
    end)
  end

  def train(imgs, labels, params, opts \\ []) do
    epochs = opts[:epochs] || 5

    for epoch <- 1..epochs, reduce: params do
      cur_params ->
        {time, {new_params, epoch_avg_loss, epoch_avg_acc}} =
          :timer.tc(__MODULE__, :train_epoch, [cur_params, imgs, labels])

        epoch_avg_loss =
          epoch_avg_loss
          |> Nx.backend_transfer()
          |> Nx.to_scalar()

        epoch_avg_acc =
          epoch_avg_acc
          |> Nx.backend_transfer()
          |> Nx.to_scalar()

        IO.puts("Epoch #{epoch} Time: #{time / 1_000_000}s")
        IO.puts("Epoch #{epoch} average loss: #{inspect(epoch_avg_loss)}")
        IO.puts("Epoch #{epoch} average accuracy: #{inspect(epoch_avg_acc)}")
        IO.puts("\n")
        new_params
    end
  end
end

Nx.default_backend(Torchx.Backend)

{train_images, train_labels} =
  MNIST.download('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')

IO.puts("Initializing parameters...\n")
params = MNIST.init_random_params()

IO.puts("Training MNIST for 10 epochs...\n\n")
final_params = MNIST.train(train_images, train_labels, params, epochs: 10)

IO.puts("The result of the first batch")
IO.inspect(MNIST.predict(final_params, hd(train_images)) |> Nx.argmax(axis: :output))

IO.puts("Labels for the first batch")
IO.inspect(hd(train_labels) |> Nx.argmax(axis: :output))
