defmodule Torchx.MNIST do
  import Nx.Defn

  @batch_size 30
  @step 0.01
  @epochs 10

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

  defn update(
         {w1, b1, w2, b2} = _params,
         batch_images,
         batch_labels,
         avg_loss,
         avg_accuracy,
         total
       ) do
    z1 = Nx.dot(batch_images, w1) |> Nx.add(b1)
    a1 = Nx.logistic(z1)
    z2 = Nx.dot(a1, w2) |> Nx.add(b2)
    preds = softmax(z2)

    batch_loss =
      Nx.sum(Nx.mean(Nx.multiply(Nx.log(preds), batch_labels), axes: [:output])) |> Nx.negate()

    batch_accuracy =
      Nx.mean(
        Nx.equal(
          Nx.argmax(batch_labels, axis: :output),
          Nx.argmax(preds, axis: :output)
        )
        |> Nx.as_type({:s, 8})
      )

    total = Nx.tensor(total)

    avg_loss = Nx.add(avg_loss, Nx.divide(batch_loss, total))
    avg_accuracy = Nx.add(avg_accuracy, Nx.divide(batch_accuracy, total))

    grad_z2 = Nx.subtract(preds, batch_labels) |> Nx.transpose()

    batch_size = Nx.tensor(@batch_size)

    grad_w2 = Nx.divide(Nx.dot(grad_z2, a1), batch_size) |> Nx.transpose()
    grad_b2 = Nx.mean(Nx.transpose(grad_z2), axes: [:output], keep_axes: true)

    grad_a1 = Nx.dot(w2, grad_z2) |> Nx.transpose()
    grad_z1 = Nx.multiply(grad_a1, a1) |> Nx.multiply(Nx.subtract(Nx.tensor(1.0), a1))

    grad_w1 = Nx.divide(Nx.dot(Nx.transpose(grad_z1), batch_images), batch_size) |> Nx.transpose()
    grad_b1 = Nx.mean(grad_z1, axes: [1], keep_axes: true)

    step = Nx.tensor(@step)

    {{
       Nx.subtract(w1, Nx.multiply(grad_w1, step)),
       Nx.subtract(b1, Nx.multiply(grad_b1, step)),
       Nx.subtract(w2, Nx.multiply(grad_w2, step)),
       Nx.subtract(b2, Nx.multiply(grad_b2, step))
     }, avg_loss, avg_accuracy}
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
      |> Nx.divide(Nx.tensor(255))
      |> Nx.to_batched_list(@batch_size)

    IO.puts("#{n_images} #{n_rows}x#{n_cols} images\n")

    <<_::32, n_labels::32, labels::binary>> = unzip_cache_or_download(labels)

    train_labels =
      labels
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n_labels, 1}, names: [:batch, :output])
      |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
      |> Nx.as_type({:s, 8})
      |> Nx.to_batched_list(@batch_size)

    IO.puts("#{n_labels} labels\n")

    {train_images, train_labels}
  end

  def train_epoch(cur_params, imgs, labels) do
    total_batches = Enum.count(imgs)

    imgs
    |> Enum.zip(labels)
    |> Enum.reduce({cur_params, Nx.tensor(0.0), Nx.tensor(0.0)}, fn
      {imgs, tar}, {cur_params, avg_loss, avg_accuracy} ->
        update(cur_params, imgs, tar, avg_loss, avg_accuracy, total_batches)
    end)
  end

  def train(imgs, labels, params, opts \\ []) do
    epochs = opts[:epochs] || @epochs

    IO.puts("Training MNIST for #{epochs} epochs...\n\n")

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

alias Torchx.MNIST
Nx.default_backend(Torchx.Backend)

{train_images, train_labels} =
  MNIST.download('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')

IO.puts("Initializing parameters...\n")
params = MNIST.init_random_params()

final_params = MNIST.train(train_images, train_labels, params)

IO.puts("The result of the first batch")
IO.inspect(MNIST.predict(final_params, hd(train_images)) |> Nx.argmax(axis: :output))

IO.puts("Labels for the first batch")
IO.inspect(hd(train_labels) |> Nx.argmax(axis: :output))
