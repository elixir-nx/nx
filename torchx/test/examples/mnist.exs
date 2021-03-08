defmodule Torchx.MNIST do

  def init_random_params do
    w1 = Nx.random_normal({784, 128}, 0.0, 0.1, names: [:input, :layer])
    b1 = Nx.random_normal({128}, 0.0, 0.1, names: [:layer])
    w2 = Nx.random_normal({128, 10}, 0.0, 0.1, names: [:layer, :output])
    b2 = Nx.random_normal({10}, 0.0, 0.1, names: [:output])
    {w1, b1, w2, b2}
  end

  defp softmax(logits) do
    exp = Nx.exp(logits)
    Nx.divide(exp, Nx.sum(exp, axes: [:output], keep_axes: true))
  end

  def predict({w1, b1, w2, b2}, batch) do
    batch
    |> Nx.dot(w1)
    |> Nx.add(b1)
    |> Nx.logistic()
    |> Nx.dot(w2)
    |> Nx.add(b2)
    |> softmax()
  end

  defp accuracy({w1, b1, w2, b2}, batch_images, batch_labels) do
    Nx.mean(
      Nx.equal(
        Nx.argmax(Nx.as_type(batch_labels, {:s, 8}), axis: :output),
        Nx.argmax(predict({w1, b1, w2, b2}, batch_images), axis: :output)
      ) |> Nx.as_type({:s, 8})
    )
  end

  defp loss({w1, b1, w2, b2}, batch_images, batch_labels) do
    preds = predict({w1, b1, w2, b2}, batch_images)
    Nx.sum(Nx.mean(Nx.multiply(Nx.log(preds), batch_labels), axes: [:output])) |> Nx.negate()
  end

  defp update({w1, b1, w2, b2} = params, batch_images, batch_labels, step) do
    # {grad_w1, grad_b1, grad_w2, grad_b2} = grad(params, loss(params, batch_images, batch_labels))


    # preds = predict({w1, b1, w2, b2}, batch_images)

    z1 = Nx.dot(batch_images, w1) |> Nx.add(b1)
    a1 = Nx.logistic(z1)
    z2 = Nx.dot(a1, w2) |> Nx.add(b2)
    preds = a2 = softmax(z2)

    # loss = Nx.sum(Nx.mean(Nx.multiply(Nx.log(preds), batch_labels), axes: [:output])) |> Nx.negate()

    # gradients at last layer (Py2 need 1. to transform to float)
    # dW2 = (1. / m_batch) * np.matmul(dZ2, cache["A1"].T)
    # db2 = (1. / m_batch) * np.sum(dZ2, axis=1, keepdims=True)

    # # back propgate through first layer
    # dA1 = np.matmul(params["W2"].T, dZ2)
    # dZ1 = dA1 * sigmoid(cache["Z1"]) * (1 - sigmoid(cache["Z1"]))

    # # gradients at first layer (Py2 need 1. to transform to float)
    # dW1 = (1. / m_batch) * np.matmul(dZ1, X.T)
    # db1 = (1. / m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    grad_z2 = Nx.subtract(preds, batch_labels)

    m_batch = 30 # Nx.size(grad_z2) |> IO.inspect()

    grad_w2 = Nx.divide(Nx.dot(Nx.transpose(grad_z2), a1), m_batch) |> Nx.transpose()
    grad_b2 = Nx.mean(grad_z2, axes: [:output], keep_axes: true)

    grad_a1 = Nx.dot(w2, Nx.transpose(grad_z2)) |> Nx.transpose()
    grad_z1 = Nx.multiply(grad_a1, Nx.logistic(z1)) |> Nx.multiply(Nx.subtract(1.0, Nx.logistic(z1)))

    grad_w1 = Nx.divide(Nx.dot(Nx.transpose(grad_z1), batch_images), m_batch) |> Nx.transpose()
    grad_b1 = Nx.mean(grad_z1, axes: [1], keep_axes: true)

    {
      Nx.subtract(w1, Nx.multiply(grad_w1, step)),
      Nx.subtract(b1, Nx.multiply(grad_b1, step)),
      Nx.subtract(w2, Nx.multiply(grad_w2, step)),
      Nx.subtract(b2, Nx.multiply(grad_b2, step))
    }
  end

  defp update_with_averages({_, _, _, _} = cur_params, imgs, tar, avg_loss, avg_accuracy, total) do
    batch_loss = loss(cur_params, imgs, tar)
    batch_accuracy = accuracy(cur_params, imgs, tar)
    avg_loss = Nx.add(avg_loss, Nx.divide(batch_loss, total))
    avg_accuracy = Nx.add(avg_accuracy, Nx.divide(batch_accuracy, total))
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

        # epoch_avg_loss =
        #   epoch_avg_loss
        #   |> Nx.backend_transfer(Nx.BinaryBackend)
        #   |> Nx.to_scalar()

        # epoch_avg_acc =
        #   epoch_avg_acc
        #   |> Nx.backend_transfer()
        #   |> Nx.to_scalar()

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

IO.puts("Training MNIST for 10 epochs...\n\n")
final_params = MNIST.train(train_images, train_labels, params, epochs: 3)

IO.puts("Bring the parameters back from the device and print them")
final_params = Nx.backend_transfer(final_params)
IO.inspect(final_params)

# IO.puts("AOT-compiling a trained neural network that predicts a batch")
# Nx.Defn.aot(
#   MNIST.Trained,
#   [{:predict, &MNIST.predict(final_params, &1), [Nx.template({30, 784}, {:f, 32})]}],
#   EXLA
# )

IO.puts("The result of the first batch")
IO.inspect MNIST.predict(final_params, hd(train_images)) |> Nx.argmax(axis: :output)

IO.puts("Labels for the first batch")
IO.inspect hd(train_labels) |> Nx.as_type({:s, 8}) |> Nx.argmax(axis: :output)

# first30 =
#   hd(train_images)
#   |> Nx.reshape({30, 28, 28})
#   |> Nx.backend_transfer(Nx.BinaryBackend)
#   # |> Nx.to_heatmap()

# Nx.default_backend(Nx.BinaryBackend)

# first30
# |> Nx.to_heatmap()
# |> IO.puts()
