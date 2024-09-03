defmodule Nx.Serving do
  @moduledoc """
  Serving encapsulates client and server work to perform batched requests.

  Servings can be executed on the fly, without starting a server, but most
  often they are used to run servers that batch requests until a given size
  or timeout is reached.

  More specifically, servings are a mechanism to apply a computation on a
  `Nx.Batch`, with hooks for preprocessing input from and postprocessing
  output for the client. Thus we can think of an instance of `t:Nx.Serving.t/0`
  (a serving) as something that encapsulates batches of Nx computations.

  ## Inline/serverless workflow

  First, let's define a simple numerical definition function:

      defmodule MyDefn do
        import Nx.Defn

        defn print_and_multiply(x) do
          x = print_value(x, label: "debug")
          x * 2
        end
      end

  The function prints the given tensor and doubles its contents.
  We can use `new/1` to create a serving that will return a JIT
  or AOT compiled function to execute on batches of tensors:

      iex> serving = Nx.Serving.new(fn opts -> Nx.Defn.jit(&MyDefn.print_and_multiply/1, opts) end)
      iex> batch = Nx.Batch.stack([Nx.tensor([1, 2, 3])])
      iex> Nx.Serving.run(serving, batch)
      debug: #Nx.Tensor<
        s64[1][3]
        [
          [1, 2, 3]
        ]
      >
      #Nx.Tensor<
        s64[1][3]
        [
          [2, 4, 6]
        ]
      >

  We started the serving by passing a function that receives
  compiler options and returns a JIT or AOT compiled function.
  We called `Nx.Defn.jit/2` passing the options received as
  argument, which will customize the JIT/AOT compilation.

  You should see two values printed. The former is the result of
  `Nx.Defn.Kernel.print_value/1`, which shows the tensor that was
  actually part of the computation and how it was batched.
  The latter is the result of the computation.

  When defining a `Nx.Serving`, we can also customize how the data is
  batched by using the `client_preprocessing` as well as the result by
  using `client_postprocessing` hooks. Let's give it another try,
  this time using `jit/2` to create the serving, which automatically
  wraps the given function in `Nx.Defn.jit/2` for us:

      iex> serving = (
      ...>   Nx.Serving.jit(&MyDefn.print_and_multiply/1)
      ...>   |> Nx.Serving.client_preprocessing(fn input -> {Nx.Batch.stack(input), :client_info} end)
      ...>   |> Nx.Serving.client_postprocessing(&{&1, &2})
      ...> )
      iex> Nx.Serving.run(serving, [Nx.tensor([1, 2]), Nx.tensor([3, 4])])
      debug: #Nx.Tensor<
        s64[2][2]
        [
          [1, 2],
          [3, 4]
        ]
      >
      {{#Nx.Tensor<
          s64[2][2]
          [
            [2, 4],
            [6, 8]
          ]
        >,
        :server_info},
       :client_info}

  You can see the results are a bit different now. First of all, notice that
  we were able to run the serving passing a list of tensors. Our custom
  `client_preprocessing` function stacks those tensors into a batch of two
  entries and returns a tuple with a `Nx.Batch` struct and additional client
  information which we represent as the atom `:client_info`. The default
  client preprocessing simply enforces a batch (or a stream of batches)
  was given and returns no client information.

  Then the result is a triplet tuple, returned by the client
  postprocessing function, containing the result, the server information
  (which we will later learn how to customize), and the client information.
  From this, we can infer the default implementation of `client_postprocessing`
  simply returns the result, discarding the server and client information.

  So far, `Nx.Serving` has not given us much. It has simply encapsulated the
  execution of a function. Its full power comes when we start running our own
  `Nx.Serving` process. That's when we will also learn why we have a `client_`
  prefix in some of the function names.

  ## Stateful/process workflow

  `Nx.Serving` allows us to define an Elixir process to handle requests.
  This process provides several features, such as batching up to a given
  size or time, partitioning, and distribution over a group of nodes.

  To do so, we need to start a `Nx.Serving` process with a serving inside
  a supervision tree:

      children = [
        {Nx.Serving,
         serving: Nx.Serving.jit(&MyDefn.print_and_multiply/1),
         name: MyServing,
         batch_size: 10,
         batch_timeout: 100}
      ]

      Supervisor.start_child(children, strategy: :one_for_one)

  > Note: in your actual application, you want to make sure
  > `Nx.Serving` comes early in your supervision tree, for example
  > before your web application endpoint or your data processing
  > pipelines, as those processes may end-up hitting Nx.Serving.

  Now you can send batched runs to said process:

      iex> batch = Nx.Batch.stack([Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])])
      iex> Nx.Serving.batched_run(MyServing, batch)
      debug: #Nx.Tensor<
        s64[2][3]
        [
          [1, 2, 3],
          [4, 5, 6]
        ]
      >
      #Nx.Tensor<
        s64[2][3]
        [
          [2, 4, 6],
          [8, 10, 12]
        ]
      >

  In the example, we pushed a batch of 2 and eventually got a reply.
  The process will wait for requests from other processes, for up to
  100 milliseconds or until it gets 10 entries. Then it merges all
  batches together and once the result is computed, it slices and
  distributes those responses to each caller.

  If there is any `client_preprocessing` function, it will be executed
  before the batch is sent to the server. If there is any `client_postprocessing`
  function, it will be executed after getting the response from the
  server.

  ### Partitioning

  You can start several partitions under the same serving by passing
  `partitions: true` when starting the serving. The number of partitions
  will be determined according your compiler and for which host it is
  compiling.

  For example, when creating the serving, you may pass the following
  `defn_options`:

      Nx.Serving.new(computation, compiler: EXLA, client: :cuda)

  Now when booting up the serving:

      children = [
        {Nx.Serving,
         serving: serving,
         name: MyServing,
         batch_size: 10,
         batch_timeout: 100,
         partitions: true}
      ]

  If you have two GPUs, `batched_run/3` will now gather batches and send
  them to the GPUs as they become available to process requests.

  > #### Cross-device operations {: .warning}
  >
  > When `partitions: true` is set, you will receive results from
  > different GPU devices and Nx won't automatically transfer data
  > across devices to avoid surprising performance pitfalls, which
  > may lead to errors. In such cases, you probably want to transfer
  > tensors back to host on your serving execution.

  ### Distribution

  All `Nx.Serving`s are distributed by default. If the current machine
  does not have an instance of `Nx.Serving` running, `batched_run/3` will
  automatically look for one in the cluster. The nodes do not need to run
  the same code and applications. It is only required that they run the
  same `Nx` version.

  The load balancing between servings is done randomly by default, however,
  the number of partitions are considered if the `partitions: true` option is also given.
  For example, if you have a node with 2 GPUs and another with 4, the latter
  will receive the double of requests compared to the former.

  Furthermore, the load balancing allows for assigning weights to servings.
  Similarly to the number of partitions, when running a serving with `distribution_weight: 1`
  and another one with `distribution_weight: 2`, the latter will receive double the requests
  compared to the former.

  `batched_run/3` receives an optional `distributed_preprocessing` callback as
  third argument for preprocessing the input for distributed requests. When
  using libraries like EXLA or Torchx, the tensor is often allocated in memory
  inside a third-party library so it is necessary to either transfer or copy
  the tensor to the binary backend before sending it to another node.
  This can be done by passing either `Nx.backend_transfer/1` or `Nx.backend_copy/1`
  as third argument:

      Nx.Serving.batched_run(MyDistributedServing, input, &Nx.backend_copy(&1, Nx.BinaryBackend))

  Use `backend_transfer/1` if you know the input will no longer be used.

  Similarly, the serving has a `distributed_postprocessing` callback which is
  called on the remote machine before sending the reply to the caller. It can
  be used to transfer resources to the binary backend before sending them over
  the network.

  The servings are dispatched using Erlang Distribution. You can use
  `Node.connect/1` to manually connect nodes. In a production setup, this is
  often done with the help of libraries like [`libcluster`](https://github.com/bitwalker/libcluster).

  ## Advanced notes

  ### Module-based serving

  In the examples so far, we have been using the default version of
  `Nx.Serving`, which executes the given function for each batch.

  However, we can also use `new/2` to start a module-based version of
  `Nx.Serving` which gives us more control over both inline and process
  workflows. A simple module implementation of a `Nx.Serving` could look
  like this:

      defmodule MyServing do
        @behaviour Nx.Serving

        defnp print_and_multiply(x) do
          x = print_value({:debug, x})
          x * 2
        end

        @impl true
        def init(_inline_or_process, :unused_arg, [defn_options]) do
          {:ok, Nx.Defn.jit(&print_and_multiply/1, defn_options)}
        end

        @impl true
        def handle_batch(batch, 0, function) do
          {:execute, fn -> {function.(batch), :server_info} end, function}
        end
      end

  It has two functions. The first, `c:init/3`, receives the type of serving
  (`:inline` or `:process`) and the serving argument. In this step,
  we capture `print_and_multiply/1`as a jitted function.

  The second function is called `c:handle_batch/3`. This function
  receives a `Nx.Batch` and returns a function to execute.
  The function itself must return a two element-tuple: the batched
  results and some server information. The server information can
  be any value and we set it to the atom `:server_info`.

  Now let's give it a try by defining a serving with our module and
  then running it on a batch:

      iex> serving = Nx.Serving.new(MyServing, :unused_arg)
      iex> batch = Nx.Batch.stack([Nx.tensor([1, 2, 3])])
      iex> Nx.Serving.run(serving, batch)
      {:debug, #Nx.Tensor<
        s64[1][3]
        [
          [1, 2, 3]
        ]
      >}
      #Nx.Tensor<
        s64[1][3]
        [
          [2, 4, 6]
        ]
      >

  From here on, you use `start_link/1` to start this serving in your
  supervision and even customize `client_preprocessing/1` and
  `client_postprocessing/1` callbacks to this serving, as seen in the
  previous sections.

  Note in our implementation above assumes it won't run partitioned.
  In partitioned mode, `c:init/3` may receive multiple `defn_options`
  as the third argument and `c:handle_batch/3` may receive another partition
  besides 0.

  ### Streaming

  `Nx.Serving` allows both inputs and outputs to be streamed.

  In order to stream inputs, you only need to return a stream of `Nx.Batch`
  from the `client_preprocessing` callback. Serving will automatically take
  care of streaming the inputs in, regardless if using `run/2` or `batched_run/2`.
  It is recommended that the streaming batches have the same size as `batch_size`,
  to avoid triggering `batch_timeout` on every iteration (except for the last one
  which may be incomplete).

  To stream outputs, you must invoke `streaming/2` with any additional
  streaming configuration. When this is invoked, the `client_postprocessing`
  will receive a stream which you can further manipulate lazily using the
  functions in the `Stream` module. `streaming/2` also allows you to configure
  hooks and stream values directly from `Nx.Defn` hooks. However, when hook
  streaming is enabled, certain capabilities are removed: you cannot stream
  inputs nor have batches larger than the configured `batch_size`.

  You can enable both input and output streaming at once.

  ### Batch keys

  Sometimes it may be necessary to execute different functions under the
  same serving. For example, sequence transformers must pad the sequence
  to a given length. However, if you are batching, the length must be
  padded upfront. If the length is too small, you have to either discard
  data or support only small inputs. If the length is too large, then you
  decrease performance with the extra padding.

  Batch keys provide a mechanism to accumulate different batches, based on
  their key, which execute independently. As an example, we will do a
  serving which performs different operations based on the batch key,
  but it could also be used to perform the same operation for different
  templates:

      iex> args = [Nx.template({10}, :s32)]
      iex> serving = Nx.Serving.new(fn
      ...>   :double, opts -> Nx.Defn.compile(&Nx.multiply(&1, 2), args, opts)
      ...>   :half, opts -> Nx.Defn.compile(&Nx.divide(&1, 2), args, opts)
      ...> end)
      iex> double_batch = Nx.Batch.concatenate([Nx.iota({10})]) |> Nx.Batch.key(:double)
      iex> Nx.Serving.run(serving, double_batch)
      #Nx.Tensor<
        s32[10]
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
      >
      iex> half_batch = Nx.Batch.concatenate([Nx.iota({10})]) |> Nx.Batch.key(:half)
      iex> Nx.Serving.run(serving, half_batch)
      #Nx.Tensor<
        f32[10]
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
      >

  When using a process-based serving, you must specify the supported
  `:batch_keys` when the process is started. The batch keys will be
  available inside the `defn_options` passed as the third argument of
  the `c:init/3` callback. The batch keys will also be verified
  when the batch is returned from the client-preprocessing callback.
  """

  @doc false
  @enforce_keys [:module, :arg]
  defstruct [
    :module,
    :arg,
    :client_preprocessing,
    :client_postprocessing,
    :streaming,
    :batch_size,
    distributed_postprocessing: &Function.identity/1,
    process_options: [],
    defn_options: []
  ]

  @type metadata() :: term()
  @type client_info() :: term()
  @type client_preprocessing() ::
          (term() -> {Nx.Batch.t() | Enumerable.t(Nx.Batch.t()), client_info()})
  @type client_postprocessing() :: ({Nx.Container.t(), metadata()}, client_info() -> term())
  @type distributed_preprocessing() :: (term() -> term())
  @type distributed_postprocessing() :: (term() -> term())

  @type t :: %__MODULE__{
          module: atom(),
          arg: term(),
          client_preprocessing: client_preprocessing(),
          client_postprocessing: client_postprocessing(),
          distributed_postprocessing: distributed_postprocessing(),
          process_options: keyword(),
          defn_options: keyword(),
          streaming: nil | %{hooks: [atom()]},
          batch_size: nil | pos_integer()
        }

  @axis 0

  @process_keys [
    :batch_size,
    :batch_timeout,
    :batch_keys,
    :partitions,
    :shutdown,
    :hibernate_after,
    :spawn_opt,
    :distribution_weight
  ]

  @doc """
  The callback used to initialize the serving.

  The first argument reveals if the serving is executed inline,
  such as by calling `run/2`, by started with the process.
  The second argument is the serving argument given to `new/2`.
  The third argument option is a list of compiler options to be
  used to compile each partition the serving will run.

  It must return `{:ok, state}`, where the `state` can be any term.
  """
  @callback init(type :: :inline | :process, arg :: term(), [defn_options :: keyword]) ::
              {:ok, state :: term()}

  @doc """
  Receives a batch, a partition, and returns a function to execute the batch.

  In case of serving processes, the function is executed is an
  separate process.
  """
  @callback handle_batch(Nx.Batch.t(), partition :: non_neg_integer(), state) ::
              {:execute, (-> {Nx.Container.t(), metadata()}), state}
            when state: term()

  @doc """
  Creates a new function serving.

  It expects a single- or double-arity function. If a single-arity
  function is given, it receives the compiler options and must
  return a JIT (via `Nx.Defn.jit/2`) or AOT compiled (via
  `Nx.Defn.compile/3`) one-arity function.

  If a double-arity function is given, it receives the batch
  key as first argument and the compiler options as second argument.
  It must return a JIT (via `Nx.Defn.jit/2`) or AOT compiled
  (via `Nx.Defn.compile/3`) one-arity function, but in practice
  it will be a `Nx.Defn.compile/3`, since the purpose of the
  batch key is often to precompile different versions of the
  same function upfront. The batch keys can be given on
  `start_link/1`.

  The function will be called with the arguments returned by the
  `client_preprocessing` callback.
  """
  def new(function, defn_options \\ [])

  def new(function, defn_options)
      when (is_function(function, 1) or is_function(function, 2)) and is_list(defn_options) do
    new(Nx.Serving.Default, function, defn_options)
  end

  def new(function, process_options)
      when is_function(function, 0) and is_list(process_options) do
    IO.warn(
      "passing a zero-arity function to Nx.Serving.new is deprecated, " <>
        "please pass a single arity function that will receive the compiler options"
    )

    new(Nx.Serving.Default, fn _ -> function.() end, [])
    |> process_options(process_options)
  end

  def new(module, arg) when is_atom(module) do
    new(module, arg, [])
  end

  @doc """
  Sets the batch size for this serving.

  This batch size is used to split batches given to both `run/2` and
  `batched_run/2`, enforcing that the batch size never goes over a limit.
  If you only want to batch within the serving process, you must set
  `:batch_size` via `process_options/2` (or on `start_link/1`).

  Note that `:batch_size` only guarantees a batch does not go over a limit.
  Batches are not automatically padded to the batch size. Such can be done
  as necessary inside your serving function by calling `Nx.Batch.pad/2`.

  > #### Why batch on `run/2`? {: .info}
  >
  > By default, `run/2` does not place a limit on its input size. It always
  > processes inputs directly within the current process. On the other hand,
  > `batched_run/2` always sends your input to a separate process, which
  > will batch and execute the serving only once the batch is full or a
  > timeout has elapsed.
  >
  > However, in some situations, an input given to `run/2` needs to be
  > broken into several batches. If we were to very large batches to our
  > computation, the computation could require too much memory. In such
  > cases, setting a batch size even on `run/2` is beneficial, because
  > Nx.Serving takes care of splitting a large batch into smaller ones
  > that do not exceed the `batch_size` value.
  """
  def batch_size(%Nx.Serving{} = serving, batch_size) when batch_size > 0 do
    %{serving | batch_size: batch_size}
  end

  @doc """
  Creates a new serving by jitting the given `fun` with `defn_options`.

  This is equivalent to:

      new(fn opts -> Nx.Defn.jit(fun, opts) end, defn_options)

  """
  def jit(fun, defn_options \\ []) do
    new(fn opts -> Nx.Defn.jit(fun, opts) end, defn_options)
  end

  @doc """
  Creates a new module-based serving.

  It expects a module and an argument that is given to its `init`
  callback.

  A third optional argument called `defn_options` are additional
  compiler options which will be given to the module. Those options
  will be merged into `Nx.Defn.default_options/0`.
  """
  def new(module, arg, defn_options) when is_atom(module) and is_list(defn_options) do
    defn_options = Keyword.merge(Nx.Defn.default_options(), defn_options)
    %Nx.Serving{module: module, arg: arg, defn_options: defn_options}
  end

  @doc """
  Sets the client preprocessing function.

  The default implementation expects a `Nx.Batch` or a stream of
  Nx.Batch to be given as input and return them as is.
  """
  def client_preprocessing(%Nx.Serving{} = serving, function)
      when is_function(function, 1) or is_nil(function) do
    %{serving | client_preprocessing: function}
  end

  @doc """
  Sets the client postprocessing function.

  The client postprocessing receives a tuple with the
  `{output, metadata}` or a stream as first argument.
  The second argument is always the additional information
  returned by the client preprocessing.

  The default implementation returns either the output or
  the stream.
  """
  def client_postprocessing(%Nx.Serving{} = serving, function)
      when is_function(function, 2) or is_nil(function) do
    %{serving | client_postprocessing: function}
  end

  def client_postprocessing(%Nx.Serving{} = serving, function)
      when is_function(function, 3) do
    IO.warn(
      "Passing a 3-arity function to client_postprocessing is deprecated, " <>
        "instead a two-arity function that receives the output and metadata must be given"
    )

    %{
      serving
      | client_postprocessing: fn {output, metadata}, info ->
          function.(output, metadata, info)
        end
    }
  end

  @doc """
  Sets the distributed postprocessing function.

  The default implementation is `Function.identity/1`.
  """
  def distributed_postprocessing(%Nx.Serving{} = serving, function)
      when is_function(function, 1) do
    %{serving | distributed_postprocessing: function}
  end

  @doc """
  Configure the serving to stream its results.

  Once `run/2` or `batched_run/2` are invoked, it will then
  return a stream. The stream must be consumed in the same
  process that calls `run/2` or `batched_run/2`.

  Batches will be streamed as they arrive. You may also opt-in
  to stream `Nx.Defn` hooks.

  ## Options

    * `:hooks` - a list of hook names that will become streaming events

  ## Implementation details

  ### Client postprocessing

  Once streaming is enabled, the client postprocessing callback
  will receive a stream which will emit events for each hook
  in the shape of:

      {hook_name, term()}

  The stream will also receive events in the shape of
  `{:batch, output, metadata}` as batches are processed by the
  serving. The client postprocessing is often expected to call
  `Stream.transform/3` to process those events into something
  usable by callers.

  If the `:hooks` option is given, only a single `:batch` event
  is emitted, at the end, as detailed next.

  ### Batch limits

  If you are streaming hooks, the serving server can no longer break
  batch and you are unable to push a payload bigger than `:batch_size`.
  For example, imagine you have a `batch_size` of 3 and you push three
  batches of two elements (AA, BB, and CC). Without hooks, the batches
  will be consumed as:

      AAB -> BCC

  With streaming, we can't break the batch `BB`, as above, so we will
  consistently pad with zeroes:

      AA0 -> BB0 -> CC0

  In practice, this should not be a major problem, as you should
  generally avoid having a batch size that is not a multiple of the
  most common batches.
  """
  def streaming(%Nx.Serving{} = serving, opts \\ []) do
    hooks = Keyword.get(opts, :hooks, [])

    if serving.streaming do
      raise ArgumentError, "serving is already marked as streaming"
    end

    %{serving | streaming: %{hooks: hooks}}
  end

  @doc """
  Sets the process options of this serving.

  These are the same options as supported on `start_link/1`,
  except `:name` and `:serving` itself.
  """
  def process_options(%Nx.Serving{} = serving, opts) when is_list(opts) do
    %{serving | process_options: Keyword.validate!(opts, @process_keys)}
  end

  @doc """
  Sets the defn options of this serving.

  These are the options supported by `Nx.Defn.default_options/1`.
  """
  def defn_options(%Nx.Serving{} = serving, defn_options) when is_list(defn_options) do
    %{serving | defn_options: defn_options}
  end

  @doc """
  Runs `serving` with the given `input` inline with the current process.

  The `serving` is executed immediately, without waiting or batching inputs
  from other processes. If a `batch_size/2` is specified, then the input may
  be split or padded, but they are still executed immediately inline.
  """
  def run(%Nx.Serving{} = serving, input) do
    %{
      module: module,
      arg: arg,
      client_preprocessing: preprocessing,
      client_postprocessing: postprocessing,
      defn_options: defn_options,
      streaming: streaming,
      batch_size: limit
    } = serving

    {batch_or_stream, info} = handle_preprocessing(preprocessing, input)
    {pid_ref, defn_options} = run_streaming(streaming, defn_options, batch_or_stream, limit)
    stream = run_batch_or_stream(batch_or_stream, limit)

    execution_result =
      case pid_ref do
        {pid, ref} ->
          send(pid, {ref, module, arg, defn_options, stream})
          receive_stream("run/2", ref, :unknown)

        nil ->
          stream
          |> Enum.map_reduce(nil, fn %Nx.Batch{key: key, size: size} = batch, cache ->
            {:ok, state} =
              cache || handle_init(module, :inline, arg, [[batch_keys: [key]] ++ defn_options])

            {{run_execute(batch, module, state), size}, {:ok, state}}
          end)
          |> elem(0)
          |> case do
            [{{output, metadata}, size}] ->
              {remove_maybe_padded(output, 0, size), metadata}

            [{{output, metadata}, _size} | _] = all ->
              output =
                all
                |> Enum.map(fn {{output, _}, size} -> composite_to_tensors(output, 0, size) end)
                |> tensors_to_composite(output)

              {output, metadata}
          end
      end

    handle_postprocessing(postprocessing, execution_result, info)
  end

  defp run_streaming(nil, defn_options, _batch_or_stream, _limit),
    do: {nil, defn_options}

  defp run_streaming(%{hooks: []}, defn_options, _batch_or_stream, _limit),
    do: {run_streaming(), defn_options}

  defp run_streaming(%{hooks: hooks}, defn_options, batch_or_stream, limit) do
    size =
      case batch_or_stream do
        %Nx.Batch{size: size} ->
          if limit == nil or size <= limit do
            size
          else
            raise ArgumentError,
                  "batch size (#{size}) cannot exceed Nx.Serving server batch size of #{limit} when streaming hooks"
          end

        _ ->
          raise ArgumentError,
                "streaming hooks do not support input streaming, input must be a Nx.Batch"
      end

    {pid, ref} = run_streaming()

    defn_options =
      update_in(defn_options[:hooks], fn acc ->
        Enum.reduce(hooks, acc || %{}, fn hook, acc ->
          Map.put(acc, hook, &run_hook(ref, size, &1, hook))
        end)
      end)

    {{pid, ref}, defn_options}
  end

  defp run_streaming do
    pid =
      spawn_link(fn ->
        receive do
          {ref, module, arg, defn_options, stream} ->
            Enum.reduce(stream, {0, nil}, fn
              %Nx.Batch{key: key, size: size} = batch, {start, cache} ->
                {:ok, state} =
                  cache ||
                    handle_init(module, :inline, arg, [[batch_keys: [key]] ++ defn_options])

                {output, metadata} = run_execute(batch, module, state)
                send(ref, {ref, {:batch, {0, size, output, metadata}}})
                {start + size, {:ok, state}}
            end)
        end
      end)

    {pid, Process.monitor(pid, alias: :demonitor)}
  end

  defp run_hook(ref, size, result, hook) do
    send(ref, {ref, {:hook, {0, size, result, hook}}})
  end

  defp run_batch_or_stream(%Nx.Batch{size: size} = batch, limit)
       when is_nil(limit) or size < limit do
    [batch]
  end

  defp run_batch_or_stream(%Nx.Batch{} = batch, limit) do
    Stream.unfold(batch, fn
      %Nx.Batch{size: size} = batch when size > limit ->
        Nx.Batch.split(batch, limit)

      %Nx.Batch{} = batch ->
        {batch, :done}

      :done ->
        nil
    end)
  end

  defp run_batch_or_stream(stream, limit) do
    Stream.each(stream, fn
      %Nx.Batch{size: size} when is_nil(limit) or size <= limit ->
        :ok

      other ->
        raise "client_preprocessing must return a stream of Nx.Batch" <>
                if(limit, do: " of maximum size #{limit}", else: "") <> ", got: #{inspect(other)}"
    end)
  end

  defp run_execute(batch, module, state) do
    {:execute, function, _} = handle_batch(module, batch, 0, state)

    :telemetry.span([:nx, :serving, :execute], %{module: module}, fn ->
      {output, metadata} = handle_executed(module, function.())
      {{output, metadata}, %{module: module, metadata: metadata}}
    end)
  end

  defp remove_maybe_padded(output, start, size) do
    Nx.Defn.Composite.traverse(output, &Nx.slice_along_axis(&1, start, size, axis: @axis))
  end

  defp composite_to_tensors(output, start, size) do
    output
    |> Nx.Defn.Composite.reduce(
      [],
      &[Nx.slice_along_axis(&1, start, size, axis: @axis) | &2]
    )
    |> Enum.reverse()
  end

  defp tensors_to_composite(tensors, template) do
    concat = Enum.zip_with(tensors, &Nx.concatenate(&1, axis: @axis))

    {output, []} =
      Nx.Defn.Composite.traverse(template, concat, fn _template, [tensor | tensors] ->
        {tensor, tensors}
      end)

    output
  end

  ## Process API

  @doc false
  def child_spec(opts) when is_list(opts) do
    name = opts[:name]

    if name == nil or not is_atom(name) do
      raise ArgumentError, ":name option is expected when starting Nx.Serving and must be an atom"
    end

    opts[:serving] ||
      raise ArgumentError, ":serving option is expected when starting Nx.Serving"

    %{
      id: name,
      start: {__MODULE__, :start_link, [opts]},
      type: :supervisor
    }
  end

  @doc """
  Starts a `Nx.Serving` process to batch requests to a given serving.

  ## Options

  All options, except `:name` and `:serving`, can also be set via
  `process_options/2`.

    * `:name` - an atom with the name of the process

    * `:serving` - a `Nx.Serving` struct with the serving configuration

    * `:batch_keys` - all available batch keys. Batch keys allows Nx.Serving
      to accumulate different batches with different properties. Defaults to
      `[:default]`

    * `:batch_size` - the maximum batch size. A default value can be set with
      `batch_size/2`, which applies to both `run/2` and `batched_run/2`.
      Setting this option only affects `batched_run/2` and it defaults to `1`
      if none is set. Note batches received by the serving are not automatically
      padded to the batch size, such can be done with `Nx.Batch.pad/2`.

    * `:batch_timeout` - the maximum time to wait, in milliseconds,
      before executing the batch (defaults to `100`ms)

    * `:partitions` - when `true`, starts several partitions under this serving.
      The number of partitions will be determined according to your compiler
      and for which host it is compiling. See the module docs for more information

    * `:distribution_weight` - weight used for load balancing when running
      a distributed serving. Defaults to `1`.
      If it is set to a higher number `w`, the serving process will receive,
      on average, `w` times the number of requests compared to the
      default. Note that the weight is multiplied with the number of
      partitions, if partitioning is enabled.

    * `:shutdown` - the maximum time for the serving to shutdown. This will
      block until the existing computation finishes (defaults to `30_000`ms)

    * `:hibernate_after` and `:spawn_opt` - configure the underlying serving
      workers (see `GenServer.start_link/3`)
  """
  def start_link(opts) do
    opts = Keyword.validate!(opts, [:name, :serving] ++ @process_keys)
    name = Keyword.fetch!(opts, :name)
    serving = Keyword.fetch!(opts, :serving)
    opts = Keyword.merge(serving.process_options, opts)

    serving_batch_size = serving.batch_size
    opts_batch_size = opts[:batch_size]

    batch_size =
      if serving_batch_size && opts_batch_size && serving_batch_size != opts_batch_size do
        raise ArgumentError,
              "the batch size set via Nx.Serving.batch_size/2 (#{serving_batch_size}) " <>
                "does not match the batch size given to the serving process (#{opts_batch_size})"
      else
        serving_batch_size || opts_batch_size || 1
      end

    shutdown = Keyword.get(opts, :shutdown, 30_000)
    partitions = Keyword.get(opts, :partitions, false)
    batch_keys = Keyword.get(opts, :batch_keys, [:default])
    batch_timeout = Keyword.get(opts, :batch_timeout, 100)
    weight = Keyword.get(opts, :distribution_weight, 1)
    process_options = Keyword.take(opts, [:name, :hibernate_after, :spawn_opt])

    unless is_integer(weight) do
      raise ArgumentError, ":distribution_weight must be an integer"
    end

    supervisor = Module.concat(name, "Supervisor")
    task_supervisor = Module.concat(name, "TaskSupervisor")

    arg =
      {name, serving, partitions, batch_keys, batch_size, batch_timeout, task_supervisor, weight}

    children = [
      {Task.Supervisor, name: task_supervisor},
      %{
        id: __MODULE__,
        start: {GenServer, :start_link, [__MODULE__, arg, process_options]},
        shutdown: shutdown
      }
    ]

    Supervisor.start_link(children, strategy: :one_for_all, max_restarts: 0, name: supervisor)
  end

  @doc """
  Runs the given `input` on the serving process given by `name`.

  `name` is either an atom representing a local or distributed
  serving process. First it will attempt to dispatch locally, then it
  falls back to the distributed serving. You may specify
  `{:local, name}` to force a local lookup or `{:distributed, name}`
  to force a distributed one.

  The `client_preprocessing` callback will be invoked on the `input`
  which is then sent to the server. The server will batch requests
  and send a response either when the batch is full or on timeout.
  Then `client_postprocessing` is invoked on the response. See the
  module documentation for more information. In the distributed case,
  the callbacks are invoked in the distributed node, but still outside of
  the serving process.

  Note that you cannot batch an `input` larger than the configured
  `:batch_size` in the server.

  ## Distributed mode

  To run in distributed mode, the nodes do not need to run the same
  code and applications. It is only required that they run the
  same `Nx` version.

  If the current node is running a serving given by `name` locally
  and `{:distributed, name}` is used, the request will use the same
  distribution mechanisms instead of being handled locally, which
  is useful for testing locally without a need to spawn nodes.

  This function receives an optional `distributed_preprocessing` callback as
  third argument for preprocessing the input for distributed requests. When
  using libraries like EXLA or Torchx, the tensor is often allocated in memory
  inside a third-party library so it may be necessary to either transfer or copy
  the tensor to the binary backend before sending it to another node.
  This can be done by passing either `Nx.backend_transfer/1` or `Nx.backend_copy/1`
  as third argument:

      Nx.Serving.batched_run(MyDistributedServing, input, &Nx.backend_copy/1)

  Use `backend_transfer/1` if you know the input will no longer be used.

  Similarly, the serving has a `distributed_postprocessing` callback which can do
  equivalent before sending the reply to the caller.
  """
  def batched_run(name, input, distributed_preprocessing \\ &Function.identity/1)

  def batched_run(name, input, distributed_preprocessing) when is_atom(name) do
    if pid = Process.whereis(name) do
      local_batched_run!(pid, name, input)
    else
      distributed_batched_run!(name, input, distributed_preprocessing)
    end
  end

  def batched_run({:local, name}, input, _distributed_preprocessing) when is_atom(name) do
    pid =
      Process.whereis(name) || exit({:noproc, {__MODULE__, :local_batched_run, [name, input]}})

    local_batched_run!(pid, name, input)
  end

  def batched_run({:distributed, name}, input, distributed_preprocessing) when is_atom(name) do
    distributed_batched_run!(name, input, distributed_preprocessing)
  end

  defp local_batched_run!(pid, name, input) do
    case local_batched_run(pid, name, input) do
      {:ok, result} -> result
      {:DOWN, reason} -> exit({reason, {__MODULE__, :local_batched_run, [name, input]}})
    end
  end

  defp local_batched_run(pid, name, input) do
    %{
      preprocessing: preprocessing,
      postprocessing: postprocessing,
      limit: limit,
      mode: mode,
      batch_keys: batch_keys
    } =
      :persistent_term.get(persistent_key(name), nil) ||
        raise(
          ArgumentError,
          "could not find Nx.Serving with name #{inspect(name)}. " <>
            "Make sure your Nx.Serving is running and/or started as part of your supervision tree"
        )

    {preprocessed, info} = handle_preprocessing(preprocessing, input)

    ref = Process.monitor(pid, alias: :demonitor)

    size_or_unknown =
      case preprocessed do
        %Nx.Batch{size: size} = batch ->
          if mode == :hooks and batch.size > limit do
            raise ArgumentError,
                  "batch size (#{batch.size}) cannot exceed Nx.Serving server batch size of #{limit} when streaming hooks"
          end

          validate_batch_key!(batch, batch_keys)
          Process.send(pid, {__MODULE__, :batched_run, [ref], batch}, [:noconnect])
          size

        stream ->
          if mode == :hooks do
            raise ArgumentError,
                  "streaming hooks do not support input streaming, input must be a Nx.Batch"
          end

          spawn_link(fn ->
            # We also need to monitor the streaming process. To avoid leaking
            # messages in the parent inbox, we ask the serving to do it.
            Process.send(pid, {__MODULE__, :proxy_monitor, self(), ref}, [:noconnect])
            monitor_ref = Process.monitor(pid)

            acc =
              Enum.reduce(stream, 0, fn
                %Nx.Batch{size: size} = batch, acc when size <= limit ->
                  receive_size(monitor_ref, ref, acc)
                  validate_batch_key!(batch, batch_keys)
                  refs = [ref, self()]
                  Process.send(pid, {__MODULE__, :batched_run, refs, batch}, [:noconnect])
                  size

                other, _acc ->
                  raise "client_preprocessing must return a stream of Nx.Batch " <>
                          "of maximum size #{limit}, got: #{inspect(other)}"
              end)

            receive_size(monitor_ref, ref, acc)
          end)

          :unknown
      end

    case mode do
      :execute ->
        case receive_execute(ref, size_or_unknown) do
          {:ok, tensor, metadata} ->
            {:ok, handle_postprocessing(postprocessing, {tensor, metadata}, info)}

          {:DOWN, reason} ->
            {:DOWN, reason}
        end

      _ ->
        stream = receive_stream("batched_run/2", ref, size_or_unknown)
        {:ok, handle_postprocessing(postprocessing, stream, info)}
    end
  end

  defp validate_batch_key!(batch, batch_keys) do
    unless is_map_key(batch_keys, batch.key) do
      raise ArgumentError,
            "unknown batch key: #{inspect(batch.key)} (expected one of #{inspect(Map.keys(batch_keys))})"
    end
  end

  defp distributed_batched_run!(name, input, distributed_callback) do
    distributed_batched_run_with_retries!(name, distributed_callback.(input), 3)
  end

  defp distributed_batched_run_with_retries!(name, input, 0) do
    exit({:noproc, {__MODULE__, :distributed_batched_run, [name, input, [retries: 0]]}})
  end

  defp distributed_batched_run_with_retries!(name, input, retries) do
    case :pg.get_members(Nx.Serving.PG, __MODULE__) do
      [] ->
        exit({:noproc, {__MODULE__, :distributed_batched_run, [name, input, [retries: retries]]}})

      entries ->
        pid = Enum.random(entries)
        ref = make_ref()
        args = [self(), ref, name, input]

        {_, monitor_ref} =
          Node.spawn_monitor(node(pid), __MODULE__, :__distributed_batched_run__, args)

        receive do
          {^ref, :streaming} ->
            owner = self()

            Stream.resource(
              fn ->
                if self() != owner do
                  raise "the stream returned from Nx.Serving.batched_run/2 must be consumed in the same process"
                end

                :ok
              end,
              fn :ok ->
                receive do
                  {^ref, event} ->
                    {[event], :ok}

                  {:DOWN, ^monitor_ref, _, _, {^ref, :streaming}} ->
                    {:halt, :ok}

                  {:DOWN, ^monitor_ref, _, _, reason} ->
                    exit({reason, {Nx.Serving, :streaming, []}})
                end
              end,
              fn _ -> :ok end
            )

          {:DOWN, ^monitor_ref, _, _, {^ref, result}} ->
            result

          {:DOWN, ^monitor_ref, _, _, :noproc} ->
            distributed_batched_run_with_retries!(name, input, retries - 1)

          {:DOWN, ^monitor_ref, _, _, reason} ->
            exit_args = [name, input, [retries: retries]]
            exit({reason, {__MODULE__, :distributed_batched_run, exit_args}})
        end
    end
  end

  @doc false
  def __distributed_batched_run__(client_pid, ref, name, input) do
    pid = Process.whereis(name) || exit(:noproc)

    case local_batched_run(pid, name, input) do
      {:ok, result} ->
        %{mode: mode, distributed_postprocessing: dist_post} =
          :persistent_term.get(persistent_key(name))

        if mode == :execute do
          exit({ref, dist_post.(result)})
        else
          send(client_pid, {ref, :streaming})
          Enum.each(dist_post.(result), &send(client_pid, {ref, &1}))
          exit({ref, :streaming})
        end

      {:DOWN, reason} ->
        exit(reason)
    end
  end

  ## Client message receiving

  defp receive_size(_monitor, _ref, 0), do: :ok

  defp receive_size(monitor_ref, ref, pending) do
    receive do
      {^ref, size} ->
        receive_size(monitor_ref, ref, pending - size)

      {:DOWN, ^monitor_ref, _, _, reason} ->
        exit(reason)
    end
  end

  defp receive_stream(fun, ref, size) when is_integer(size) or size == :unknown do
    owner = self()

    Stream.resource(
      fn ->
        if self() != owner do
          raise "the stream returned from Nx.Serving.#{fun} must be consumed in the same process"
        end

        0
      end,
      fn
        ^size ->
          {:halt, :done}

        index ->
          case receive_each(ref, size, index) do
            :done ->
              {:halt, :done}

            {:hook, {hook_start, hook_size, output, hook}} ->
              value = remove_maybe_padded(output, hook_start, hook_size)
              {[{hook, value}], index}

            {:batch, {output_start, output_size, output, metadata}} ->
              value = remove_maybe_padded(output, output_start, output_size)
              {[{:batch, value, metadata}], index + output_size}

            {:DOWN, reason} ->
              exit({reason, {Nx.Serving, :streaming, []}})
          end
      end,
      fn _ -> :ok end
    )
  end

  defp receive_execute(ref, size) when is_integer(size) or size == :unknown do
    receive_execute(ref, size, 0, [], nil)
  end

  defp receive_execute(ref, size, index, acc, template_metadata) do
    case receive_each(ref, size, index) do
      :done ->
        {template, metadata} =
          template_metadata || raise "unexpected error: streaming finished before it started"

        {:ok, acc |> Enum.reverse() |> tensors_to_composite(template), metadata}

      {:batch, {output_start, output_size, output, metadata}} ->
        # If we have a single response, slice and return immediately.
        # Otherwise we collect their contents and build the concatenated result later.
        if acc == [] and output_size + index == size do
          {:ok, remove_maybe_padded(output, output_start, output_size), metadata}
        else
          funs = composite_to_tensors(output, output_start, output_size)
          receive_execute(ref, size, index + output_size, [funs | acc], {output, metadata})
        end

      {:DOWN, reason} ->
        {:DOWN, reason}
    end
  end

  defp receive_each(_ref, size, size) do
    :done
  end

  defp receive_each(ref, size, index) do
    receive do
      {^ref, {:hook, _} = reply} ->
        reply

      {^ref, {:batch, {_output_start, output_size, _output, _metadata}} = reply} ->
        if output_size + index == size do
          Process.demonitor(ref, [:flush])
        end

        reply

      # The serving itself never finishes with normal reason,
      # but the streaming process does to signal it is concluded
      # and its messages are proxied here.
      {:DOWN, ^ref, _, _, :normal} ->
        Process.demonitor(ref, [:flush])
        :done

      {:DOWN, ^ref, _, _, reason} ->
        # We fake monitor messages, so still demonitor and flush.
        Process.demonitor(ref, [:flush])
        {:DOWN, reason}
    end
  end

  ## Process callbacks

  require Logger
  @behaviour GenServer

  @empty_stack {[], 0, :none}
  @empty_queue :queue.new()
  @timeout_message {__MODULE__, :timeout}

  @impl true
  def init(
        {name, serving, partitions?, batch_keys, batch_size, batch_timeout, task_supervisor,
         weight}
      ) do
    Process.flag(:trap_exit, true)
    partitions_opts = serving_partitions(serving, partitions?)
    partitions_count = length(partitions_opts)
    {mode, partitions_opts, hooks_table} = serving_streaming(serving, partitions_opts)
    partitions_opts = Enum.map(partitions_opts, &Keyword.put(&1, :batch_keys, batch_keys))
    {:ok, module_state} = handle_init(serving.module, :process, serving.arg, partitions_opts)

    :persistent_term.put(
      persistent_key(name),
      %{
        limit: batch_size,
        preprocessing: serving.client_preprocessing,
        postprocessing: serving.client_postprocessing,
        distributed_postprocessing: serving.distributed_postprocessing,
        mode: mode,
        batch_keys: Map.from_keys(batch_keys, [])
      }
    )

    serving_weight = max(1, weight * partitions_count)
    :pg.join(Nx.Serving.PG, __MODULE__, List.duplicate(self(), serving_weight))

    for batch_key <- batch_keys do
      stack_init(batch_key)
    end

    # We keep batches in a stack. Once the stack is full
    # or it times out, we either execute or enqueue it.
    state = %{
      module: serving.module,
      module_state: module_state,
      limit: batch_size,
      timeout: batch_timeout,
      in_queue: @empty_queue,
      out_queue: Enum.reduce(0..(partitions_count - 1), :queue.new(), &:queue.in/2),
      tasks: [],
      pending_batches: Map.from_keys(batch_keys, @empty_queue),
      task_supervisor: task_supervisor,
      hooks_table: hooks_table
    }

    {:ok, state}
  end

  defp serving_partitions(%Nx.Serving{defn_options: defn_options}, true) do
    compiler = Keyword.get(defn_options, :compiler, Nx.Defn.Evaluator)
    compiler.__partitions_options__(defn_options)
  end

  defp serving_partitions(%Nx.Serving{defn_options: defn_options}, false) do
    [defn_options]
  end

  defp serving_streaming(%Nx.Serving{streaming: nil}, partitions) do
    {:execute, partitions, nil}
  end

  defp serving_streaming(%Nx.Serving{streaming: %{hooks: []}}, partitions) do
    {:batches, partitions, nil}
  end

  defp serving_streaming(%Nx.Serving{streaming: %{hooks: hooks}}, partitions) do
    ets = :ets.new(__MODULE__, [:public, :set, read_concurrency: true])

    partitions =
      Enum.with_index(partitions, fn defn_options, index ->
        update_in(defn_options[:hooks], fn acc ->
          Enum.reduce(hooks, acc || %{}, fn hook, acc ->
            Map.put(acc, hook, &server_hook(ets, index, hook, &1))
          end)
        end)
      end)

    {:hooks, partitions, ets}
  end

  defp server_hook(ets, index, hook, result) do
    for {[ref | _pids], start, size} <- :ets.lookup_element(ets, index, 2) do
      send(ref, {ref, {:hook, {start, size, result, hook}}})
    end
  end

  @impl true
  def handle_info({__MODULE__, :proxy_monitor, pid, ref}, state) do
    Process.monitor(pid, tag: {:proxy, ref})
    {:noreply, state}
  end

  def handle_info({__MODULE__, :batched_run, refs, %Nx.Batch{key: key} = batch}, state) do
    %{limit: limit} = state
    count = stack_count(key)

    state =
      cond do
        # Single entry takes the whole batch.
        # Execute what we have (if any) and execute a new one.
        batch.size == limit ->
          state
          |> server_execute(key)
          |> server_stack(key, refs, batch, :skip_timer)
          |> server_execute(key)

        # We go over the limit, but if using hooks, we can't split.
        batch.size + count > limit and state.hooks_table != nil ->
          state
          |> server_execute(key)
          |> server_stack(key, refs, batch, :set_timer)

        # Split as necessary.
        true ->
          server_stack_and_execute_loop(state, batch, count, key, refs)
      end

    {:noreply, state}
  end

  def handle_info({@timeout_message, key, ref}, %{out_queue: out_queue} = state) do
    case stack_timer(key) do
      # We have processing power, so execute it immediately.
      {^ref, _timer_ref} when out_queue != @empty_queue ->
        {:noreply, server_execute(state, key)}

      # Otherwise we will queue it but keep on increasing the batch.
      {^ref, _timer_ref} ->
        stack_update(key, fn {[_ | _] = stack, count, _timer} ->
          {stack, count, :done}
        end)

        {:noreply, update_in(state.in_queue, &:queue.in(key, &1))}

      # Otherwise this is an old timer message, just ignore it.
      _ ->
        {:noreply, state}
    end
  end

  def handle_info({ref, :done}, %{tasks: tasks} = state) do
    case Enum.split_with(tasks, &(elem(&1, 0).ref == ref)) do
      {[{_task, partition, _ref_sizes}], tasks} ->
        Process.demonitor(ref, [:flush])
        noreply_task_done_and_continue(state, tasks, partition)

      _ ->
        {:noreply, state}
    end
  end

  def handle_info({{:proxy, ref}, _ref, type, info, reason}, state) do
    send(ref, {:DOWN, ref, type, info, reason})
    {:noreply, state}
  end

  def handle_info({:DOWN, ref, :process, _process, reason}, %{tasks: tasks} = state) do
    case Enum.split_with(tasks, &(elem(&1, 0).ref == ref)) do
      {[{_task, partition, ref_sizes}], tasks} ->
        server_reply_down(reason, ref_sizes)
        noreply_task_done_and_continue(state, tasks, partition)

      _ ->
        {:noreply, state}
    end
  end

  def handle_info(msg, state) do
    Logger.warning("Unknown message in Nx.Serving: #{inspect(msg)}")
    {:noreply, state}
  end

  @impl true
  def handle_continue(:maybe_task, state) do
    {:noreply, server_maybe_task(state)}
  end

  @impl true
  def terminate(_reason, %{tasks: tasks, pending_batches: pending_batches}) do
    for {batch_key, queue} <- pending_batches do
      # Emulate the process is gone for entries in the queue
      for {_batch, ref_sizes} <- :queue.to_list(queue) do
        server_reply_down(:noproc, ref_sizes)
      end

      # As well as for entries in the stack
      for {[ref | _pids], _batch} <- stack_entries(batch_key) do
        send(ref, {:DOWN, ref, :process, self(), :noproc})
      end
    end

    # And wait until all current tasks are processed
    for {%Task{ref: ref}, _partition, ref_sizes} <- tasks do
      receive do
        {^ref, :done} -> Process.demonitor(ref, [:flush])
        {:DOWN, ^ref, :process, _, reason} -> server_reply_down(reason, ref_sizes)
      end
    end

    :ok
  end

  # We don't spawn the task here because, if it crashes,
  # we want a checked-in version of the state that knows
  # the current task has finished.
  defp noreply_task_done_and_continue(%{out_queue: out_queue} = state, tasks, partition) do
    out_queue = :queue.in(partition, out_queue)
    {:noreply, %{state | tasks: tasks, out_queue: out_queue}, {:continue, :maybe_task}}
  end

  defp server_reply_down(reason, ref_sizes) do
    for {[ref | _refs], _start, _size} <- ref_sizes do
      send(ref, {:DOWN, ref, :process, self(), reason})
    end
  end

  defp server_stack_and_execute_loop(state, batch, count, key, refs) do
    %{limit: limit} = state
    %{size: size} = batch

    cond do
      size + count < limit ->
        server_stack(state, key, refs, batch, :set_timer)

      size + count > limit ->
        {current, batch} = Nx.Batch.split(batch, limit - count)

        state
        |> server_stack(key, refs, current, :skip_timer)
        |> server_execute(key)
        |> server_stack_and_execute_loop(batch, 0, key, refs)

      true ->
        state
        |> server_stack(key, refs, batch, :skip_timer)
        |> server_execute(key)
    end
  end

  defp server_stack(%{limit: limit} = state, key, refs, batch, timer_mode) do
    stack_update(key, fn {stack, count, timer} when batch.size + count <= limit ->
      timer =
        if timer == :none and timer_mode == :set_timer do
          ref = make_ref()
          {ref, Process.send_after(self(), {@timeout_message, key, ref}, state.timeout)}
        else
          timer
        end

      {[{refs, batch} | stack], count + batch.size, timer}
    end)

    state
  end

  defp server_execute(state, key) do
    if stack_count(key) == 0 do
      state
    else
      {batch_refs, timer} = stack_to_batch_refs(key)
      state = update_in(state.pending_batches[key], &:queue.in(batch_refs, &1))

      state =
        if timer == :done do
          state
        else
          update_in(state.in_queue, &:queue.in(key, &1))
        end

      server_maybe_task(state)
    end
  end

  defp server_maybe_task(state) do
    %{out_queue: out_queue, in_queue: in_queue, pending_batches: pending_batches} = state

    with {{:value, partition}, out_queue} <- :queue.out(out_queue),
         {{:value, key}, in_queue} <- :queue.out(in_queue) do
      {{batch, ref_sizes}, pending_batches} =
        case :queue.out(pending_batches[key]) do
          {:empty, _pending_batches} ->
            # If there is no entry pending, then we have a timed-out in-construction batch.
            {batch_refs, :done} = stack_to_batch_refs(key)
            {batch_refs, pending_batches}

          {{:value, batch_refs}, queue} ->
            {batch_refs, Map.put(pending_batches, key, queue)}
        end

      %{module: module, module_state: module_state, hooks_table: hooks_table} = state
      {:execute, function, module_state} = handle_batch(module, batch, partition, module_state)

      wrapped_function = fn ->
        :telemetry.span([:nx, :serving, :execute], %{module: module}, fn ->
          if hooks_table do
            :ets.insert(hooks_table, {partition, ref_sizes})
          end

          {output, metadata} = function.()

          for {[ref | pids], start, size} <- ref_sizes do
            send(ref, {ref, {:batch, {start, size, output, metadata}}})

            for pid <- pids do
              send(pid, {ref, size})
            end
          end

          {:done, %{metadata: metadata, module: module}}
        end)
      end

      task = Task.Supervisor.async_nolink(state.task_supervisor, wrapped_function)
      tasks = [{task, partition, ref_sizes} | state.tasks]

      %{
        state
        | module_state: module_state,
          tasks: tasks,
          out_queue: out_queue,
          in_queue: in_queue,
          pending_batches: pending_batches
      }
    else
      _ -> state
    end
  end

  ## Stack management
  #
  # The stack is stored in the process dictionary for performance
  # since the common case does not use any batch key.

  defp stack_init(key) do
    Process.put({__MODULE__, key}, @empty_stack)
    :ok
  end

  defp stack_count(key) do
    {_stack, count, _timer} = Process.get({__MODULE__, key})
    count
  end

  defp stack_timer(key) do
    {_stack, _count, timer} = Process.get({__MODULE__, key})
    timer
  end

  defp stack_entries(key) do
    {stack, _count, _timer} = Process.get({__MODULE__, key})
    stack
  end

  defp stack_update(key, fun) do
    Process.put({__MODULE__, key}, fun.(Process.get({__MODULE__, key})))
    :ok
  end

  defp stack_to_batch_refs(key) do
    {[_ | _] = stack, count, timer} = Process.get({__MODULE__, key})
    :ok = stack_init(key)

    with {ref, timer_ref} <- timer do
      Process.cancel_timer(timer_ref)

      receive do
        {@timeout_message, ^key, ^ref} -> :ok
      after
        0 -> :ok
      end
    end

    {ref_sizes, batches, _} =
      Enum.reduce(stack, {[], [], count}, fn {refs, batch}, {ref_sizes, batches, ending} ->
        size = batch.size
        {[{refs, ending - size, size} | ref_sizes], [batch | batches], ending - size}
      end)

    {{Nx.Batch.merge(batches), ref_sizes}, timer}
  end

  ## Shared helpers

  defp persistent_key(name) when is_atom(name) do
    {__MODULE__, name}
  end

  defp handle_init(module, type, arg, [_ | _] = partitions) do
    case module.init(type, arg, partitions) do
      {:ok, _} = pair ->
        pair

      other ->
        raise "#{inspect(module)}.init/3 must return {:ok, state}. Got: #{inspect(other)}"
    end
  end

  defp handle_batch(module, batch, partition, state) do
    case module.handle_batch(batch, partition, state) do
      {:execute, function, _} = pair when is_function(function, 0) ->
        pair

      other ->
        raise "#{inspect(module)}.handle_batch/3 must return {:execute, function, state}, " <>
                "where function is a function that receives no arguments and returns a tuple. " <>
                "Got: #{inspect(other)}"
    end
  end

  defp handle_executed(module, result) do
    case result do
      {output, metadata} ->
        {output, metadata}

      other ->
        raise "the function returned by #{inspect(module)}.handle_batch/3 must return {output, metadata}. " <>
                "Got: #{inspect(other)}"
    end
  end

  defp handle_preprocessing(nil, input) do
    batch_or_stream =
      validate_batch_or_stream(input) ||
        raise(
          ArgumentError,
          "the default client_preprocessing expects a Nx.Batch or a stream of Nx.Batch as input. " <>
            "Give a batch or use a custom preprocessing"
        )

    {batch_or_stream, :client_info}
  end

  defp handle_preprocessing(preprocessing, input) do
    meta = %{input: input}

    :telemetry.span([:nx, :serving, :preprocessing], meta, fn ->
      result = preprocessing.(input)

      case result do
        {batch_or_stream, info} ->
          batch_or_stream =
            validate_batch_or_stream(batch_or_stream) ||
              raise_bad_client_preprocessing!(preprocessing, result)

          {{batch_or_stream, info}, Map.put(meta, :info, info)}

        _ ->
          raise_bad_client_preprocessing!(preprocessing, result)
      end
    end)
  end

  defp raise_bad_client_preprocessing!(preprocessing, result) do
    raise "client_preprocessing function #{inspect(preprocessing)} must return a two element tuple " <>
            "where the first element is a Nx.Batch or a stream of batches and the second is any value. Got: #{inspect(result)}"
  end

  defp validate_batch_or_stream(%Nx.Batch{size: 0}),
    do: raise(ArgumentError, "cannot run with empty Nx.Batch")

  defp validate_batch_or_stream(%Nx.Batch{} = batch), do: batch

  defp validate_batch_or_stream(stream) do
    if Enumerable.impl_for(stream) do
      stream
    end
  end

  defp handle_postprocessing(nil, {output, _metadata}, _info), do: output
  defp handle_postprocessing(nil, stream, _info), do: stream

  defp handle_postprocessing(postprocessing, result, info) do
    meta = %{info: info}

    :telemetry.span([:nx, :serving, :postprocessing], meta, fn ->
      {postprocessing.(result, info), meta}
    end)
  end
end

defmodule Nx.Serving.Default do
  @moduledoc false
  @behaviour Nx.Serving

  @impl true
  def init(_type, fun, partitions) do
    batch_funs =
      Enum.with_index(partitions, fn defn_options, index ->
        value =
          cond do
            is_function(fun, 1) ->
              validate_batch_fun!(fun.(defn_options))

            is_function(fun, 2) ->
              {batch_keys, defn_options} = Keyword.pop!(defn_options, :batch_keys)

              for batch_key <- batch_keys,
                  into: %{},
                  do: {batch_key, validate_batch_fun!(fun.(batch_key, defn_options))}
          end

        {index, value}
      end)

    {:ok, Map.new(batch_funs)}
  end

  defp validate_batch_fun!(batch_fun) when is_function(batch_fun, 1), do: batch_fun

  defp validate_batch_fun!(other) do
    raise "anonymous function given to Nx.Serving.new/2 should return an AOT or " <>
            "JIT compiled function that expects one argument. Got: #{inspect(other)}"
  end

  @impl true
  def handle_batch(batch, partition, batch_funs) do
    batch_fun =
      case batch_funs do
        %{^partition => batch_keys} when is_map(batch_keys) -> Map.fetch!(batch_keys, batch.key)
        %{^partition => fun} -> fun
      end

    {:execute, fn -> {batch_fun.(batch), :server_info} end, batch_funs}
  end
end
