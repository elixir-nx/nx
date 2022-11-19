defmodule Nx.Serving do
  @moduledoc """
  Serving encapsulates client and server work to perform batched requests.

  Serving can be executed on the fly, without starting a server, but most
  often it is used to run servers that batch requests until a given size
  or timeout is reached.

  ## Inline/serverless workflow

  First, let's create a simple serving module:

      defmodule MyServing do
        @behaviour Nx.Serving
        import Nx.Defn

        defnp print_and_multiply(x) do
          print_value({:debug, x})
          x * 2
        end

        @impl true
        def init(_inline_or_process, :unused_arg) do            
          {:ok, Nx.Defn.jit(&print_and_multiply/1)}
        end

        @impl true
        def handle_batch(batch, function) do
          {:execute, fn -> {function.(batch), :metadata} end, function}
        end
      end

  It has two functions: `c:init/2`, which receives some metadata
  about the type of serving (`:inline` or `:process`) and the
  serving argument. In this step, we capture `print_and_multiply/1`
  as a jitted function.

  The second function is called `handle_batch/2`. This function
  receives a batch and it must return a function to execute.
  The function itself must return a two element-tuple: the batched
  results and some metadata. The metadata can be any value and we
  set it to the atom `:metadata`.

  Now let's give it a try by defining a serving with our module and
  then running it:

      iex> serving = Nx.Serving.new(MyServing, :unused_arg)
      iex> batch = Nx.Batchs.stack([Nx.tensor([1, 2, 3])])
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

  You should see two values printed. The first is the result of our inspection,
  which shows the tensor that was actually part of the computation and
  how it was batched. Then we see the result of the computation.

  When defining a `Nx.Serving`, we can also customize how the data is
  batched by using the `client_preprocessing` as well as the result by
  using `client_postprocessing` hooks. Let's give it another try:

      iex> serving = (
      ...>   Nx.Serving.new(MyServing, :unused_arg)
      ...>   |> Nx.Serving.client_preprocessing(&Nx.Batch.stack/1)
      ...>   |> Nx.Serving.client_postprocessing(&{:postprocessing, &1, &2})
      ...> )
      iex> Nx.Serving.run(serving, [Nx.tensor([1, 2]), Nx.tensor([3, 4])])
      {:debug, #Nx.Tensor<
        s64[2][2]
        [
          [1, 2],
          [3, 4]
        ]
      >}
      {:post_processing, #Nx.Tensor<
         s64[2][2]
         [
           [2, 4],
           [6, 8]
         ]
       >, :metadata}

  You can see the results are a bit different now. First of all, notice we
  were able to run the serving passing a list of tensors. Our custom
  `client_preprocessing` function stacks those tensors into a batch of two
  entries. With the `client_preprocessing` function, you can transform the
  input in any way you desire before batching. It must return a `Nx.Batch`
  struct. The default client preprocessing simply enforces a batch was given.

  Then the result is a `{:postprocessing, ..., ...}` tuple containing the
  result and the execution metadata as second and third elements respectively.
  From this, we can infer the default implementation of `client_postprocessing`
  simply returns the result, discarding the metadata.

  Why these functions have a `client_` prefix in their name will become clearer
  in the next section.

  ## Stateful/process workflow

  TODO.
  """

  defstruct [:module, :arg, :client_preprocessing, :client_postprocessing]

  @doc """
  The callback used to initialize the serving.

  The first argument reveals if the serving is executed inline,
  such as by calling `run/2`, by started with the process.
  The second argument is the serving argument given to `new/2`.

  It must return `{:ok, state}`, where the `state` can be any term.
  """
  @callback init(:inline | :process, arg :: term) :: {:ok, state :: term}

  @doc """
  Receives a batch and returns a function to execute the batch.

  In case of serving processes, the function is executed is an
  separate process.
  """
  @callback handle_batch(Nx.Batch.t(), state) ::
              {:execute, (() -> {Nx.Container.t(), term}), state}
            when state: term()

  @doc """
  Creates a new serving.

  It expects a module and an argument that is given to its `init` callback.
  """
  def new(module, arg) when is_atom(module) do
    %Nx.Serving{module: module, arg: arg}
  end

  @doc """
  Sets the client preprocessing function.

  The default implementation creates a single element batch
  with the given argument and is equivalent to `&Nx.Batch.stack([&1])`.
  """
  def client_preprocessing(%Nx.Serving{} = serving, function)
      when is_function(function, 1) or is_nil(function) do
    %{serving | client_preprocessing: function}
  end

  @doc """
  Sets the client postprocessing function.

  The default implementation returns the first element given
  to the function.
  """
  def client_postprocessing(%Nx.Serving{} = serving, function)
      when is_function(function, 2) or is_nil(function) do
    %{serving | client_postprocessing: function}
  end

  @doc """
  Runs `serving` with the given `input` inline with the current process.
  """
  def run(%Nx.Serving{} = serving, input) do
    %{
      module: module,
      arg: arg,
      client_preprocessing: preprocessing,
      client_postprocessing: postprocessing
    } = serving

    {:ok, state} = handle_init(module, :inline, arg)
    batch = handle_preprocessing(preprocessing, input)
    {:execute, function, _} = handle_batch(module, batch, state)

    case function.() do
      {output, metadata} ->
        handle_postprocessing(postprocessing, output, metadata)

      other ->
        raise "the function returned by #{inspect(module)}.handle_batch/2 must return {output, metadata}. " <>
                "Got: #{inspect(other)}"
    end
  end

  defp handle_init(module, type, arg) do
    case module.init(type, arg) do
      {:ok, _} = pair ->
        pair

      other ->
        raise "#{inspect(module)}.init/2 must return {:ok, state}. Got: #{inspect(other)}"
    end
  end

  defp handle_batch(module, batch, state) do
    case module.handle_batch(batch, state) do
      {:execute, function, _} = pair when is_function(function, 0) ->
        pair

      other ->
        raise "#{inspect(module)}.handle_batch/2 must return {:execute, function, state}, " <>
                "where function is a function that receives no arguments and returns a tuple. " <>
                "Got: #{inspect(other)}"
    end
  end

  defp handle_preprocessing(nil, input) do
    case input do
      %Nx.Batch{} ->
        input

      _ ->
        raise ArgumentError,
              "the default client_preprocessing expects a `Nx.Batch` as input. " <>
                "Give a batch or use a custom preprocessing"
    end
  end

  defp handle_preprocessing(preprocessing, input) do
    case preprocessing.(input) do
      %Nx.Batch{} = batch ->
        batch

      other ->
        raise "client_preprocessing function #{inspect(preprocessing)} must return a Nx.Batch. " <>
                "Got: #{inspect(other)}"
    end
  end

  defp handle_postprocessing(nil, output, _metadata),
    do: output

  defp handle_postprocessing(postprocessing, output, metadata),
    do: postprocessing.(output, metadata)
end
