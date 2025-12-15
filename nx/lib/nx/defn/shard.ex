defmodule Nx.Defn.Shard do

  @callback __shard_jit__(
    key :: term,
    vars,
    fun :: (vars -> Nx.Container.t()),
    args_list :: [[(-> Nx.Tensor.t())]],
    opts :: keyword
  ) :: [Nx.Container.t()]
  when vars: [Nx.Container.t()]

  def __shard_jit__(fun, params, args_list, opts) do
    {compiler, runtime_fun, opts} = prepare_options(fun, opts)
    compiler.__shard_jit__(fun, params, runtime_fun, args_list, opts)
  rescue
    e in [UndefinedFunctionError] ->
      raise_missing_callback(e, :__shard_jit__, 5, __STACKTRACE__)
  end

  def shard_jit(fun, opts \\ []) when is_function(fun) and is_list(opts) do
    wrap(fun, &shard_jit_apply(fun, &1, opts))
  end


  defp raise_missing_callback(exception, name, arity, stacktrace) do
    case exception do
      %UndefinedFunctionError{module: compiler, function: ^name, arity: ^arity} ->
        raise ArgumentError,
              "the expected compiler callback #{name}/#{arity} is missing. Please check that the module #{inspect(compiler)} is an Nx.Defn.Compiler."

      _ ->
        # This is not an error that should've been caught by this function, so we pass the exception along
        reraise exception, stacktrace
    end
  end

  defp prepare_options(fun, opts) do
    {compiler, opts} = Keyword.pop(opts, :compiler, Nx.Defn.Evaluator)
    {compiler, &runtime_fun(&1, fun, compiler), opts}
  end






  def jit(fun, mesh, opts \\ []) do
    ...
  end

  def jit_apply(fun, mesh, args, opts \\ []) do
    jit(fun, mesh, opts).(args)
  end
end

defmodule Nx.Defn.Shard.Mesh do
  defstruct [:name, :shape]
  @type t :: %__MODULE__{name: String.t(), shape: tuple()}
end
