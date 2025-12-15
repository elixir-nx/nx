defmodule Nx.Defn.Shard do

  # Exemplo original
  # @callback __shard_jit__(fun, mesh, opts)

  @callback __shard_jit__(
    key :: term,
    mesh :: Nx.Defn.Shard.Mesh.t(),
    vars,
    fun :: (vars -> Nx.Container.t()),
    args_list :: [[(-> Nx.Tensor.t())]],
    opts :: keyword
  ) :: [Nx.Container.t()]
  when vars: [Nx.Container.t()]

  def __shard_jit__(fun, mesh, params, args_list, opts) do
    {module, runtime_fun, opts} = prepare_options(fun, mesh, opts)
    module.__shard_jit__(fun, mesh, params, runtime_fun, args_list, opts)
  rescue
    e in [UndefinedFunctionError] ->
      raise_missing_callback(e, :__shard_jit__, 6, __STACKTRACE__)
  end

  def shard_jit(fun, mesh, opts \\ []) when is_function(fun) and is_list(opts) do
    wrap(fun, &shard_jit_apply(fun, mesh, &1, opts))
  end

  def shard_jit_apply(fun, mesh, args, opts \\ [])
    when is_function(fun) and is_list(args) and is_list(opts) do
  {on_conflict, opts} = Keyword.pop(opts, :on_conflict, :raise)

    cond do
      Nx.Defn.Shard.current() == nil ->
        do_shard_jit_apply(fun, mesh, args, opts)

      on_conflict == :raise ->
        raise "cannot invoke Shard JITed function when there is a Shard JIT compilation happening"

      on_conflict == :force ->
        do_shard_jit_apply(fun, mesh, args, opts)

      on_conflict == :reuse ->
        apply(fun, args)
    end
  end


  defp raise_missing_callback(exception, name, arity, stacktrace) do
    case exception do
      %UndefinedFunctionError{module: module, function: ^name, arity: ^arity} ->
        raise ArgumentError,
              "the expected shard callback #{name}/#{arity} is missing. Please check that the module #{inspect(module)} is an Nx.Defn.Shard."

      _ ->
        # This is not an error that should've been caught by this function, so we pass the exception along
        reraise exception, stacktrace
    end
  end

  defp prepare_options(fun, mesh, opts) do
    {module, opts} = Keyword.pop(opts, :module, Nx.Defn.Evaluator)
    {module, &runtime_fun(&1, fun, mesh, module), opts}
  end

  defp runtime_fun(args, fun, mesh, module) do
    previous_backend = Process.put(Nx.Shared.backend_pdict_key(), {Nx.Defn.Expr, []})
    previous = Process.put(Nx.Defn.Shard, module)

    try do
      fun
      |> apply(args)
      |> Nx.Defn.Composite.traverse(&Nx.Defn.Expr.tensor/1)
    after
      if previous_backend do
        Process.put(Nx.Shared.backend_pdict_key(), previous_backend)
      else
        Process.delete(Nx.Shared.backend_pdict_key())
      end

      if previous do
        Process.put(Nx.Defn.Shard, module)
      else
        Process.delete(Nx.Defn.Shard)
      end
    end
  end

  defp do_shard_jit_apply(fun, mesh, args, opts) do
  opts = prepare_options(opts)
  {fun, params, _templates, flatten} = Nx.Defn.Compiler.to_lazy_params(fun, args)
  [res] = Nx.Defn.Shard.__shard_jit__(fun, mesh, params, [flatten], opts)
  res
end

defmodule Nx.Defn.Shard.Mesh do
  defstruct [:name, :shape]
  @type t :: %__MODULE__{name: String.t(), shape: tuple()}
end

end
