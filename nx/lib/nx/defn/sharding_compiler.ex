defmodule Nx.Defn.ShardingCompiler do
  @behaviour Nx.Defn.Compiler

  def build_comprehension(slices) do
    args = Macro.generate_arguments(length(slices), nil)

    generators =
      Enum.zip_with(args, slices, fn arg, slice ->
        quote do
          unquote(arg) <- unquote(Macro.escape(slice))
        end
      end)

    {fun, []} =
      Code.eval_quoted(
        quote do
          fn arg ->
            for unquote_splicing(generators) do
              arg[unquote(args)]
            end
          end
        end
      )

    fun
  end

  def compile(fun, args, shard_args) do
    %{data: %{sharding_config: output_config}} = apply(fun, shard_args)

    slices =
      Enum.map(output_config, fn
        nil -> [..]
        %{slices: slices} -> slices
      end)

    dbg(slices)

    slicer = build_comprehension(slices)

    shards = Enum.map(args, slicer)

    funs =
      Enum.zip_with(shards, fn args ->
        Nx.Defn.compile(fun, args, compiler: Nx.Defn.Evaluator)
      end)

    {shards, funs}
  end
end
