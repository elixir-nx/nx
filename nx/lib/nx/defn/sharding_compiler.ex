defmodule Nx.Defn.ShardingCompiler do
  @behaviour Nx.Defn.Compiler


  def __jit__(key, vars, fun, args_list, opts) do
    __compile__(key, vars, fun, opts).(args_list)
  end

  def __compile__(_key, vars, fun, opts) do

  end
end
