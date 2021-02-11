defmodule Nx.Defn.Module do
  @moduledoc false

  # TODO: remove this module once Elixir v1.12 is out.

  @doc """
  Returns the definition for the name-arity pair.

  It returns a tuple with the `version`, the `kind`,
  the definition `metadata`, and a list with each clause.
  Each clause is a four-element tuple with metadata,
  the arguments, the guards, and the clause AST.

  The clauses are returned in the expanded AST format,
  which is a subset of Elixir's AST but already normalized.
  This makes it a useful AST for analyzing code but it
  cannot be reinjected into the module as it may have
  lost some of its original context. Given this AST
  representation is mostly internal, it is versioned
  and it may change at any time. Therefore, **use this
  API with caution**.
  """
  # @spec get_definition(module, definition) ::
  #         {:v1, kind, meta :: keyword,
  #          [{meta :: keyword, arguments :: [Macro.t()], guards :: [Macro.t()], Macro.t()}]}
  def get_definition(module, {name, arity})
      when is_atom(module) and is_atom(name) and is_integer(arity) do
    # assert_not_compiled!(__ENV__.function, module, @extra_error_msg_definitions_in)
    {set, bag} = data_tables_for(module)

    case :ets.lookup(set, {:def, {name, arity}}) do
      [{_key, kind, meta, _, _, _}] ->
        {:v1, kind, meta, bag_lookup_element(bag, {:clauses, {name, arity}}, 2)}

      [] ->
        nil
    end
  end

  @doc """
  Deletes a definition from a module.

  It returns true if the definition exists and it was removed,
  otherwise it returns false.
  """
  # @spec delete_definition(module, definition) :: boolean()
  def delete_definition(module, {name, arity})
      when is_atom(module) and is_atom(name) and is_integer(arity) do
    # assert_not_readonly!(__ENV__.function, module)

    case :elixir_def.take_definition(module, {name, arity}) do
      false ->
        false

      _ ->
        :elixir_locals.yank({name, arity}, module)
        true
    end
  end

  ## These helpers already exist in Elixir's lib/module.ex

  defp data_tables_for(module) do
    :elixir_module.data_tables(module)
  end

  defp bag_lookup_element(table, key, pos) do
    :ets.lookup_element(table, key, pos)
  catch
    :error, :badarg -> []
  end
end
