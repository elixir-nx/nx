defmodule Nx.Defn.PrintQuotedTransform do
  @moduledoc """
  A transform that prints and returns the expanded numerical expression.
  """

  @behaviour Nx.Defn.Transform

  @impl true
  def __transform__(_env, version, _meta, ast, _opts) do
    IO.puts(Macro.to_string(normalize_vars(ast)))
    {version, ast}
  end

  defguardp is_var(var)
            when is_tuple(var) and tuple_size(var) == 3 and is_atom(elem(var, 0)) and
                   is_atom(elem(var, 2))

  defguardp is_underscore(var)
            when is_tuple(var) and tuple_size(var) == 3 and elem(var, 0) == :_ and
                   is_atom(elem(var, 2))

  @doc """
  Normalizes variables based on their counters.
  """
  def normalize_vars(ast) do
    {ast, _map} =
      Macro.prewalk(ast, %{counter: 0}, fn
        var, map when is_var(var) and not is_underscore(var) ->
          var_counter = var_counter(var)

          case map do
            %{^var_counter => var_name} ->
              {{var_name, [], nil}, map}

            %{} ->
              var_name =
                map.counter
                |> counter_to_name()
                |> IO.iodata_to_binary()
                |> String.to_atom()

              map = put_in(map[var_counter], var_name)
              map = update_in(map.counter, &(&1 + 1))
              {{var_name, [], nil}, map}
          end

        expr, map ->
          {expr, map}
      end)

    ast
  end

  defp var_counter({var, meta, ctx}) when is_atom(var) and is_atom(ctx) do
    Keyword.fetch!(meta, :counter)
  end

  defp counter_to_name(counter) when counter >= 26 do
    [counter_to_name(div(counter, 26)) | counter_to_name(rem(counter, 26))]
  end

  defp counter_to_name(counter), do: [Enum.at(?a..?z, counter)]
end
