defmodule Nx.Defn.Graph do
  @moduledoc """
  A module for splitting `Nx.Defn.Expr` into stages.

  This module is used to split an `Nx.Defn.Expr` into stages, which are then
  executed in a chain.

  `split/2` and `t:Stage.t()` describe how to split
  the graph and what's the expected result.

  `run/2` executes the given graph against the provided arguments in a sequential manner.
  """
  alias Nx.Defn.Composite

  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  defmodule Stage do
    @typedoc """
    A stage in the graph splitter.

      * `:arguments`: a list of maps that point to the source from which to fetch the corresponding
        value for the given argument.
      * `:expr`: the expression that represents the computation for the Stage.
      * `:id`: the unique id for the Stage.
    """
    @type t :: %__MODULE__{
            id: reference(),
            expr: %{__struct__: Nx.Defn.Expr},
            arguments: [%{source: {reference() | nil, non_neg_integer()}}]
          }

    defstruct [:id, :expr, :arguments]
  end

  @doc """
  Splits the received Nx.Defn.Expr into stages given the rules.

  `expr_split_fn` is a function that receives an `Nx.Tensor` containing an `Nx.Defn.Expr`
  and returns `true` when a split must happen, and `false` otherwise.

  ## Examples

      iex> expr = Nx.Defn.debug_expr(fn x, y -> x |> Nx.negate() |> Nx.sin() |> Nx.cos() |> Nx.add(y) end).(1, 2)
      iex> [stage0, stage1] = Nx.Defn.Graph.split(expr, fn %Nx.Tensor{data: %Nx.Defn.Expr{op: op}} -> op == :cos end)
      iex> {out0} = stage0.expr
      iex> out0
      #Nx.Tensor<
        f32
        \n\
        Nx.Defn.Expr
        parameter a:0   s32
        b = negate a    s32
        c = sin b       f32
      >
      iex> stage1.expr
      #Nx.Tensor<
        f32
        \n\
        Nx.Defn.Expr
        parameter a:1   f32
        parameter c:0   s32
        b = cos a       f32
        d = add b, c    f32
      >
  """
  def split(expr, expr_split_fn) when is_function(expr_split_fn, 1) do
    {chain, _, _} = __split__(expr, expr_split_fn)
    chain
  end

  @doc """
  Executes the stage chain with the given arguments.
  """
  def run(chain, args) do
    scope =
      Enum.with_index(args, fn arg, idx -> {{nil, idx}, arg} end)
      |> Map.new()

    {result, _scope} =
      Enum.reduce(chain, {nil, scope}, fn stage, {_result, scope} ->
        %{id: id, expr: expr, arguments: arguments} = stage

        args =
          Enum.map(arguments, fn %{source: source} ->
            Map.fetch!(scope, source)
          end)

        case Nx.Defn.jit_apply(fn _ -> expr end, [List.to_tuple(args)]) do
          %T{} = tensor ->
            {tensor, Map.put(scope, {id, 0}, tensor)}

          tuple ->
            {_idx, scope} =
              tuple
              |> Tuple.to_list()
              |> Enum.reduce({0, scope}, fn tensor, {idx, scope} ->
                {idx + 1, Map.put(scope, {id, idx}, tensor)}
              end)

            {tuple, scope}
        end
      end)

    result
  end

  @doc false
  def __split__(expr, expr_split_fn) do
    # state.expression_chain is a reverse accumulation of the stages and
    # snapshots of the state at each one so that we can properly remap parameters for each stage.
    state = %{
      expression_chain: [],
      nodes_to_replace: %{},
      expr_split_fn: expr_split_fn,
      # args is a map of id -> {stage_id, output_container_position}
      args: %{}
    }

    cache = %{}
    {expr, {cache, state}} = composite_eval(expr, state, cache)

    expr_chain =
      Enum.reduce(
        [{make_ref(), expr, state.nodes_to_replace} | state.expression_chain],
        [],
        fn {id, expr, nodes_to_replace}, acc ->
          # TO-DO: we need to also do a pass to avoid recalculating results that have been previously calculated.
          # For example:
          # x = arg0 + arg1
          # y = arg0 - arg1
          # z = x + y
          # -----
          # w = dot(z, arg1)
          # y + w <- here, we currently have to recalculate y given that only z, arg0 and arg1 will be passed as arguments.
          #          ideally, we should also pass y as a value to avoid recalculating it.
          #          We might be able to calculate this in the first traversal somehow.

          {expr, %{used_args: used_args}} =
            composite_rewrite_subtree(
              expr,
              %{state | nodes_to_replace: nodes_to_replace}
            )

          arg_remapping =
            used_args
            |> Enum.sort_by(fn {_id, %T{data: %Expr{op: :parameter, args: [idx]}}} -> idx end)
            |> Enum.with_index(fn
              {id, expr}, idx ->
                {id, put_in(expr.data.args, [idx])}
            end)
            |> Map.new()

          {expr, _} =
            composite_rewrite_subtree(expr, %{state | nodes_to_replace: arg_remapping})

          arguments =
            arg_remapping
            |> Enum.map(fn {_id, arg_expr} ->
              id = arg_expr.data.id
              [idx] = arg_expr.data.args
              source = Map.fetch!(state.args, id)
              {idx, %{source: source}}
            end)
            |> Enum.sort_by(fn {idx, _} -> idx end)
            |> Enum.map(fn {_, arg} -> arg end)

          [
            %Stage{
              id: id,
              expr: expr,
              arguments: arguments
            }
            | acc
          ]
        end
      )

    {expr_chain, cache, Map.delete(state, :expression_chain)}
  end

  defp composite_eval(expr, state, cache) do
    Composite.traverse(expr, {cache, state}, &eval/2)
  end

  defp eval(%T{data: %Expr{id: id, op: op}} = ans, {cache, state}) do
    case {cache, state.nodes_to_replace} do
      {_, %{^id => res}} ->
        # Replace the node with the corresponding parameter
        {res, {Map.put(cache, id, res), state}}

      {%{^id => res}, _} ->
        {res, {cache, state}}

      _ ->
        if state.expr_split_fn.(ans) do
          split_expr(ans, {cache, state})
        else
          eval_apply(op, ans, {cache, state})
        end
    end
  end

  defp eval(other, {cache, state}) do
    {other, {cache, state}}
  end

  defp split_expr(expr, {cache, state}) do
    {args, {cache, state}} = Nx.Defn.Tree.apply_args(expr, {cache, state}, &eval/2)
    # We need to save this so that each previous stage
    # isn't affected by following ones
    nodes_to_replace = state.nodes_to_replace

    stage_id = make_ref()

    {args, {tensor_args, _out_position, state}} =
      Enum.map_reduce(args, {[], 0, state}, fn
        %T{} = expr, {tensor_args, out_position, state} ->
          arg = Expr.parameter(expr, map_size(state.args))

          state = %{
            state
            | args: Map.put(state.args, arg.data.id, {stage_id, out_position}),
              nodes_to_replace: Map.put(state.nodes_to_replace, expr.data.id, arg)
          }

          {arg, {[expr | tensor_args], out_position + 1, state}}

        non_tensor_arg, acc ->
          {non_tensor_arg, acc}
      end)

    new_expr = put_in(expr.data.args, args)

    state =
      update_in(
        state.expression_chain,
        &[
          {stage_id, List.to_tuple(Enum.reverse(tensor_args)), nodes_to_replace}
          | &1
        ]
      )

    cache = Map.put(cache, new_expr.data.id, new_expr)

    {new_expr, {cache, state}}
  end

  defp eval_apply(:parameter, %T{data: %Expr{id: id, args: [idx]}} = expr, {cache, state}) do
    state = put_in(state.args[id], {nil, idx})
    {expr, {Map.put(cache, id, expr), state}}
  end

  defp eval_apply(:elem, %T{data: %Expr{id: id, args: [tuple, i]}}, {cache, state}) do
    {tuple, cache} = composite_eval(tuple, state, cache)
    res = elem(tuple, i)
    {res, {Map.put(cache, id, res), state}}
  end

  defp eval_apply(_op, %T{data: %Expr{id: id}} = ans, {cache, state}) do
    {args, {cache, state}} = Nx.Defn.Tree.apply_args(ans, {cache, state}, &eval/2)
    ans = put_in(ans.data.args, args)
    {ans, {Map.put(cache, id, ans), state}}
  end

  defp composite_rewrite_subtree(container, state, acc \\ %{used_args: %{}})

  defp composite_rewrite_subtree(container, state, acc) when is_list(container) do
    Enum.map_reduce(container, acc, fn
      %T{} = arg, acc ->
        composite_rewrite_subtree(arg, state, acc)

      arg, acc when is_list(arg) ->
        composite_rewrite_subtree(arg, state, acc)

      arg, acc ->
        {arg, acc}
    end)
  end

  defp composite_rewrite_subtree(container, state, acc) do
    Composite.traverse(container, acc, &rewrite_subtree(&1, state, &2))
  end

  defp rewrite_subtree(%T{data: %Expr{id: id, op: :parameter}} = expr, state, acc) do
    case state.nodes_to_replace do
      %{^id => res} ->
        {res, put_in(acc.used_args[id], res)}

      _ ->
        {expr, put_in(acc.used_args[id], expr)}
    end
  end

  defp rewrite_subtree(
         %T{data: %Expr{op: :optional, id: id, args: [call, subexpr, fun]}} = expr,
         state,
         acc
       ) do
    case state.nodes_to_replace do
      %{^id => res} ->
        {res, put_in(acc.used_args[id], res)}

      _ ->
        {call, acc} = rewrite_subtree(call, state, acc)
        # `subexpr` is hermetic, in the sense that it is a self-contained scope
        # from which the arguments always come from `call`, so we can
        # keep it as is.

        {put_in(expr.data.args, [call, subexpr, fun]), acc}
    end
  end

  defp rewrite_subtree(%T{data: %Expr{id: id, args: args}} = expr, state, acc) do
    case state.nodes_to_replace do
      %{^id => res} ->
        # nodes_to_replace always contains a param
        {res, put_in(acc.used_args[id], res)}

      _ ->
        {args, acc} = composite_rewrite_subtree(args, state, acc)
        {put_in(expr.data.args, args), acc}
    end
  end

  defp rewrite_subtree(other, _, acc), do: {other, acc}
end
