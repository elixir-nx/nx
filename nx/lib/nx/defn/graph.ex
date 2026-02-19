defmodule Nx.Defn.Graph do
  @moduledoc """
  A module for splitting `Nx.Defn.Expr` into stages.

  This module is used to split an `Nx.Defn.Expr` into stages,
  which are then executed in a chain.

  `split/2` and `t:Stage.t()` describe how to split
  the graph and what's the expected result.

  `run/2` executes the given graph against the provided arguments
  in a sequential manner.
  """
  alias Nx.Defn.Composite

  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  defmodule Stage do
    @moduledoc false

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
  Splits the received Nx.Defn.Expr into stages based on each tensor.

  `expr_split_fn` is a function that receives an `Nx.Tensor` containing an `Nx.Defn.Expr`
  and returns one of:

  * `:before` - creates a stage that computes all arguments to the current node,
    then creates parameters for those arguments in subsequent stages
  * `:after` - creates a stage that computes the current node and outputs it
  * `:both` - applies both `:before` and `:after` in sequence, creating stages for dependencies and the target operation
  * `:none` - no split occurs

  ## Examples

      iex> expr = Nx.Defn.debug_expr(fn x, y -> x |> Nx.negate() |> Nx.sin() |> Nx.cos() |> Nx.add(y) end).(1, 2)
      iex> [stage0, stage1] = Nx.Defn.Graph.split(expr, fn %Nx.Tensor{data: %Nx.Defn.Expr{op: op}} -> if op == :cos, do: :before, else: :none end)
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
    normalized_fn = fn node, acc ->
      decision = expr_split_fn.(node)
      normalized_decision = normalize_split_decision(decision)
      {normalized_decision, acc}
    end

    {chain, _, _} = __split__(expr, nil, normalized_fn)
    chain
  end

  @doc """
  Splits the received Nx.Defn.Expr into stages based on each tensor and the accumulator.

  `expr_split_fn` is a function that receives an `Nx.Tensor` and the accumulator,
  returning `{decision, new_acc}` where `decision` is one of:

  * `:before` - creates a stage that computes all arguments to the current node,
    then creates parameters for those arguments in subsequent stages
  * `:after` - creates a stage that computes the current node and outputs it
  * `:both` - applies both `:before` and `:after` in sequence, creating stages for dependencies and the target operation
  * `:none` - no split occurs

  The decision to split is made based on the expression and the accumulator.
  This allows for more complex decisions to be made, such as splitting every 3 operations as in the example below.

      # Count operations and split every 3 operations
      split_fn = fn _tensor, count ->
        new_count = count + 1
        decision = if count > 0 and rem(new_count, 3) == 0, do: :before, else: :none
        {decision, new_count}
      end

      stages = Nx.Defn.Graph.split(expr, 0, split_fn)
  """
  def split(expr, initial_acc, expr_split_fn) when is_function(expr_split_fn, 2) do
    normalized_fn = fn node, acc ->
      {decision, new_acc} = expr_split_fn.(node, acc)
      normalized_decision = normalize_split_decision(decision)
      {normalized_decision, new_acc}
    end

    {chain, _, _} = __split__(expr, initial_acc, normalized_fn)
    chain
  end

  # Normalizes split decisions
  defp normalize_split_decision(:before), do: :before
  defp normalize_split_decision(:after), do: :after
  defp normalize_split_decision(:both), do: :both
  defp normalize_split_decision(:none), do: :none

  defp normalize_split_decision(other) do
    raise ArgumentError,
          "Invalid split decision: #{inspect(other)}. Expected :before, :after, :both, or :none"
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
  def __split__(expr, initial_acc, expr_split_fn) do
    # Normalize the callback to handle both old and new formats
    normalized_fn = fn node, acc ->
      result = expr_split_fn.(node, acc)

      case result do
        {decision, new_acc} ->
          # New format: {decision, new_acc}
          {normalize_split_decision(decision), new_acc}

        decision when is_boolean(decision) or decision in [:before, :after, :both, :none] ->
          # Old format: just the decision (for arity-1 callbacks wrapped by split/2)
          {normalize_split_decision(decision), acc}
      end
    end

    # state.expression_chain is a reverse accumulation of the stages and
    # snapshots of the state at each one so that we can properly remap parameters for each stage.
    state = %{
      expression_chain: [],
      nodes_to_replace: %{},
      expr_split_fn: normalized_fn,
      split_acc: initial_acc,
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

          {arg_remapping, _, _} =
            used_args
            |> Enum.sort_by(fn {_id, %T{data: %Expr{op: :parameter, args: [idx]}}} -> idx end)
            |> Enum.reduce({%{}, %{}, 0}, fn
              {id, expr}, {acc, sources, idx} ->
                # For replacement parameters, use the original parameter ID to find the source
                id = if Map.has_key?(state.args, expr.data.id), do: expr.data.id, else: id
                source = Map.fetch!(state.args, id)

                if visited_expr = Map.get(sources, source) do
                  {Map.put(acc, id, visited_expr), sources, idx}
                else
                  expr = put_in(expr.data.args, [idx])
                  {Map.put(acc, id, expr), Map.put(sources, source, expr), idx + 1}
                end
            end)

          {expr, _} =
            composite_rewrite_subtree(expr, %{state | nodes_to_replace: arg_remapping})

          # Create arguments list from final remapping, preserving the deduplicated order
          arguments =
            arg_remapping
            |> Enum.map(fn {original_id, arg_expr} ->
              [idx] = arg_expr.data.args
              # Use the same logic as above to find the correct source
              source_id =
                if Map.has_key?(state.args, arg_expr.data.id),
                  do: arg_expr.data.id,
                  else: original_id

              source = Map.fetch!(state.args, source_id)
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
        case op do
          :parameter ->
            eval_apply(:parameter, ans, {cache, state})

          :elem ->
            eval_apply(:elem, ans, {cache, state})

          _ ->
            # First process the arguments with the current accumulator
            {args, {cache, state}} = Nx.Defn.Tree.apply_args(ans, {cache, state}, &eval/2)

            # Then check if we should split based on this node
            {split_decision, new_acc} = state.expr_split_fn.(ans, state.split_acc)
            state = %{state | split_acc: new_acc}

            case split_decision do
              :none ->
                # No split - apply the operation with the processed args
                ans = put_in(ans.data.args, args)
                {ans, {Map.put(cache, ans.data.id, ans), state}}

              :before ->
                # Use the already processed args for splitting
                split_before(ans, args, {cache, state})

              :after ->
                split_after(ans, args, {cache, state})

              :both ->
                split_both(ans, args, {cache, state})
            end
        end
    end
  end

  defp eval(other, {cache, state}) do
    {other, {cache, state}}
  end

  defp split_before(expr, args, {cache, state}) do
    # We need to save this so that each previous stage
    # isn't affected by following ones
    nodes_to_replace = state.nodes_to_replace

    stage_id = make_ref()

    {args, {tensor_args, _out_position, state}} =
      Enum.map_reduce(args, {[], 0, state}, fn
        %T{data: %Expr{op: :parameter}} = arg, {tensor_args, out_position, state} ->
          # Parameters are not computed values, so don't add them to tensor_args
          # Just update the state if needed
          state =
            case Map.has_key?(state.args, arg.data.id) do
              false ->
                %{state | args: Map.put(state.args, arg.data.id, {stage_id, out_position})}

              true ->
                state
            end

          {arg, {tensor_args, out_position, state}}

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

    {stage_expr, result_expr} =
      case {tensor_args, expr.data.op, args} do
        {_, :metadata, [wrapped_expr, _]} when is_tuple(wrapped_expr) ->
          # We're effectively splitting on a tuple, so we need to create a
          # stage output for each element
          {wrapped_expr, new_expr}

        {[], _, _} ->
          # No intermediate computations - create a parameter for this split operation
          # The current expression will be computed in the next stage
          param = Expr.parameter(new_expr, map_size(state.args))
          {{param}, param}

        _ ->
          # There are intermediate computations - only include those in the current stage
          # The current expression will be computed in the next stage using these outputs
          stage_expr = List.to_tuple(Enum.reverse(tensor_args))
          {stage_expr, new_expr}
      end

    # Update state with parameter mapping if we created one
    state =
      case {tensor_args, expr.data.op, args} do
        {_, :metadata, [wrapped_expr, _]} when is_tuple(wrapped_expr) ->
          # Register each tuple element as a stage output and create a replacement parameter
          {state, _} =
            wrapped_expr
            |> Tuple.to_list()
            |> Enum.reduce({state, 0}, fn %T{} = elem_expr, {state, index} ->
              param = Expr.parameter(elem_expr, index)

              state = %{
                state
                | args:
                    state.args
                    |> Map.put(elem_expr.data.id, {stage_id, index})
                    |> Map.put(param.data.id, {stage_id, index}),
                  nodes_to_replace: Map.put(state.nodes_to_replace, elem_expr.data.id, param)
              }

              {state, index + 1}
            end)

          state

        {[], _, _} ->
          # Add parameter mapping and node replacement for the split operation
          # Extract the parameter from the tuple
          param = elem(stage_expr, 0)

          %{
            state
            | args: Map.put(state.args, param.data.id, {stage_id, 0}),
              nodes_to_replace: Map.put(state.nodes_to_replace, new_expr.data.id, param)
          }

        _ ->
          state
      end

    state =
      update_in(
        state.expression_chain,
        &[
          {stage_id, stage_expr, nodes_to_replace}
          | &1
        ]
      )

    cache = Map.put(cache, result_expr.data.id, result_expr)

    {result_expr, {cache, state}}
  end

  defp split_after(expr, args, {cache, state}) do
    # For :after mode, we create a stage that computes the current node
    nodes_to_replace = state.nodes_to_replace
    stage_id = make_ref()

    # The stage computes the current expression with its original args
    new_expr = put_in(expr.data.args, args)
    stage_expr = {new_expr}

    # Create a parameter that represents the output of this stage
    result_param = Expr.parameter(new_expr, map_size(state.args))

    # Update state to track this stage output
    state = %{
      state
      | args: Map.put(state.args, result_param.data.id, {stage_id, 0}),
        nodes_to_replace: Map.put(state.nodes_to_replace, new_expr.data.id, result_param)
    }

    state =
      update_in(
        state.expression_chain,
        &[
          {stage_id, stage_expr, nodes_to_replace}
          | &1
        ]
      )

    cache = Map.put(cache, result_param.data.id, result_param)

    {result_param, {cache, state}}
  end

  defp split_both(expr, args, {cache, state}) do
    # For :both mode, we need to check if split_before would create intermediate computations
    # We use the same logic as split_before to determine this

    tensor_args =
      Enum.reduce(args, [], fn
        %T{data: %Expr{op: :parameter}}, acc -> acc
        %T{} = tensor_expr, acc -> [tensor_expr | acc]
        _, acc -> acc
      end)

    # Check if split_before would create a meaningful stage or just a parameter wrapper
    has_intermediate_computations =
      case {tensor_args, expr.data.op} do
        # No tensor args means no intermediate computations
        {[], _} -> false
        # Metadata operations always create meaningful stages
        {_, :metadata} -> true
        # Non-empty tensor_args means intermediate computations
        {_non_empty, _} -> true
      end

    case has_intermediate_computations do
      false ->
        # No intermediate computations - skip :before and go straight to :after
        split_after(expr, args, {cache, state})

      true ->
        # There are intermediate computations - do :before then :after
        {before_result, {cache, state}} = split_before(expr, args, {cache, state})

        # Now apply :after to the before_result
        # The before_result should be the new_expr with parameterized args
        split_after(before_result, before_result.data.args, {cache, state})
    end
  end

  defp eval_apply(:parameter, %T{data: %Expr{id: id, args: [idx]}} = expr, {cache, state}) do
    state = put_in(state.args[id], {nil, idx})
    {expr, {Map.put(cache, id, expr), state}}
  end

  defp eval_apply(:elem, %T{data: %Expr{id: id, args: [tuple, i]}} = expr, {cache, state}) do
    {tuple, {cache, state}} = composite_eval(tuple, state, cache)

    res =
      case tuple do
        t when is_tuple(t) -> elem(t, i)
        %T{} -> put_in(expr.data.args, [tuple, i])
      end

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
        # This parameter is being replaced by a stage output - collect the replacement
        # We collect both the original id and the replacement id to ensure proper tracking
        acc = put_in(acc.used_args[id], res)
        acc = put_in(acc.used_args[res.data.id], res)
        {res, acc}

      _ ->
        # This is an original parameter - collect it
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

  defp rewrite_subtree(
         %T{data: %Expr{id: id, op: :elem, args: [tuple_expr, index]}} = expr,
         state,
         acc
       ) do
    case state.nodes_to_replace do
      %{^id => res} ->
        {res, put_in(acc.used_args[id], res)}

      _ ->
        {tuple_expr, acc} = rewrite_subtree(tuple_expr, state, acc)

        case tuple_expr do
          # Literal tuple: turn elem into a parameter for that element
          t when is_tuple(t) ->
            elem_expr = elem(t, index)
            param = Expr.parameter(elem_expr, index)
            {param, put_in(acc.used_args[elem_expr.data.id], param)}

          # Metadata-wrapped tuple: same as above
          %T{data: %Expr{op: :metadata, args: [wrapped, _]}} when is_tuple(wrapped) ->
            elem_expr = elem(wrapped, index)
            param = Expr.parameter(elem_expr, index)
            {param, put_in(acc.used_args[elem_expr.data.id], param)}

          # Tuple tensor: create a parameter pointing to this index
          %T{type: {:tuple, _}} ->
            param = Expr.parameter(expr, index)
            {param, put_in(acc.used_args[param.data.id], param)}

          _ ->
            {put_in(expr.data.args, [tuple_expr, index]), acc}
        end
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
