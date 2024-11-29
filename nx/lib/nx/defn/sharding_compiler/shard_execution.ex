defmodule Nx.Defn.ShardingCompiler.ShardExecution do
  # processes a single shard of an output entry, given the corresponding input data sections (1 per input)
  defstruct [
    :compiled_fun,
    :stage,
    :input_data_sections,
    :output_entry_index,
    :output_data_section_id,
    :output_starts,
    :output_lengths,
    :fetched_inputs,
    :output
  ]

  use GenServer

  alias Nx.Defn.ShardingCompiler.Passes.GraphSplitter.Stage
  alias Nx.Defn.ShardingCompiler.ShardRegistry

  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  def init([
        %Stage{} = stage,
        input_data_sections,
        output_entry_index,
        output_data_section_id,
        output_starts,
        output_lengths
      ]) do
    Process.send_after(self(), :fetch_inputs, 0)

    require IEx
    IEx.pry()

    input_data_sections =
      input_data_sections
      |> Enum.sort_by(fn {idx, _} -> idx end)
      |> Enum.with_index(fn {_idx, {arg_id, shard_ids}}, idx ->
        {idx, {arg_id, shard_ids}}
      end)

    fetched_inputs =
      Map.new(input_data_sections, fn
        {idx, {arg_id, nil}} ->
          {arg_id, {idx, 0}}

        {_idx, {arg_id, _}} ->
          {arg_id, nil}
      end)

    arg_templates =
      input_data_sections
      |> Enum.with_index(fn
        {_idx, {arg_id, :scalar}}, idx ->
          arg = stage.arguments[arg_id]
          shape = arg.shape
          type = arg.type

          arg = %T{
            data: nil,
            shape: shape,
            type: type,
            names: List.duplicate(nil, tuple_size(shape))
          }

          Expr.parameter(arg, :root, idx)

        {_idx, {_arg_id, nil}}, idx ->
          # TO-DO: this is not the proper way to handle this case
          # as we should be keeping the container together.
          # This is a hack so we can get the POC running.
          arg = %T{
            data: nil,
            shape: {},
            type: {:u, 8},
            names: []
          }

          Expr.parameter(arg, :root, idx)

        {_idx, {arg_id, _shard_ids}}, idx ->
          arg = stage.arguments[arg_id]

          arg = %T{
            data: nil,
            shape: arg.shape,
            type: arg.type,
            names: arg.names
          }

          Expr.parameter(arg, :root, idx)
      end)

    compiled_fun =
      Nx.Defn.Evaluator.__compile__(
        make_ref(),
        arg_templates,
        fn _ ->
          if is_tuple(stage.expr) do
            elem(stage.expr, output_entry_index)
          else
            stage.expr
          end
        end,
        []
      )

    fun = fn [args] ->
      [res] =
        compiled_fun.([
          Enum.map(Tuple.to_list(args), fn arg ->
            fn -> arg end
          end)
        ])

      res
    end

    {:ok,
     %__MODULE__{
       stage: stage,
       input_data_sections: input_data_sections,
       output_entry_index: output_entry_index,
       output_data_section_id: output_data_section_id,
       output_starts: output_starts,
       output_lengths: output_lengths,
       fetched_inputs: fetched_inputs,
       # TO-DO: pass compiled_fun as argument
       compiled_fun: fun
     }}
  end

  def start_link(args) do
    GenServer.start_link(__MODULE__, args, name: via_tuple(args))
  end

  defp via_tuple([
         stage,
         _input_data_sections,
         output_entry_index,
         output_data_section_id,
         _starts,
         _lengths
       ]) do
    {:via, Registry, {ShardRegistry, {stage.id, output_entry_index, output_data_section_id}}}
  end

  def handle_info(:fetch_inputs, state) do
    state =
      for {arg_idx, {arg_id, data_section_id}} <- state.input_data_sections,
          is_nil(state.fetched_inputs[arg_id]),
          reduce: state do
        state ->
          {stage_id, stage_idx} = state.stage.argument_sources[arg_id]

          case get(stage_id, stage_idx, data_section_id) do
            {:ok, data} ->
              put_in(state.fetched_inputs[arg_id], {arg_idx, data})

            {:error, :pending} ->
              state
          end
      end

    if Enum.any?(state.fetched_inputs, fn {_arg_id, data} -> is_nil(data) end) do
      Process.send_after(self(), :fetch_inputs, 10)
      {:noreply, state}
    else
      state = compute(state)
      {:noreply, state}
    end
  end

  def handle_info(_, state) do
    {:noreply, state}
  end

  def get(stage_id, stage_idx, data_section_id) do
    key = {stage_id, stage_idx, data_section_id}

    case ShardRegistry.lookup(key) do
      {:ok, pid} -> GenServer.call(pid, :get)
      {:error, :pending} -> {:error, :pending}
    end
  end

  def handle_call(:get, _from, state) do
    result =
      case state.output do
        nil -> {:error, :pending}
        data -> {:ok, {data, state.output_starts, state.output_lengths}}
      end

    {:reply, result, state}
  end

  defp compute(state) do
    args =
      state.fetched_inputs
      |> Enum.map(fn {_id, {idx, data}} -> {idx, data} end)
      |> Enum.sort()
      |> Enum.map(fn {_idx, data} -> data end)
      |> List.to_tuple()

    output = state.compiled_fun.([args])

    %{state | output: output}
  end
end
