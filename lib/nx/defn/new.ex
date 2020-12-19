defmodule Nx.Defn.Expr do
  @moduledoc false
  # Represents a value at a point of execution in the Nx.Defn AST
  # The value of the Expr is equivalent to the main unit
  # of computation in whatever defn backend you are using
  @enforce_keys [:shape, :op, :value]
  defstruct [:shape, :op, :value]
end

defmodule Nx.Defn.Translation do
  @moduledoc false
  # The purpose of this module is to describe each Nx operation
  # in terms of how it translates the given tensor in space.
  #
  # A translation is simply just a change in shape. As an example,
  # consider calling `Nx.sum/2` on a 3x2x3 tensor along axis 1, the
  # "translation" condenses the shape of the output tensor to a
  # 3x3 tensor.
  #
  # These translation rules remain the same regardless of the defn
  # compiler used, therefore any defn backend needs to ensure their
  # operation semantics enforce these translations.

  ## Unary Ops

  # The shapes of element-wise unary ops remain unchanged regardless
  # of the input.
  def exp(shape), do: shape
end

defmodule Nx.Defn.New do
  @moduledoc false
  # Refactor of the original defn compiler
  alias Nx.Defn.Expr
  alias Nx.Defn.Translation
  alias Nx.Tensor, as: T

  def __compile__(_env, _kind, _meta, _vars, ast, _options) do
    {ast, _state} = traverse(ast, %{})

    quote do
      Nx.Defn.New.evaluate(unquote(ast), Nx.Defn.Default)
    end
  end

  ## AST Evaluation

  # This function evaluates an Expr by traversing the AST
  # and applying it's op to it's value (either an expr or a literal)
  # using the given module
  def evaluate(%Expr{value: %Expr{} = expr, op: op}, module) do
    apply(module, op, [evaluate(expr, module)])
  end

  def evaluate(%Expr{value: value, op: op}, module) do
    apply(module, op, [value])
  end

  ## Expr creation

  defp to_expr(%Expr{} = expr), do: expr

  defp to_expr(integer) when is_integer(integer),
    do: %Expr{value: integer, op: :tensor, shape: {}}

  defp to_expr(number) when is_number(number),
    do: %Expr{value: number, op: :tensor, shape: {}}

  defp to_expr(%T{shape: shape} = t), do: %Expr{value: t, op: :tensor, shape: shape}

  ## Operations

  def exp(expr) do
    %Expr{shape: shape} = expr = to_expr(expr)
    output_shape = Translation.exp(shape)
    %Expr{shape: output_shape, op: :exp, value: expr}
  end

  ## Traversal

  defp traverse({{:., dot_meta, [Nx, name]}, meta, args}, state) do
    {args, state} = traverse(args, state)
    {to_expr_call(dot_meta, meta, :"#{name}", args), state}
  end

  defp traverse({left, right}, state) do
    {left, state} = traverse(left, state)
    {right, state} = traverse(right, state)
    {{left, right}, state}
  end

  defp traverse({name, meta, args}, state) do
    {args, state} = traverse(args, state)
    {{name, meta, args}, state}
  end

  defp traverse(list, state) when is_list(list) do
    Enum.map_reduce(list, state, &traverse/2)
  end

  defp traverse(other, state) do
    {other, state}
  end

  defp to_expr_call(dot_meta, meta, fun, args) do
    {{:., dot_meta, [__MODULE__, fun]}, meta, args}
  end
end

defmodule Nx.Defn.Default do
  @moduledoc false

  defdelegate tensor(t), to: Nx
  defdelegate exp(t), to: Nx
end