defmodule Nx.TemplateBackend do
  @moduledoc """
  An opaque backend written that is used as template
  to declare the type, shape, and names of tensors to
  be expected in the future.

  It doesn't perform any operation, it always raises.
  """

  @behaviour Nx.Backend
  defstruct []

  @impl true
  def inspect(%Nx.Tensor{}, opts) do
    if opts.custom_options[:skip_template_backend_header] do
      Inspect.Algebra.empty()
    else
      "Nx.TemplateBackend"
    end
  end

  funs =
    Nx.Backend.behaviour_info(:callbacks) --
      (Nx.Backend.behaviour_info(:optional_callbacks) ++ Module.definitions_in(__MODULE__, :def))

  for {fun, arity} <- funs do
    args = Macro.generate_arguments(arity, __MODULE__)

    @impl true
    def unquote(fun)(unquote_splicing(args)) do
      raise "cannot perform operations on a Nx.TemplateBackend tensor"
    end
  end
end
