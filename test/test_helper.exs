target = System.get_env("EXLA_TARGET", "host")

ExUnit.start(
  exclude: :platform,
  include: [platform: String.to_atom(target)],
  assert_receive_timeout: 1000
)

  defmodule GradHelpers do
    import Nx.Shared

    # Check the gradient of numerical function `func`.
    #
    # You must hold the function constant on every other
    # variable with a partial application of `func`:
    #
    # check_grads(& my_function(1.0, 1.0, &1, 1.0))
    def check_grads(func, grad_func, x, eps \\ 1.0e-4) do
      est_grad = finite_differences(func, x, eps)
      comp_grad = grad_func.(x)
      approx_equal?(est_grad, comp_grad, eps)
    end

    # Determines if `lhs` approx. equals `rhs` given
    # `eps`
    #
    # TODO: defn/simplify when predicates are worked out
    def approx_equal?(lhs, rhs, eps) do
      output_type = Nx.Type.merge(lhs.type, rhs.type)
      binary = Nx.Util.to_bitstring(Nx.abs(Nx.subtract(lhs, rhs)))
      value =
        match_types [output_type] do
          <<match!(var, 0)>> = binary
          read!(var, 0)
        end

      value < eps
    end

    # Numerical method for estimating the gradient
    # of `func` with respect to `x` using the finite
    # difference `eps`
    def finite_differences(func, x, eps) do
      Nx.divide(
        Nx.subtract(
          func.(Nx.add(x, Nx.divide(eps, 2.0))),
          func.(Nx.subtract(x, Nx.divide(eps, 2.0)))
        ),
        eps
      )
    end
  end

defmodule ExlaHelpers do
  @doc """
  Returns the default Exla client.
  """
  def client(), do: Exla.Client.fetch!(:default)

  @doc """
  Compiles the given function.

  It expects a list of shapes which will be given as parameters.
  """
  def compile(shapes, fun) do
    builder = Exla.Builder.new("test")

    {params, _} =
      Enum.map_reduce(shapes, 0, fn shape, pos ->
        {Exla.Op.parameter(builder, pos, shape, <<?a + pos>>), pos + 1}
      end)

    op = apply(fun, [builder | params])
    comp = Exla.Builder.build(op)
    Exla.Client.compile(client(), comp, shapes)
  end

  @doc """
  Compiles and runs the given function.

  It expects a list of buffers which will be have their shapes
  used for compilation and then given on execution.
  """
  def run(args, opts \\ [], fun) do
    exec = compile(Enum.map(args, & &1.shape), fun)
    Exla.Executable.run(exec, args, opts)
  end
end

defmodule Nx.ProcessDevice do
  @behaviour Nx.Device

  def allocate(data, _type, _shape, opts) do
    key = Keyword.fetch!(opts, :key)
    Process.put(key, data)
    {__MODULE__, key}
  end

  def read(key), do: Process.get(key) || raise "deallocated"

  def deallocate(key), do: if(Process.delete(key), do: :ok, else: :already_deallocated)
end