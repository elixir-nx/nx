Mix.install([{:exla, path: "."}, {:nx, path: "../nx"}, {:pythonx, "~> 0.4"}])

Pythonx.uv_init("""
[project]
name = "project"
version = "0.0.0"
requires-python = "==3.13.*"
dependencies = [
  "numpy==2.2.2"
]
""")

defmodule MyModule do
  import Nx.Defn

  defn run(t) do
    out =
      Nx.elixir_call(
        %{t | type: Nx.Type.to_floating(t.type)},
        [t, [value: 10]],
        &elixir_fun/2
      )

    Nx.negate(out)
  end

  def elixir_fun(t, opts) do
    input = Nx.to_flat_list(t)

    {res, _ctx} =
      Pythonx.eval(
        """
        import numpy as np
        arr = np.array(input)

        print("python np.array:", arr)

        c = np.cos(arr) + offset

        list(c)
        """,
        %{"input" => input, "offset" => opts[:value]}
      )

    Nx.tensor(Pythonx.decode(res))
  end
end

Nx.Defn.default_options(compiler: Nx.Defn.Evaluator)
Nx.global_default_backend(EXLA.Backend)

t = Nx.iota({10})

dbg(t)

expr = Nx.Defn.debug_expr(&MyModule.run/1).(t)
dbg(expr)

result = MyModule.run(t)
dbg(result)
