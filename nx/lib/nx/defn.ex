defmodule Nx.Defn do
  @moduledoc ~S"""
  Numerical functions.

  A numerical function is a subset of Elixir tailored for
  numerical computations. For example, the following function:

      defn add_and_mult(a, b, c) do
        a * b + c
      end

  will work with scalars, vector, matrices, and n-dimensional
  tensors. Depending on your compiler of choice, the code can even
  be JIT-compiled or AOT-compiled and run either on the CPU or GPU.

  To support these features, `defn` is a subset of Elixir. It
  replaces Elixir's `Kernel` by `Nx.Defn.Kernel`. `Nx.Defn.Kernel`
  provides tensor-aware operators, such as `+`, `-`, etc, while
  also preserving many high-level constructs known to Elixir
  developers, such as pipe operator, aliases, conditionals,
  pattern-matching, the access syntax, and more:

  For example, the code above can also be written as:

      defn add_and_mult(a, b, c) do
        a
        |> Nx.multiply(b)
        |> Nx.add(c)
      end

  Please consult `Nx.Defn.Kernel` for a complete reference.

  ## Operators

  `defn` attempts to keep as close to the Elixir semantics as
  possible but that's not achievable. For example, mathematical
  and bitwise operators (`+`, `-`, `&&&`, `<<<`, etc.) in Elixir
  work on numbers, which means mapping them to tensors is
  straight-forward and they largely preserve the same semantics,
  except they are now multi-dimensional.

  On the other hand, the logical operators `and`, `or`, and `not`
  work with booleans in Elixir (`true` and `false`), which map
  to `0` and `1` in `defn`.

  Therefore, when working with logical operators inside `defn`,
  `0` is considered `false` and all other numbers are considered
  `true`, which is represented as the number `1`. For example, in
  `defn`, `0 and 1` as well as `0 and 2` return `0`, while
  `1 and 1` or `1 and -1` will return `1`.

  The same semantics apply to conditional expressions inside `defn`.

  ## JIT compilers

  The power of `Nx.Defn` is given by its compilers. The default
  compiler is the `Nx.Defn` module itself, which executes the code
  in pure Elixir. However, you can use module attributes to specify
  how a `defn` function will behave. For example, assuming you
  are using the `EXLA` compiler:

      @defn_compiler {EXLA, client: :host}
      defn add_and_mult(a, b, c) do
        a * b + c
      end

  To set the compiler for the all definitions, you can set the
  `@default_defn_compiler` attribute:

      @default_defn_compiler {EXLA, client: :cuda}

  `defn` functions are compiled when they are invoked, based on
  the type and shapes of the tensors given as arguments. Once
  invoked for the first time, the compilation is cached based
  on the tensors shapes and types. Calling the same function with
  a tensor of different values but same shape and type means no
  further compilation is performed.

  Also note that the defn compiler only applies to the first
  call to `defn`. All other calls that happen within that `defn`
  will use the same compiler. For example, imagine this code:

      @defn_compiler Nx.Defn.Evaluator # the default
      defn add(a, b), do: do_add(a, b)

      @defn_compiler EXLA
      defnp do_add(a, b), do: a + b

  When calling `add/2` directly, even though it calls `do_add/2`
  which uses EXLA, the call to `add/2` will be compiled with
  `Nx.Defn` and `Nx.Defn` exclusively. In other words, only the
  entry-point compiler matters.

  For those interested in writing custom compilers, see `Nx.Defn.Compiler`.

  ## Inputs and outputs types

  The arguments to `defn` functions must be either tensors, numbers,
  or anonymous functions. It may also be a map with numbers/tensors as
  values, a tuple of numbers/tensors or a tuple of functions.

  To pass any other values to numerical definitions, they must be
  declared as default arguments (see next subsection).

  `defn` functions can only return tensors or tuples of tensors.

  ### Default arguments

  `defn` functions support default arguments. They are typically used
  as options. For example, imagine you want to create a function named
  zeros, which returns a tensor of zeroes with a given type and shape.
  It could be implemented like this:

      defn zeros(opts \\ []) do
        opts = keyword!(opts, type: {:f, 32}, shape: {})
        Nx.broadcast(Nx.tensor(0, type: opts[:type]), opts[:shape])
      end

  The function above accepts `opts` which are then validated and given
  default values via the `keyword!/2` function. Note that while it is
  possible to access options via the `Access` syntax, such as `opts[:shape]`,
  it is not possible to directly call functions in the `Keyword` module
  inside `defn`. To freely manipulate any Elixir value inside `defn`,
  you have to use transforms, as described in the "Invoking custom Elixir
  code" section.

  When it comes to JIT compilation, it is important to notice that each
  different set of options will lead to a different compilation of the
  numerical function. Also note that, if tensors are given as default
  arguments, the whole tensor will be used as the compilation key. So
  even if you pass different tensors with the same type and shape, it
  will lead to different compilation artifacts. For this reason, it
  is **extremely discouraged to pass tensors through default arguments**.

  ### Working with maps

  While `Nx` works supports maps in `defn`, you must be careful
  if your numerical definitions are receiving maps and returning
  maps. For example, imagine this code:

      defn update_a(map) do
        %{map | a: Nx.add(map.a, 1)}
      end

  The following code increments the value under the key `:a` by
  1. However, because the function receives the whole map and
  returns the whole map, it means if the map has 120 keys, the
  whole map will be copied to the CPU/GPU, and then brought back.

  However, if you do this instead:

      defn update_a(map) do
        Nx.add(map.a, 1)
      end

  And then update the map on Elixir:

      %{map | a: update_a(map)}

  `Nx` will only send the parts of the map that matters.

  ## Invoking custom Elixir code

  Inside `defn` you can only call other `defn` functions and
  the functions in the `Nx` module. However, it is possible
  to use transforms to invoke any Elixir code:

      defn add_and_mult(a, b, c) do
        res = a * b + c
        transform(res, &IO.inspect/1)
      end

  For example, the code above invokes `&IO.inspect/1`, which is
  not a `defn` function, with the value of `res`. This is useful
  as it allows developers to transform `defn` code at runtime,
  in order to optimize, add new properties, and so on.

  Transforms can also be used to manipulate Elixir data structures,
  such as options. For example, imagine you want to support options
  where the :axis key is required. While you can't invoke `Keyword`
  directly, you can do it via a transform:

      defn sum_axis(t, opts \\ []) do
        opts = keyword!(opts, [:axis])
        axis = transform(opts, &Keyword.fetch!(opts, :axis))
        Nx.sum(t, axes: [axis])
      end

  """

  @doc """
  Invokes the anonymous function with just-in-time compilation.

  The anonymous function will be invoked with tensor expressions
  which are JIT compiled and then invoked. For example, take the
  following definition:

      defn softmax(t), do: Nx.exp(t) / Nx.sum(Nx.exp(t))

  **Note:** `jit/3` will ignore the `@defn_compiler` on the executed
  function. Be sure to pass the `compiler` and its `opts` as keywords
  instead:

      Nx.Defn.jit(&Mod.softmax/1, [my_tensor], compiler: EXLA)
      Nx.Defn.jit(&Mod.softmax/1, [my_tensor], compiler: EXLA, run_options: [keep_on_device: true])

  """
  def jit(fun, args, opts \\ [])
      when is_function(fun) and is_list(args) and is_list(opts) do
    Nx.Defn.Compiler.__jit__(fun, args, opts)
  end

  @doc """
  JITs the given function if outside of `defn`, otherwise invokes it.

  It is not possible to invoke `jit/3` inside `defn`, as all code inside
  `defn` is already either jitted or aot-compiled. However, some libraries
  may want to provide abstractions that can be invoked either inside `defn`
  or outside. In such cases, `jit_or_apply/3` can be used to start jitting
  if it has been invoked outside of a numerical definition.

  The `opts` are the same as the ones given to `jit/3` and they are only
  used if invoking this function outside of `defn`.
  """
  def jit_or_apply(fun, args, opts \\ [])
      when is_function(fun) and is_list(args) and is_list(opts) do
    if Nx.Defn.Compiler.current() do
      apply(fun, args)
    else
      jit(fun, args, opts)
    end
  end

  @doc """
  Starts streaming the given anonymous function with just-in-time
  compilation.

  At least two arguments are expected:

    1. The first argument is a tensor template of the data to
       be streamed in

    2. The second argument is a tensor with the stream initial state

  The streaming function must return a two element tuple, the
  first element is the data to be sent and the second is the
  accumulator.

  For each streamed chunk, you must call `Nx.Stream.send/2` and
  `Nx.Stream.recv/1`. You don't need to call `recv` immediately
  after `send`, but doing so can be a useful mechanism to provide
  backpressure. Once all chunks are sent, you must use `Nx.Stream.done/1`
  to receive the accumulated result. Let's see an example:

      defmodule Streamed do
        import Nx.Defn

        defn sum(tensor, acc) do
          {acc, tensor + acc}
        end
      end

  Now let's invoke it:

      stream = Nx.Defn.stream(&Streamed.sum/2, [Nx.template({}, {:s, 64}), 0])

      for i <- 1..5 do
        Nx.Stream.send(stream, i)
        IO.inspect {:chunk, Nx.Stream.recv(stream)}
      end

      IO.inspect {:result, Nx.Stream.done(stream)}

  It will print:

      {:chunk, 0}
      {:chunk, 1}
      {:chunk, 2}
      {:chunk, 3}
      {:chunk, 4}
      {:result, 5}

  **Note:** similar to `jit/3`, `stream/3` will ignore the `@defn_compiler`
  on the executed function. Be sure to pass the `compiler` and its `opts`
  as arguments instead.
  """
  def stream(fun, args, opts \\ [])
      when is_function(fun) and is_list(args) and is_list(opts) do
    case args do
      [input, acc | args] ->
        acc = Nx.Defn.Tree.composite(acc, &Nx.to_tensor/1)
        Nx.Defn.Compiler.__stream__(fun, Nx.to_template(input), acc, args, opts)

      _ ->
        raise ArgumentError, "Nx.Defn.stream/3 expects at least two arguments"
    end
  end

  @doc """
  Exports the ahead-of-time (AOT) definition of a module with the
  given `functions` using the given `compiler`. For example:

      functions = [
        {:softmax, &Nx.exp(&1)/Nx.sum(Nx.exp(&1)), [Nx.template({100, 100}, {:f, 32})]},
        {:normalize, &Nx.divide(&1, 255), [Nx.template({100, 100}, {:f, 32})]}
      ]

      :ok = Nx.Defn.export_aot("priv", MyModule, functions, compiler: MyCompiler)

  The above will export a module definition called `MyModule`
  to the given directory with `softmax/1` and `normalize/1` as
  functions that expects f32 tensors that are 100x100 of shape.
  This definition can then be imported at will.

  `functions` is a list of 3- or 4-element tuples. The first element
  is the function name, the second is an anonymous function that returns
  the tensor expression for a given function, the third is a list of
  the arguments as templates, and the fourth is an option list of
  options for the given tensor expression (often similar to the options
  you would pass on the call to `jit/2`).

  The options to each function as long as the `aot_options` are specific
  to the given compiler.

  ## AOT export with Mix

  Ahead-of-time exports with Mix are useful because you only need
  the compilation environment when exporting.
  In practice, you can do this:

    1. Add `{:my_compiler, ..., only: :export_aot}` as a dependency

    2. Define an exporting script at `script/export_my_module.exs`

    3. Run the script to export the AOT `mix run script/export_my_module.exs`

    4. Now inside `lib/my_module.ex` you can import it:

          if File.exists?("priv/MyModule.nx.aot") do
            defmodule MyModule do
              Nx.Defn.import_aot("priv", __MODULE__)
            end
          else
            IO.warn "Skipping MyModule because aot definition was not found"
          end

  """
  def export_aot(dir, module, functions, aot_opts \\ [])
      when is_binary(dir) and is_atom(module) and is_list(functions) and is_list(aot_opts) do
    functions =
      for tuple <- functions do
        case tuple do
          {name, fun, args} ->
            {name, fun, args, []}

          {name, fun, args, opts} ->
            {name, fun, args, opts}

          _ ->
            raise ArgumentError,
                  "expected 3- or 4-element tuples as functions, got: #{inspect(tuple)}"
        end
      end

    Nx.Defn.Compiler.__export_aot__(dir, module, functions, aot_opts)
  end

  @doc """
  Imports a previousy exported AOT definition for `module` at `dir`.

  See `export_aot/4` for more information.
  """
  def import_aot(dir, module) when is_binary(dir) and is_atom(module) do
    unless Module.open?(module) do
      raise ArgumentError,
            """
            cannot import_aot/2 for #{inspect(module)} because module was already defined.
            You should call import_aot/2 while the module is being defined:

                defmodule MyModule do
                  Nx.Defn.import_aot("priv", MyModule)
                end
            """
    end

    Nx.Defn.Compiler.__import_aot__(dir, module, true)
  end

  @doc """
  Defines a `module` by compiling an ahead-of-time (AOT) definition
  with the given `functions` using the given `compiler`.
  For example:

      functions = [
        {:softmax, &Nx.exp(&1)/Nx.sum(Nx.exp(&1)), [Nx.template({100, 100}, {:f, 32})]},
        {:normalize, &Nx.divide(&1, 255), [Nx.template({100, 100}, {:f, 32})]}
      ]

      Nx.Defn.aot(MyModule, functions, EXLA)

  The above will define a module called `MyModule` with
  `softmax/1` and `normalize/1` as functions that expects
  f32 tensors that are 100x100 of shape.

  While this function defines the module immediately, in
  practice most developers will use `export_aot` to export
  the AOT definition and then use `import_aot` to import it.
  This is useful because you only need the compilation
  environment, such as EXLA, only when exporting. In practice,
  this function is equivalent to the following:

      :ok = Nx.Defn.export_aot("priv/my_app/aot", MyModule, functions, compiler: EXLA)

      defmodule MyModule do
        Nx.Defn.import_aot("priv/my_app/aot", __MODULE__)
      end

  See `export_aot/5` for more information.
  """
  def aot(module, functions, aot_opts \\ [])
      when is_atom(module) and is_list(functions) and is_list(aot_opts) do
    output_dir = Path.join(System.tmp_dir(), "elixir-nx/aot#{System.unique_integer()}")

    try do
      case export_aot(output_dir, module, functions, aot_opts) do
        :ok ->
          defmodule module do
            @moduledoc false
            Nx.Defn.Compiler.__import_aot__(output_dir, module, false)
            :ok
          end

        {:error, exception} ->
          raise exception
      end
    after
      File.rm_rf!(output_dir)
    end
  end

  @doc """
  Receives an anonymous function and returns a new anonymous function
  that returns the gradient of the input function when invoked.

  This function is typically used to compute the gradient outside of
  `defn`. To compute them inside `defn`, prefer `grad/2` instead.

  ## Examples

      iex> fun = Nx.Defn.grad(fn x -> Nx.sin(x) end)
      iex> fun.(Nx.tensor(0))
      #Nx.Tensor<
        f32
        1.0
      >

  """
  def grad(fun) when is_function(fun, 1) do
    fn t -> jit_or_apply(fn t -> grad(t, fun) end, [t]) end
  end

  @doc """
  Computes the gradient of the given `var` on `fun`.

  This is typically used inside `defn`. The result of the `grad`
  function must be a scalar tensor. If a non-scalar tensor is given,
  it is assumed the additional dimensions are batch dimensions.

  ### Examples

      defn tanh_grad(t) do
        grad(t, &Nx.tanh/&1)
      end

  To differentiate on multiple vars, pass a tuple as first argument:

      defn tanh_power_grad(a, b) do
        grad({a, b}, fn {a, b} -> Nx.tanh(a) + Nx.power(b, 2) end)
      end

  When a tuple is given, a tuple will be returned.
  """
  def grad(var_or_vars, fun) when is_function(fun, 1) do
    {_value, grad} = Nx.Defn.Grad.transform(var_or_vars, fun, & &1)
    grad
  end

  @doc """
  Receives an anonymous function and returns a new anonymous function
  that returns the value and gradient of the input function when invoked.

  This function is typically used to compute the value and gradient
  outside of `defn`. To compute them inside `defn`, prefer
  `value_and_grad/2` instead.

  ## Examples

      iex> fun = Nx.Defn.value_and_grad(fn x -> Nx.sin(x) end)
      iex> {value, grad} = fun.(Nx.tensor(0))
      iex> value
      #Nx.Tensor<
        f32
        0.0
      >
      iex> grad
      #Nx.Tensor<
        f32
        1.0
      >

  """
  def value_and_grad(fun) when is_function(fun, 1) do
    fn t -> jit_or_apply(fn t -> value_and_grad(t, fun) end, [t]) end
  end

  @doc """
  Computes the value and gradient of the given `var` on `fun`
  with an optional data transformation.

  This is typically used inside `defn`. It returns a tuple with
  the value and the gradient.

  ### Examples

      defn tanh_grad(t) do
        value_and_grad(t, &Nx.tanh/&1)
      end

  To differentiate on multiple vars, pass a tuple as first argument:

      defn tanh_power_grad(a, b) do
        value_and_grad({a, b}, fn {a, b} -> Nx.tanh(a) + Nx.power(b, 2) end)
      end

  When a tuple is given, a tuple will be returned.

  `transform` allows you to transform the expression before the gradient is
  calculated. This enables optimizations that reuse parts of expressions. As
  an example, consider the following objective function:

      defn objective(predict_fn, loss_fn, params, inputs, targets) do
        preds = predict_fn.(params, inputs)
        loss = loss_fn.(preds, targets)
        {preds, loss}
      end

  You can compute the gradient with respect to just the loss function by applying
  a transform:

      {{preds, loss}, gradient} = value_and_grad(params, &objective(predict_fn, loss_fn, &1, inputs, targets), &elem(&1, 1))

  `preds` can be re-used to compute other metrics such as accuracy, absolute error,
  etc. without having to do another forward pass.
  """
  def value_and_grad(var_or_vars, fun, transform \\ & &1)
      when Kernel.and(is_function(fun, 1), is_function(transform, 1)) do
    Nx.Defn.Grad.transform(var_or_vars, fun, transform)
  end

  @doc """
  Defines a public numerical function.
  """
  defmacro defn(call, do: block) do
    define(:def, call, block, __CALLER__)
  end

  @doc """
  Defines a private numerical function.

  Private numerical functions are always inlined by
  their callers at compilation time. This happens to
  all local function calls within `defn`.
  """
  defmacro defnp(call, do: block) do
    define(:defp, call, block, __CALLER__)
  end

  ## Callbacks

  defp define(kind, call, block, env) do
    assert_no_guards!(kind, call, env)
    # Note name here is not necessarily an atom due to unquote(name) support
    {name, args} = decompose_call!(kind, call, env)
    defaults = for {{:\\, _, [_, _]}, i} <- Enum.with_index(args), do: i
    arity = length(args)

    quote do
      unquote(__MODULE__).__define__(
        __MODULE__,
        unquote(kind),
        unquote(name),
        unquote(arity),
        unquote(defaults)
      )

      unquote(kind)(unquote(call)) do
        use Nx.Defn.Kernel
        unquote(block)
      end
    end
  end

  defp decompose_call!(_kind, {{:unquote, _, [name]}, _, args}, _env) do
    {name, args}
  end

  defp decompose_call!(kind, call, env) do
    case Macro.decompose_call(call) do
      {name, args} ->
        {name, args}

      :error ->
        compile_error!(
          env,
          "first argument of #{kind}n must be a call, got: #{Macro.to_string(call)}"
        )
    end
  end

  defp assert_no_guards!(kind, {:when, _, _}, env) do
    compile_error!(env, "guards are not supported by #{kind}n")
  end

  defp assert_no_guards!(_kind, _call, _env), do: :ok

  # Internal attributes
  @exports_key :__defn_exports__

  # Per-defn attributes
  @defn_compiler :defn_compiler

  # Module attributes
  @default_defn_compiler :default_defn_compiler

  @doc false
  def __define__(module, kind, name, arity, defaults) do
    exports =
      if exports = Module.get_attribute(module, @exports_key) do
        exports
      else
        Module.put_attribute(module, :before_compile, __MODULE__)
        %{}
      end

    compiler =
      Module.delete_attribute(module, @defn_compiler) ||
        Module.get_attribute(module, @default_defn_compiler) ||
        Nx.Defn.Evaluator

    exports =
      Map.put(exports, {name, arity}, %{
        kind: kind,
        compiler: normalize_compiler!(compiler),
        defaults: defaults
      })

    Module.put_attribute(module, @exports_key, exports)
    :ok
  end

  defp normalize_compiler!(atom) when is_atom(atom), do: {atom, []}
  defp normalize_compiler!({atom, term}) when is_atom(atom), do: {atom, term}

  defp normalize_compiler!(other) do
    raise ArgumentError,
          "expected @defn_compiler/@default_defn_compiler to be an atom or " <>
            "a tuple with an atom as first element, got: #{inspect(other)}"
  end

  defp compile_error!(env, description) do
    raise CompileError, line: env.line, file: env.file, description: description
  end

  @doc false
  defmacro __before_compile__(env) do
    exports = Module.get_attribute(env.module, @exports_key)
    Nx.Defn.Compiler.__compile__(env, exports)
  end
end
