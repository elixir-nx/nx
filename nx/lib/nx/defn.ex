defmodule Nx.Defn do
  @moduledoc ~S"""
  Numerical functions.

  A numerical function is a subset of Elixir tailored for
  numerical computations. For example, the following function:

      defmodule MyModule do
        import Nx.Defn

        defn softmax(t) do
          Nx.exp(t) / Nx.sum(Nx.exp(t))
        end
      end

  will work with scalars, vector, matrices, and n-dimensional
  tensors. Depending on your compiler of choice, the code can even
  be JIT-compiled and run either on the CPU or GPU.

  To support these features, `defn` is a subset of Elixir. It
  replaces Elixir's `Kernel` by `Nx.Defn.Kernel`. `Nx.Defn.Kernel`
  provides tensor-aware operators, such as `+`, `-`, etc, while
  also preserving many high-level constructs known to Elixir
  developers, such as pipe operator, aliases, conditionals,
  pattern-matching, the access syntax, and more:

  For example, the code above can also be written as:

      defmodule MyModule do
        import Nx.Defn

        defn softmax(t) do
          t
          |> Nx.exp(t)
          |> then(& &1 / Nx.sum(&1))
        end
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

  The same semantics apply to conditional expressions inside `defn`,
  such as `if`, `while`, etc.

  ## JIT compilers

  The power of `Nx.Defn` is given by its compilers. The default
  compiler is `Nx.Defn.Evaluator`, which evalutes the code.
  You can use `jit/3` to compile a function on the fly using a
  different compiler, such as `EXLA`:

      fun = Nx.Defn.jit(&MyModule.softmax/1, compiler: EXLA)
      fun.(my_tensor)

  The above will return an anonymous function that optimizes,
  compiles, and run `softmax` on the fly on the CPU (or the GPU)
  if available.

  You can also change the default compiler for all numerical
  definitions (`defn`) by setting the default options. This can
  be done in your `config/*.exs` files as follows:

      config :nx, :default_defn_options, compiler: EXLA

  Now calling `MyModule.softmax(my_tensor)` will use `EXLA` even
  without wrapping it in `jit/2`.

  However, note that compilation may be quite time consuming on
  the first invocation, that's why it is often preferred to use
  the `compiler: EXLA` option when calling the functions in this
  module instead. EXLA, in particular, also exports a `EXLA.jit/2`
  function for convenience.

  `defn` functions are compiled when they are invoked, based on
  the type and shapes of the tensors given as arguments. The
  compilation is then cached based on the tensors shapes and types.
  Calling the same function with a tensor of different values but
  same shape and type means no recompilation is performed.

  For those interested in writing custom compilers, see `Nx.Defn.Compiler`.

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
  as it allows developers to transform `defn` code to optimize,
  add new properties, and so on.

  Transforms can also be used to manipulate Elixir data structures,
  such as options. `defn` expects all inputs to be tensors, with the
  exception of a default argument (declared with `\\`) which will be
  treated as options.

  For example, imagine you want to support options where the :axis
  key is required. While you can't invoke `Keyword` directly, you
  can do it via a transform:

      defn sum_axis(t, opts \\ []) do
        opts = keyword!(opts, [:axis])
        axis = transform(opts, &Keyword.fetch!(opts, :axis))
        Nx.sum(t, axes: [axis])
      end

  ## Inputs and outputs types

  `Nx` and `defn` expect the arguments to be numbers, tensors,
  or one of the following composite data types:

    1. tuples of numbers/tensors
    2. maps of any key with numbers/tensors as values
    3. any struct that implements `Nx.Container`

  When numbers are given as arguments, they are always immediately
  converted to tensors on invocation. If you want to keep numbers
  as is or if you want to pass any other value to numerical definitions,
  they must be given as default arguments (see next subsection).

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

  Additionally, `defn` supports anonymous as a direct input, without wrapping
  in a default argument.

  > **Important!** When it comes to JIT compilation, each different set of
  > options and anonymous functions will lead to a different compilation of
  > the numerical function.
  >
  > Furthermore, if tensors are given through default arguments, they won't
  > be cached effectively. Tensors in `defn` are cached based on their shape
  > and type, not their value, but this is not true if the tensor is given
  > via a default argument or captured by an anonymous function. For this
  > reason, it is **extremely discouraged to pass tensors through anonymous
  > functions and default arguments**.

  ### Working with maps and structs

  While `Nx` supports maps in `defn`, you must be careful if your numerical
  definitions are receiving maps and returning maps. For example, imagine
  this code:

      defn update_a(map) do
        %{map | a: Nx.add(map.a, 1)}
      end

  The following code increments the value under the key `:a`
  by 1. However, because the function receives the whole map and
  returns the whole map, it means if the map has 120 keys, the
  whole map will be copied to the CPU/GPU, and then brought back.

  However, if you do this instead:

      defn update_a(map) do
        Nx.add(map.a, 1)
      end

  And then update the map on Elixir, outside of `defn`:

      %{map | a: update_a(map)}

  `Nx` will only send the parts of the map that matters.
  """

  @compiler_key {Nx.Defn, :default_compiler}
  @app_key :default_defn_options

  @doc """
  Sets the default options for `defn` in the current process.

  The options defined here apply to all future invocations of
  `defn` done by the current process. It also applies to calls
  to the `jit/3` and `stream/3` functions in this module.

  The default options are stored only in the process dictionary
  and override any global options. This means if you start a
  separate process, such as `Task`, the default options must be
  set on the new process too.

  This function is mostly used for scripting and testing. In your
  applications, you typically set the default options in your
  config files:

        config :nx, :#{@app_key}, [compiler: EXLA, client: :cuda]

  """
  def default_options(options) when is_list(options) do
    Process.put(@compiler_key, options) || Application.fetch_env!(:nx, @app_key)
  end

  @doc """
  Sets the default options globally.

  The options defined here apply to all future invocations of
  `defn`. It also applies to calls to the `jit/3` and `stream/3`
  functions in this module.

  You must avoid calling this function at runtime. It is mostly
  useful during scripts or code notebooks to set a default.
  If you need to configure a global default options in your
  applications, you can do so in your `config/*.exs` files:

      config :nx, :#{@app_key}, [compiler: EXLA, client: :cuda]

  """
  def global_default_options(options) when is_list(options) do
    current = Application.fetch_env!(:nx, @app_key)
    Application.put_env(:nx, @app_key, options)
    current
  end

  @doc """
  Gets the default options for the current process.
  """
  def default_options() do
    Process.get(@compiler_key) || Application.fetch_env!(:nx, @app_key)
  end

  @doc """
  Compiles the given anonymous function with the given tensor shapes.

  While `jit/2` compiles a function just-in time based on the
  input shapes, this function precompiles the given anonymous
  function based on the input shapes. This can be beneficial for
  large numerical definitions, where the cache mechanism in `jit/2`
  may take miliseconds.

  For example, take the following definition:

      defn softmax(t), do: Nx.exp(t) / Nx.sum(Nx.exp(t))

  You can jit and then apply it as:

      fun = Nx.Defn.compile(&softmax/1, [Nx.template({3}, {:s, 64})], compiler: EXLA)
      fun.(Nx.tensor([1, 2, 3]))

  If the input tensors do not match the shape of the tensors
  given on compilation, it will raise.

  ## Options

    * `:compiler` - the compiler for the JIT compilation

    * `:hooks` - a map of hooks to execute. See `Nx.Defn.Kernel.hook/3`

  """
  def compile(fun, template_args, opts \\ [])
      when is_function(fun) and is_list(template_args) and is_list(opts) do
    template_args = Enum.map(template_args, &Nx.to_template/1)
    opts = prepare_options(opts)
    compiled_fun = Nx.Defn.Compiler.__compile__(fun, template_args, opts)

    wrap(fun, fn args ->
      if Nx.Defn.Compiler.current() do
        raise "cannot invoke compiled function when there is a JIT compilation happening"
      end

      assert_compatible!(args, template_args, 1)
      flatten = Nx.Defn.Composite.flatten_runtime_args(args, [])
      [res] = compiled_fun.([flatten])
      res
    end)
  end

  defp assert_compatible!([arg | args], [template | templates], pos) do
    if Nx.compatible?(arg, template) do
      assert_compatible!(args, templates, pos + 1)
    else
      raise ArgumentError, """
      argument at position #{pos} is not compatible with compiled function template.

      Template:

      #{inspect(template)}

      Argument:

      #{inspect(arg)}

      """
    end
  end

  defp assert_compatible!([], [], _pos), do: :ok

  @doc """
  Wraps an anonymous function with just-in-time compilation.

  Once invoked, the wrapped anonymous function with perform just
  in time compilation with the configured compiler. For example,
  take the following definition:

      defn softmax(t), do: Nx.exp(t) / Nx.sum(Nx.exp(t))

  You can jit and then apply it as:

      fun = Nx.Defn.jit(&softmax/1, compiler: EXLA)
      fun.(Nx.tensor([1, 2, 3]))

  ## Options

    * `:compiler` - the compiler for the JIT compilation

    * `:hooks` - a map of hooks to execute. See `Nx.Defn.Kernel.hook/3`

    * `:on_conflict` - what to do if a JIT compilation is already in place.
      It may be `:raise` (the default), `:force` (forces a new JIT compilation),
      or `:reuse` (reuses the exiting JIT compilation). It is not recommended
      to set the `:compiler` option when reusing.

  """
  def jit(fun, opts \\ []) when is_function(fun) and is_list(opts) do
    if Keyword.keyword?(opts) do
      wrap(fun, &jit_apply(fun, &1, opts))
    else
      IO.warn("jit/3 is deprecated, use jit/2 instead")
      jit_apply(fun, opts, [])
    end
  end

  @deprecated "Use jit/2 instead"
  def jit(fun, args, opts) when is_function(fun) and is_list(args) and is_list(opts) do
    jit_apply(fun, args, opts)
  end

  @doc """
  Invokes the anonymous function with just-in-time compilation.

  This function is equivalent to calling `jit/2` and then applying
  the given arguments to the anonymous function.

  For example, take the following definition:

      defn softmax(t), do: Nx.exp(t) / Nx.sum(Nx.exp(t))

  You can `jit_apply/3` it as:

      Nx.Defn.jit_apply(&softmax/1, [Nx.tensor([1, 2, 3])], compiler: EXLA)

  It accepts the same options as `jit/2`.
  """
  def jit_apply(fun, args, opts \\ [])
      when is_function(fun) and is_list(args) and is_list(opts) do
    {on_conflict, opts} = Keyword.pop(opts, :on_conflict, :raise)

    cond do
      Nx.Defn.Compiler.current() == nil ->
        do_jit_apply(fun, args, opts)

      on_conflict == :raise ->
        raise "cannot invoke JITed function when there is a JIT compilation happening"

      on_conflict == :force ->
        do_jit_apply(fun, args, opts)

      on_conflict == :reuse ->
        apply(fun, args)
    end
  end

  defp do_jit_apply(fun, args, opts) do
    opts = prepare_options(opts)
    flatten = Nx.Defn.Composite.flatten_runtime_args(args, [])
    [res] = Nx.Defn.Compiler.__jit__(fun, args, [flatten], opts)
    res
  end

  @deprecated "Use jit/2 or jit_apply/3 with the :on_conflict option"
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

  ## Options

    * `:hooks` - a map of hooks to execute. See `Nx.Defn.Kernel.hook/3`

  """
  def stream(fun, args, opts \\ [])
      when is_function(fun) and is_list(args) and is_list(opts) do
    if Nx.Defn.Compiler.current() do
      raise "cannot call Nx.Defn.stream/3 when there is a JIT compilation happening"
    end

    case args do
      [input, acc | _] ->
        acc = Nx.Defn.Composite.traverse(acc, &Nx.to_tensor/1)
        opts = prepare_options(opts)
        flatten = Nx.Defn.Composite.flatten_runtime_args(args, [])

        [stream] =
          Nx.Defn.Compiler.__stream__(fun, Nx.to_template(input), acc, args, [flatten], opts)

        stream

      _ ->
        raise ArgumentError, "Nx.Defn.stream/3 expects at least two arguments"
    end
  end

  defp prepare_options(opts) do
    opts = Keyword.merge(default_options(), opts)

    if not is_map(Keyword.get(opts, :hooks, %{})) do
      raise ArgumentError, ":hooks option must be a map"
    end

    opts
  end

  defp wrap(fun, callback) do
    {:arity, arity} = Function.info(fun, :arity)
    wrap_arity(arity, callback)
  end

  for i <- 0..128 do
    args = Macro.generate_arguments(i, __MODULE__)

    defp wrap_arity(unquote(i), callback) do
      fn unquote_splicing(args) -> callback.(unquote(args)) end
    end
  end

  @doc """
  Receives an anonymous function and returns a new anonymous function
  that returns the gradient of the input function when invoked.

  ## Examples

      iex> fun = Nx.Defn.grad(fn x -> Nx.sin(x) end)
      iex> fun.(Nx.tensor(0))
      #Nx.Tensor<
        f32
        1.0
      >

  """
  def grad(fun) when is_function(fun, 1) do
    fn t -> grad(t, fun) end
  end

  @doc """
  Computes the gradient of the given `var` on `fun`.

  The result of the `grad` function must be a scalar tensor.
  If a non-scalar tensor is given, it is assumed the additional
  dimensions are batch dimensions.

  ### Examples

      defn tanh_grad(t) do
        grad(t, &Nx.tanh/&1)
      end

  To differentiate on multiple vars, pass a tuple as first argument:

      defn tanh_power_grad(a, b) do
        grad({a, b}, fn {a, b} -> Nx.tanh(a) + Nx.power(b, 2) end)
      end

  `var_or_vars` can be any `Nx.Container` with one or multiple
  tensors.
  """
  def grad(var_or_vars, fun) when is_function(fun, 1) do
    jit_apply(
      fn var_or_vars ->
        {_value, grad} = Nx.Defn.Grad.transform(var_or_vars, fun, & &1)
        grad
      end,
      [var_or_vars],
      on_conflict: :reuse
    )
  end

  @doc """
  Receives an anonymous function and returns a new anonymous function
  that returns the value and gradient of the input function when invoked.

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
    fn t -> value_and_grad(t, fun) end
  end

  @doc """
  Computes the value and gradient of the given `var` on `fun`
  with an optional data transformation.

  It returns a tuple with the value and the gradient.

  ### Examples

      defn tanh_grad(t) do
        value_and_grad(t, &Nx.tanh/&1)
      end

  To differentiate on multiple vars, pass a tuple as first argument:

      defn tanh_power_grad(a, b) do
        value_and_grad({a, b}, fn {a, b} -> Nx.tanh(a) + Nx.power(b, 2) end)
      end

  `var_or_vars` can be any `Nx.Container` with one or multiple
  tensors.

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
    jit_apply(
      fn var_or_vars -> Nx.Defn.Grad.transform(var_or_vars, fun, transform) end,
      [var_or_vars],
      on_conflict: :reuse
    )
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
    arity = length(args)

    defaults =
      for {{:\\, meta, [_, default]}, i} <- Enum.with_index(args),
          do: {i, {meta, default}},
          into: []

    quote do
      unquote(__MODULE__).__define__(
        __MODULE__,
        unquote(kind),
        unquote(name),
        unquote(arity),
        %{unquote_splicing(defaults)}
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

  @doc false
  def __define__(module, kind, name, arity, defaults) do
    exports =
      if exports = Module.get_attribute(module, @exports_key) do
        exports
      else
        Module.put_attribute(module, :before_compile, __MODULE__)
        %{}
      end

    exports = Map.put(exports, {name, arity}, %{kind: kind, defaults: defaults})
    Module.put_attribute(module, @exports_key, exports)
    :ok
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
