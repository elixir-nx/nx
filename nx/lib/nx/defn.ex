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
          |> Nx.exp()
          |> then(& &1 / Nx.sum(&1))
        end
      end

  Please consult `Nx.Defn.Kernel` for a complete reference.

  Some of the functions in this module may also be used within
  `defn`.

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
  if available. EXLA, in particular, also exports a `EXLA.jit/2`
  function for convenience.

  `defn` functions are compiled when they are invoked, based on
  the type and shapes of the tensors given as arguments.
  Therefore compilation may be quite time consuming on the first
  invocation. The compilation is then cached based on the tensors
  shapes and types. Calling the same function with a tensor of
  different values but same shape and type means no recompilation
  is performed.

  For those interested in writing custom compilers, see `Nx.Defn.Compiler`.

  ## Invoking custom Elixir code

  Inside `defn` you can only call other `defn` functions and
  the functions in the `Nx` module. However, it is possible
  to use transforms, defined with either `deftransform` or
  `deftransformp` to invoke any Elixir code.

  You can call code which was defined with `deftransform` from another module:

      defmodule MyRemoteModule do
        import Nx.Defn

        deftransform remote_elixir_code(value) do
          IO.inspect(value)
        end
      end

      defn add_and_mult(a, b, c) do
        res = a * b + c
        MyRemoteModule.remote_elixir_code(res)
      end

  You can also define and call a private transform defined through `deftransformp`:

      defn add_and_mult(a, b, c) do
        res = a * b + c
        custom_elixir_code(res)
      end

      deftransformp custom_elixir_code(value), do: IO.inspect(value)

  The only difference between using `deftransform` and `deftransformp`
  is whether you want to expose and share the code with other modules,
  just like `def` and `defp`.

  Transforms are useful to manipulate tensor expressions or
  Elixir data structures without the constraints of `defn`.

  ## Inputs and outputs types

  `Nx` and `defn` expect the arguments to be numbers, tensors,
  or one composite data type that implements `Nx.LazyContainer`.
  Tuples and maps implement `Nx.LazyContainer` by default.
  As previously described, `defn` are cached based on the shape,
  type, and names of the input tensors, but not their values.

  `defn` also accepts two special arguments: functions (or tuples
  of functions) and lists (most commonly as keyword lists). Those
  values are passed as is to numerical definitions and cached as
  a whole. For this reason, you must never capture tensors in
  functions or pass tensors in keyword lists.

  When numbers are given as arguments, they are always immediately
  converted to tensors on invocation. If you want to keep numbers
  as is or if you want to pass any other value to numerical definitions,
  they must be given as keyword lists.

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

  > **Important!** When it comes to JIT compilation, each different set of
  > options (as well as anonymous functions) will lead to a different
  > compilation of the numerical function.
  >
  > Furthermore, if tensors are given through keyword lists, they won't
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

  ## Recursion and loops

  Given numerical definition first build a representation of
  your code, it is not possible to write recursive (nor tail
  recursive) code inside `defn`. Instead, one must use
  `Nx.Defn.Kernel.while/4`.
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

  The function returns the values that were previously set as default
  options.

  This function must be used only for scripting and testing.

  ## Examples

      iex> Nx.Defn.default_options(compiler: EXLA, client: :cuda)
      iex> Nx.Defn.default_options()
      [compiler: EXLA, client: :cuda]
  """
  def default_options(options) when is_list(options) do
    Process.put(@compiler_key, options) || Application.fetch_env!(:nx, @app_key)
  end

  @doc """
  Sets the default options globally.

  The options defined here apply to all future invocations of
  `defn`. It also applies to calls to the `jit/3` and `stream/3`
  functions in this module.

  You must avoid calling this function at runtime and mostly for
  testing purposes. You may also set in your test environment using
  configuration:

      config :nx, :#{@app_key}, [compiler: EXLA, client: :cuda]

  The function returns the values that were previously set as global
  default options.
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
  Returns a backend corresponding to the compiler options.

  The backend matches the backend used for outputs from computations
  defined by the given compiler.
  """
  def to_backend(opts) do
    opts = prepare_options(opts)
    Nx.Defn.Compiler.__to_backend__(opts)
  end

  @doc """
  Compiles the given anonymous function with the given tensor shapes.

  While `jit/2` compiles a function just-in time based on the
  input shapes, this function precompiles the given anonymous
  function based on the input shapes. This can be beneficial for
  large numerical definitions, where the cache mechanism in `jit/2`
  may take milliseconds.

  For example, take the following definition:

      defn softmax(t), do: Nx.exp(t) / Nx.sum(Nx.exp(t))

  You can jit and then apply it as:

      fun = Nx.Defn.compile(&softmax/1, [Nx.template({3}, {:s, 32})], compiler: EXLA)
      fun.(Nx.tensor([1, 2, 3]))

  You can also pass a mixture of templates and options when
  compiling a function. In such cases, you must only pass
  the inputs when invoking the compiled function, as the options
  will already be embedded in its compiled value:

      fun = Nx.Defn.compile(&Nx.sum/2, [Nx.template({2, 2}, {:s, 32}), [axes: [1]]])
      fun.(Nx.iota({2, 2}))

  If the input tensors do not match the shape of the tensors
  given on compilation, it will raise.

  ## Options

    * `:compiler` - the compiler for the JIT compilation

    * `:hooks` - a map of hooks to execute. See `Nx.Defn.Kernel.hook/3`

  """
  def compile(fun, template_args, opts \\ [])
      when is_function(fun) and is_list(template_args) and is_list(opts) do
    {fun, params, templates, _flatten} = Nx.Defn.Compiler.to_lazy_params(fun, template_args)
    opts = prepare_options(opts)
    compiled_fun = Nx.Defn.Compiler.__compile__(fun, params, opts)

    wrap(fun, fn args ->
      if Nx.Defn.Compiler.current() do
        raise "cannot invoke compiled function when there is a JIT compilation happening"
      end

      flatten = compile_flatten(args, templates, template_args, 1, [])
      [res] = compiled_fun.([flatten])
      res
    end)
  end

  defp compile_flatten([arg | args], templates, template_args, pos, acc) do
    {_, {templates, acc}} =
      Nx.LazyContainer.traverse(arg, {templates, acc}, fn
        arg_template, fun, {[template | templates], acc} ->
          unless Nx.compatible?(arg_template, template) do
            raise ArgumentError, """
            argument at position #{pos} is not compatible with compiled function template.

            #{Nx.Defn.TemplateDiff.build_and_inspect(Enum.fetch!(template_args, pos - 1), arg, "Expected", "Argument")}
            """
          end

          {:ok, {templates, [fun | acc]}}

        _arg_template, _fun, {[], acc} ->
          raise ArgumentError, """
          cannot invoke compiled function because the given arguments do not match compiled arguments

          Compiled with:

          #{inspect(template_args)}

          Got:

          #{inspect(Enum.reverse(acc, [arg | args]))}
          """
      end)

    compile_flatten(args, templates, template_args, pos + 1, acc)
  end

  defp compile_flatten([], [], _template_args, _pos, acc), do: Enum.reverse(acc)

  defp compile_flatten([], _templates, template_args, _pos, acc) do
    raise ArgumentError, """
    cannot invoke compiled function because the given arguments do not match compiled arguments

    Compiled with:

    #{inspect(template_args)}

    Got:

    #{inspect(Enum.reverse(acc))}
    """
  end

  @doc """
  Wraps an anonymous function with just-in-time compilation.

  Once invoked, the wrapped anonymous function will perform just
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
    wrap(fun, &jit_apply(fun, &1, opts))
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
    {fun, params, _templates, flatten} = Nx.Defn.Compiler.to_lazy_params(fun, args)
    [res] = Nx.Defn.Compiler.__jit__(fun, params, [flatten], opts)
    res
  end

  @doc """
  Wraps an anonymous function to return its underlying defn expression.

  > #### Warning {: .warning}
  >
  > This function must be invoked for debugging purposes only.

  ## Options

    * `:hooks` - a map of hooks to execute. See `Nx.Defn.Kernel.hook/3`

  """
  def debug_expr(fun, opts \\ []) when is_function(fun) and is_list(opts) do
    wrap(fun, &debug_expr_apply(fun, &1, opts))
  end

  @doc """
  Invokes the anonymous function to return its underlying defn expression.

  > #### Warning {: .warning}
  >
  > This function must be invoked for debugging purposes only.

  It accepts the same options as `debug_expr/2`.
  """
  def debug_expr_apply(fun, args, opts \\ []) when is_function(fun) and is_list(args) do
    opts = opts |> prepare_options() |> Keyword.put(:compiler, Nx.Defn.Debug)
    {fun, params, _templates, flatten} = Nx.Defn.Compiler.to_lazy_params(fun, args)
    [res] = Nx.Defn.Compiler.__jit__(fun, params, [flatten], opts)
    res
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

      stream = Nx.Defn.stream(&Streamed.sum/2, [Nx.template({}, {:s, 32}), 0])

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

  ## Beware: deadlocks

  Some backends (such as XLA) place locks around devices. For example,
  if you start streaming on the GPU, you cannot perform any other
  operation on the GPU until streaming is over.

  This means if we modify the loop above to the following:

      for i <- 1..5 do
        Nx.Stream.send(stream, Nx.tensor(i) |> Nx.multiply(2))
        IO.inspect {:chunk, Nx.Stream.recv(stream)}
      end

  The loop may deadlock at the time it performs the multiplication.
  In practice, this means you should perform the streaming on the GPU
  and the remaining operations on the CPU. If you only have a single
  device (i.e. only a CPU), then it may not be possible to perform the
  above and you will have to restructure your code to manipulate the
  input before streaming starts.
  """
  @deprecated "Move the streaming loop to Elixir instead"
  def stream(fun, args, opts \\ [])
      when is_function(fun) and is_list(args) and is_list(opts) do
    if Nx.Defn.Compiler.current() do
      raise "cannot call Nx.Defn.stream/3 when there is a JIT compilation happening"
    end

    opts = prepare_options(opts)
    {fun, params, _templates, flatten} = Nx.Defn.Compiler.to_lazy_params(fun, args)

    case args do
      [_input, acc | _] ->
        acc = Nx.Defn.Composite.traverse(acc, &Nx.to_tensor/1)
        [stream] = Nx.Defn.Compiler.__stream__(fun, hd(params), acc, params, [flatten], opts)
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
    Nx.Defn.Compiler.fun(arity, callback)
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

  ## Examples

      defn tanh_grad(t) do
        grad(t, &Nx.tanh/1)
      end

  To differentiate on multiple vars, pass a tuple as first argument:

      defn tanh_power_grad(a, b) do
        grad({a, b}, fn {a, b} -> Nx.tanh(a) + Nx.pow(b, 2) end)
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

  ## Examples

      defn tanh_grad(t) do
        value_and_grad(t, &Nx.tanh/1)
      end

  To differentiate on multiple vars, pass a tuple as first argument:

      defn tanh_power_grad(a, b) do
        value_and_grad({a, b}, fn {a, b} -> Nx.tanh(a) + Nx.pow(b, 2) end)
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
    define_defn(:def, call, block, __CALLER__)
  end

  @doc """
  Defines a private numerical function.

  Private numerical functions are always inlined by
  their callers at compilation time. This happens to
  all local function calls within `defn`.
  """
  defmacro defnp(call, do: block) do
    define_defn(:defp, call, block, __CALLER__)
  end

  @doc """
  Can be used to define bodiless clauses for multi-clause transforms.

  See also: `deftransform/2`

  ## Examples

      deftransform foo(bar, baz \\ 1)
      deftransform foo(bar, 1), do: bar
      deftransform foo(bar, baz), do: bar + baz
  """
  defmacro deftransform(call) do
    define_transform(:def, call, nil, __CALLER__)
  end

  @doc """
  Defines a transform that executes the given `fun` with `arg`
  when building `defn` expressions.

  ## Example

  Take the following defn expression:

      defn tanh_power(a, b) do
        Nx.tanh(a) + Nx.pow(b, 2)
      end

  Let's see a trivial example, which is to use `IO.inspect/1` to
  print a tensor expression at definition time:

      defn tanh_power(a, b) do
        Nx.tanh(a) + Nx.pow(b, 2) |> my_inspect()
      end

      deftransformp my_inspect(expr), do: IO.inspect(expr)

  Or:

      defn tanh_power(a, b) do
        res = Nx.tanh(a) + Nx.pow(b, 2)
        my_inspect(res)
        res
      end

  When invoked in both cases, it will print the expression being built
  by `defn`:

      #Nx.Defn.Expr<
        parameter a
        parameter c
        b = tanh [ a ] ()
        d = pow [ c, 2 ] ()
        e = add [ b, d ] ()
      >

  Although, for convenience, you might use `print_expr/2` instead.
  """
  defmacro deftransform(call, do: block) do
    define_transform(:def, call, block, __CALLER__)
  end

  @doc """
  Private function version for `deftransform/1`
  """
  defmacro deftransformp(call) do
    define_transform(:defp, call, nil, __CALLER__)
  end

  @doc """
  Private function version for `deftransform/2`
  """
  defmacro deftransformp(call, do: block) do
    define_transform(:defp, call, block, __CALLER__)
  end

  ## Callbacks

  defp define_defn(kind, call, block, env) do
    assert_no_guards!(kind, call, env)
    # Note name here is not necessarily an atom due to unquote(name) support
    {name, args} = decompose_call!(kind, call, env)
    arity = length(args)

    defaults =
      for {{:\\, meta, [_, default]}, i} <- Enum.with_index(args),
          do: {i, {meta, Macro.escape(default)}},
          into: []

    quote do
      unquote(__MODULE__).__define__(
        __MODULE__,
        unquote(kind),
        unquote(name),
        unquote(arity),
        :numerical,
        %{unquote_splicing(defaults)}
      )

      unquote(kind)(unquote(call)) do
        use Nx.Defn.Kernel
        unquote(block)
      end

      Process.delete(Nx.Defn)
    end
  end

  defp define_transform(kind, call, block, env) do
    # Note name here is not necessarily an atom due to unquote(name) support
    {name, args} = decompose_call!(kind, call, env)
    arity = length(args)

    defaults =
      for {{:\\, meta, [_, default]}, i} <- Enum.with_index(args),
          do: {i, {meta, Macro.escape(default)}},
          into: []

    define_ast =
      quote do
        unquote(__MODULE__).__define__(
          __MODULE__,
          unquote(kind),
          unquote(name),
          unquote(arity),
          :transform,
          %{unquote_splicing(defaults)}
        )
      end

    def_ast =
      if block do
        quote do
          Kernel.unquote(kind)(unquote(call), do: unquote(block))
        end
      else
        quote do
          Kernel.unquote(kind)(unquote(call))
        end
      end

    {:__block__, [], [define_ast, def_ast]}
  end

  defp decompose_call!(kind, {:when, _, [call, _guards]}, env),
    do: decompose_call!(kind, call, env)

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
  @defn_exports_key :__defn_exports__

  @doc false
  def __define__(module, kind, name, arity, type, defaults) do
    exports =
      if exports = Module.get_attribute(module, @defn_exports_key) do
        exports
      else
        Module.put_attribute(module, :before_compile, __MODULE__)
        %{}
      end

    current_export = %{
      type: type,
      kind: kind,
      defaults: defaults
    }

    exports =
      if type == :transform do
        # This will ensure that we capture the defaults for a bodiless head
        # while keeping the definitions properly for the same arity
        Map.update(exports, {name, arity}, current_export, fn item ->
          %{
            type: item.type || current_export.type,
            kind: item.kind || current_export.kind,
            defaults: if(item.defaults == [], do: current_export.defaults, else: item.defaults)
          }
        end)
      else
        Process.put(Nx.Defn, true)
        Map.put(exports, {name, arity}, current_export)
      end

    Module.put_attribute(module, @defn_exports_key, exports)
    :ok
  end

  defp compile_error!(env, description) do
    raise CompileError, line: env.line, file: env.file, description: description
  end

  @doc false
  defmacro __before_compile__(env) do
    defn_exports = Module.get_attribute(env.module, @defn_exports_key)
    Nx.Defn.Compiler.__compile__(env, defn_exports)
  end
end
