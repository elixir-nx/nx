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
  and bitwise operators (`+`, `-`, `&&&`, `<<<`, etc) in Elixir
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

  ### Options

  The `Nx.Defn` compiler supports the following options:

    * `max_unsigned_type: type` - the same as `Nx.Defn.Kernel.max_unsigned_type/2`
    * `max_signed_type: type` - the same as `Nx.Defn.Kernel.max_signed_type/2`
    * `max_float_type: type` - the same as `Nx.Defn.Kernel.max_float_type/2`

  ## Inputs and outputs types

  The inputs to `defn` functions must be either tuples, numbers,
  or tensors. To pass non-numerical values to numerical definitions,
  they must be declared as default arguments (see next subsection).

  `defn` functions can only return tensors or tuples of tensors.

  ### Default arguments

  `defn` functions also support default arguments. They are typically
  used as options. For example, imagine you want to create a function
  named zeros, which returns a tensor of zeroes with a given type and
  shape. It could be implemented like this:

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

  ### Tuples and pattern matching

  When passing tuples as inputs to `defn` functions, the tuples
  must be matched on the function head. For example, this is valid:

      defn my_example({a, b}, c), do: a * b + c

  This is not:

      defn my_example(ab, c) do
        {a, b} = ab
        a * b + c
      end

  This, however, works:

      defn my_example({_, _} = ab, c) do
        {a, b} = ab
        a * b + c
      end

  In other words, it is important for `defn` to the see the shapes
  of the input. If you write the latter format and call `defn` from
  Elixir, `defn` will raise.

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
  Invokes the anonymous function with asynchronously with
  just-in-time compilation.

  The anonymous function will be invoked with tensor expressions
  which are JIT compiled and then invoked. The anonymous function
  will also run outside of the current Elixir process. You can
  retrieve the result by calling `Nx.Async.await!/1`. Take the
  following definition:

      defn softmax(t), do: Nx.exp(t) / Nx.sum(Nx.exp(t))

  We can invoke it asynchronously as follows:

      some_struct = Nx.Defn.async(&Mod.softmax/1, [my_tensor], EXLA)
      Nx.Async.await!(some_struct)

  **Note:** similar to `jit/4`, `async/4` will ignore the `@defn_compiler`
  on the executed function. Be sure to pass the `compiler` and its `opts`
  as arguments instead.
  """
  def async(fun, args, compiler \\ Nx.Defn.Evaluator, opts \\ [])
      when is_function(fun) and is_list(args) and is_atom(compiler) and is_list(opts) do
    Nx.Defn.Compiler.__async__(fun, args, compiler, opts)
  end

  @doc """
  Invokes the anonymous function with just-in-time compilation.

  The anonymous function will be invoked with tensor expressions
  which are JIT compiled and then invoked. For example, take the
  following definition:

      defn softmax(t), do: Nx.exp(t) / Nx.sum(Nx.exp(t))

  **Note:** `jit/4` will ignore the `@defn_compiler` on the executed
  function. Be sure to pass the `compiler` and its `opts` as arguments
  instead:

      Nx.Defn.jit(&Mod.softmax/1, [my_tensor], EXLA)
      Nx.Defn.jit(&Mod.softmax/1, [my_tensor], EXLA, run_options: [keep_on_device: true])

  """
  def jit(fun, args, compiler \\ Nx.Defn.Evaluator, opts \\ [])
      when is_function(fun) and is_list(args) and is_atom(compiler) and is_list(opts) do
    Nx.Defn.Compiler.__jit__(fun, args, compiler, opts)
  end

  @doc """
  Ahead-of-time compiles the anonymous function with the given
  defn compiler.
  """
  def aot(fun, args, compiler, opts \\ [])
      when is_function(fun) and is_list(args) and is_atom(compiler) and is_list(opts) do
    Nx.Defn.Compiler.__aot__(fun, args, compiler, opts)
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
