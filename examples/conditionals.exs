client = Exla.Client.create_client(platform: :host)

t1 = for i <- 1..10, do: i
t1_bin = for i <- t1, do: <<i::32-little>>, into: <<>>
t1_shape = Exla.Shape.make_shape(:int32, {10})
t1_tensor = %Exla.Tensor{data: {:binary, t1_bin}, shape: t1_shape, device: {:cpu, 0}}

build_if_conditional_exec =
fn ->
  builder = Exla.Builder.new("if_conditional")
  true_branch = Exla.Builder.new(builder, "true_branch")
  false_branch = Exla.Builder.new(builder, "false_branch")

  x = Exla.Op.parameter(builder, 0, t1_shape, "x")

  # True branch
  a = Exla.Op.parameter(true_branch, 0, t1_shape, "a")
  true_ast = Exla.Op.add(a, a)
  true_comp = Exla.Builder.build(true_ast)

  # False branch
  b = Exla.Op.parameter(false_branch, 0, t1_shape, "b")
  false_ast = Exla.Op.div(b, b)
  false_comp = Exla.Builder.build(false_ast)

  # Predicate - Has to be a scalar
  pred = Exla.Op.ne(Exla.Op.constant(builder, 0), Exla.Op.constant(builder, 1))

  # AST - Specifies `x` as the True/False operand. The operand can be anything
  # as long as the type matches what both computations expect. The computations
  # must return the same type
  ast = Exla.Op.conditional(pred, x, true_comp, x, false_comp)
  comp = Exla.Builder.build(ast)

  exec = Exla.Client.compile(client, comp, {t1_shape})
  exec
end

if_exec = build_if_conditional_exec.()

build_multi_conditiona_exec =
  fn ->
    builder = Exla.Builder.new("multi_conditional")
    branch1 = Exla.Builder.new(builder, "branch1")
    branch2 = Exla.Builder.new(builder, "branch2")
    branch3 = Exla.Builder.new(builder, "branch3")

    x = Exla.Op.parameter(builder, 0, t1_shape, "x")

    index = Exla.Op.constant(builder, 1)

    a = Exla.Op.parameter(branch1, 0, t1_shape, "a")
    b = Exla.Op.parameter(branch2, 0, t1_shape, "b")
    c = Exla.Op.parameter(branch3, 0, t1_shape, "c")

    comp1 = Exla.Builder.build(Exla.Op.add(a, a))
    comp2 = Exla.Builder.build(Exla.Op.div(b, b))
    comp3 = Exla.Builder.build(c)

    ast = Exla.Op.conditional(index, {comp1, comp2, comp3}, {x, x, x})
    comp = Exla.Builder.build(ast)

    exec = Exla.Client.compile(client, comp, {t1_shape})
    exec
  end

multi_exec = build_multi_conditiona_exec.()

IO.inspect Exla.LocalExecutable.run(if_exec, {t1_tensor})
IO.inspect Exla.LocalExecutable.run(multi_exec, {t1_tensor})
