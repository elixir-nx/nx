client = Exla.Client.create_client()

t1 = for i <- 1..100, do: i
t2 = for i <- 1..10, do: i
t1_bin = for i <- t1, do: <<i::32-little>>, into: <<>>
t2_bin = for i <- t2, do: <<i::32-little>>, into: <<>>

t1_shape = Exla.Shape.make_shape(:int32, {10, 10})
t2_shape = Exla.Shape.make_shape(:int32, {10})

t1_tensor = %Exla.Tensor{data: {:binary, t1_bin}, shape: t1_shape, device: {:cpu, 0}}
t2_tensor = %Exla.Tensor{data: {:binary, t2_bin}, shape: t2_shape, device: {:cpu, 0}}

build_slice_exec =
  fn ->
    builder = Exla.Builder.new("slice")
    x = Exla.Op.parameter(builder, 0, t1_shape, "x")
    ast = Exla.Op.slice(x, {2, 2}, {5, 5}, {1, 1})

    comp = Exla.Builder.build(ast)
    Exla.Client.compile(client, comp, {t1_shape})
  end

# Needs dim checks, SegFaults without error on bad dims!
build_slice_in_dim_exec =
  fn ->
    builder = Exla.Builder.new("slice_in_dim")
    x = Exla.Op.parameter(builder, 0, t1_shape, "x")
    ast = Exla.Op.slice_in_dim(x, 2, 5, 2, 1)

    comp = Exla.Builder.build(ast)
    Exla.Client.compile(client, comp, {t1_shape})
  end

build_dynamic_slice =
  fn ->
    builder = Exla.Builder.new("dynamic_slice")
    x = Exla.Op.parameter(builder, 0, t1_shape, "x")
    a = Exla.Op.constant(builder, 2)
    b = Exla.Op.constant(builder, 2)
    idx = Exla.Op.add(a, b)
    ast = Exla.Op.dynamic_slice(x, {idx, idx}, {5, 5})

    comp = Exla.Builder.build(ast)
    Exla.Client.compile(client, comp, {t1_shape})
  end

build_dynamic_update_slice =
  fn ->
    builder = Exla.Builder.new("dynamic_update_slice")
    x = Exla.Op.parameter(builder, 0, t2_shape, "x")
    a = Exla.Op.constant(builder, 2)
    b = Exla.Op.constant(builder, 2)
    idx = Exla.Op.add(a, b)
    update = Exla.Op.constant_r1(builder, 4, 1000)
    ast = Exla.Op.dynamic_update_slice(x, update, {idx})

    comp = Exla.Builder.build(ast)
    Exla.Client.compile(client, comp, {t2_shape})
  end

slice = build_slice_exec.()
slice_in_dim = build_slice_in_dim_exec.()
dynamic_slice = build_dynamic_slice.()
dynamic_update_slice = build_dynamic_update_slice.()

%Exla.Tensor{data: {:ref, slice_result}} = Exla.LocalExecutable.run(slice, {t1_tensor})
%Exla.Tensor{data: {:ref, slice_in_dim_result}} = Exla.LocalExecutable.run(slice_in_dim, {t1_tensor})
%Exla.Tensor{data: {:ref, dynamic_slice}} = Exla.LocalExecutable.run(dynamic_slice, {t1_tensor})
%Exla.Tensor{data: {:ref, dynamic_update_slice}} = Exla.LocalExecutable.run(dynamic_update_slice, {t2_tensor})

IO.puts Exla.NIF.literal_to_string(Exla.NIF.shaped_buffer_to_literal(client.ref, slice_result))
IO.puts Exla.NIF.literal_to_string(Exla.NIF.shaped_buffer_to_literal(client.ref, slice_in_dim_result))
IO.puts Exla.NIF.literal_to_string(Exla.NIF.shaped_buffer_to_literal(client.ref, dynamic_slice))
IO.puts Exla.NIF.literal_to_string(Exla.NIF.shaped_buffer_to_literal(client.ref, dynamic_update_slice))
