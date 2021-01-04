defmodule Exla.ExecutableTest do
  use ExUnit.Case, async: true

  alias Exla.{Buffer, Executable, Op, Shape, ShardedBuffer}

  import ExlaHelpers

  test "run/2 succeeds with no inputs and default options" do
    assert %Buffer{data: <<1, 0, 0, 0>>} =
             run([], fn builder ->
               Op.constant_r0(builder, 1, {:s, 32})
             end)
  end

  test "run/2 succeeds with 1 input and default options" do
    t1 = %Buffer{data: <<1::8-native>>, shape: Shape.make_shape({:s, 8}, {})}
    assert %Buffer{data: <<1>>} = run([t1], fn _builder, param -> param end)
  end

  test "run/2 succeeds with 2 inputs and default options" do
    t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
    t2 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
    assert %Buffer{data: <<2::32-native>>} = run([t1, t2], fn _builder, x, y -> Op.add(x, y) end)
  end

  test "run/2 returns a ref when keep_on_device is true" do
    t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
    t2 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}

    assert %Buffer{ref: {ref, _}} =
             run([t1, t2], [keep_on_device: true], fn _builder, x, y -> Op.add(x, y) end)

    assert is_reference(ref)
  end

  test "run/2 succeeds when data is pre-loaded" do
    t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
    t2 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
    t1 = Buffer.place_on_device(t1, client(), 0)
    t2 = Buffer.place_on_device(t2, client(), 0)

    assert %Buffer{ref: {ref, _}} =
             run([t1, t2], [keep_on_device: true], fn _builder, x, y -> Op.add(x, y) end)

    assert is_reference(ref)

    # We can run it again
    assert %Buffer{ref: {ref, _}} =
             run([t1, t2], [keep_on_device: true], fn _builder, x, y -> Op.add(x, y) end)

    assert is_reference(ref)

    # And they have not changed
    assert Buffer.read(t1.ref) == <<1::32-native>>
    assert Buffer.read(t2.ref) == <<1::32-native>>
  end

  test "run/2 succeeds with data from a previous run" do
    t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
    t2 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}

    exec = compile([t1.shape, t2.shape], fn _builder, x, y -> Op.add(x, y) end)
    assert t3 = %Buffer{ref: {ref, _}} = Executable.run(exec, [t1, t2], keep_on_device: true)
    assert is_reference(ref)
    assert %Buffer{data: <<4::32-native>>} = Executable.run(exec, [t3, t3])
  end

  test "run/2 succeeds with mixed data" do
    t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
    t2 = %Buffer{data: <<2::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
    t1 = Buffer.place_on_device(t1, client(), 0)
    assert %Buffer{data: <<3::32-native>>} = run([t1, t2], fn _builder, x, y -> Op.add(x, y) end)
  end

  test "run/2 with constants" do
    assert %Buffer{data: <<0::64-unsigned>>} =
             run([], fn builder ->
               Op.constant_r0(builder, 0, {:u, 64})
             end)
  end

  test "run/2 returns a tuple" do
    t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
    t2 = %Buffer{data: <<2::32-native>>, shape: Shape.make_shape({:s, 32}, {})}

    assert {:tuple,
            [
              {:tuple, [%Buffer{data: <<1, 0, 0, 0>>}, %Buffer{data: <<2, 0, 0, 0>>}]},
              {:tuple, []},
              %Buffer{data: <<2, 0, 0, 0>>}
            ]} =
             run([t1, t2], fn builder, x, y ->
               Op.tuple(
                 builder,
                 [Op.tuple(builder, [x, y]), Op.tuple(builder, []), y]
               )
             end)
  end

  test "run/2 returns with tuple and keep_on_device true" do
    t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
    t2 = %Buffer{data: <<2::32-native>>, shape: Shape.make_shape({:s, 32}, {})}

    assert {:tuple, [{:tuple, [a = %Buffer{}, b = %Buffer{}]}, {:tuple, []}, c = %Buffer{}]} =
             run([t1, t2], [keep_on_device: true], fn builder, x, y ->
               Op.tuple(
                 builder,
                 [Op.tuple(builder, [x, y]), Op.tuple(builder, []), y]
               )
             end)

    assert <<1, 0, 0, 0>> == Buffer.read(a.ref)
    assert <<2, 0, 0, 0>> == Buffer.read(b.ref)
    assert <<2, 0, 0, 0>> == Buffer.read(c.ref)
  end

  @tag :multi_device
  test "run_parallel/2 succeeds with 2 inputs and default options" do
    d1 = <<1::64-native, 2::64-native>>
    d2 = <<3::64-native, 4::64-native>>
    shape = Shape.make_shape({:s, 64}, {2, 1})

    sb1 = ShardedBuffer.sharded_buffer(d1, shape)
    sb2 = ShardedBuffer.sharded_buffer(d2, shape)

    assert %ShardedBuffer{buffers: [b1, b2]} =
             run_parallel([sb1, sb2], 2, fn _, x, y ->
               Op.add(x, y)
             end)

    assert b1.data == <<4::64-native>>
    assert b2.data == <<6::64-native>>
  end

  @tag :multi_device
  test "run_parallel/2 succeeds with 2 inputs and keep_on_device" do
    d1 = <<1::64-native, 2::64-native>>
    d2 = <<3::64-native, 4::64-native>>
    shape = Shape.make_shape({:s, 64}, {2, 1})

    sb1 = ShardedBuffer.sharded_buffer(d1, shape)
    sb2 = ShardedBuffer.sharded_buffer(d2, shape)

    assert %ShardedBuffer{buffers: [b1, b2]} =
             run_parallel([sb1, sb2], 2, [keep_on_device: true], fn _, x, y ->
               Op.add(x, y)
             end)

    assert Buffer.read(b1.ref) == <<4::64-native>>
    assert Buffer.read(b2.ref) == <<6::64-native>>
  end

  @tag :multi_device
  test "run_parallel/2 succeeds with mixed data" do
    d1 = <<1::64-native, 2::64-native>>
    d2 = <<3::64-native, 4::64-native>>
    shape = Shape.make_shape({:s, 64}, {2, 1})

    sb1 = ShardedBuffer.sharded_buffer(d1, shape)
    sb2 = ShardedBuffer.sharded_buffer(d2, shape)
    sb2 = ShardedBuffer.place_on_device(sb2, client())

    assert %ShardedBuffer{buffers: [b1, b2]} =
             run_parallel([sb1, sb2], 2, fn _, x, y ->
               Op.add(x, y)
             end)

    assert b1.data == <<4::64-native>>
    assert b2.data == <<6::64-native>>
  end
end
