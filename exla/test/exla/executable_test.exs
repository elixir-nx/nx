defmodule EXLA.ExecutableTest do
  use ExUnit.Case, async: true

  alias EXLA.{Buffer, Executable, Op, Shape}

  import EXLAHelpers

  test "raises on invalid tuples" do
    t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
    t2 = %Buffer{data: <<2::32-native>>, shape: Shape.make_shape({:s, 32}, {})}

    assert_raise ArgumentError, ~r"can only compile computations with a tuple at the root", fn ->
      run([t1, t2], [], fn b, x, y ->
        Op.tuple(b, [Op.tuple(b, [x]), Op.tuple(b, [y])])
      end)
    end

    assert_raise ArgumentError, ~r"can only compile computations with a tuple at the root", fn ->
      run([t1, t2], [], fn _b, x, y -> Op.add(x, y) end)
    end
  end

  describe "run" do
    test "succeeds with no inputs and default options" do
      assert [%Buffer{data: <<1::32-native>>}] =
               run([], fn b ->
                 Op.tuple(b, [Op.constant_r0(b, 1, {:s, 32})])
               end)
    end

    test "succeeds with 2 inputs and default options" do
      t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
      t2 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}

      assert [%Buffer{data: <<2::32-native>>}] =
               run([t1, t2], fn b, x, y -> Op.tuple(b, [Op.add(x, y)]) end)
    end

    test "succeeds when data is preloaded" do
      t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
      t2 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
      t1 = Buffer.place_on_device(t1, client(), 0)
      t2 = Buffer.place_on_device(t2, client(), 0)

      assert [%Buffer{ref: {ref, _}}] =
               run([t1, t2], [keep_on_device: true], fn b, x, y -> Op.tuple(b, [Op.add(x, y)]) end)

      assert is_reference(ref)

      # We can run it again
      assert [%Buffer{ref: {ref, _}}] =
               run([t1, t2], [keep_on_device: true], fn b, x, y -> Op.tuple(b, [Op.add(x, y)]) end)

      assert is_reference(ref)

      # And they have not changed
      assert Buffer.read(t1.ref) == <<1::32-native>>
      assert Buffer.read(t2.ref) == <<1::32-native>>
    end

    test "succeeds with keep_on_device is true" do
      t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
      t2 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}

      assert [%Buffer{ref: {ref, _}}] =
               run([t1, t2], [keep_on_device: true], fn b, x, y -> Op.tuple(b, [Op.add(x, y)]) end)

      assert is_reference(ref)
    end

    test "succeeds with data from a previous run" do
      t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
      t2 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}

      exec = compile([t1.shape, t2.shape], fn b, x, y -> Op.tuple(b, [Op.add(x, y)]) end)
      assert [t3 = %Buffer{ref: {ref, _}}] = Executable.run(exec, [t1, t2], keep_on_device: true)
      assert is_reference(ref)
      assert [%Buffer{data: <<4::32-native>>}] = Executable.run(exec, [t3, t3])
    end

    test "succeeds with mixed data" do
      t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
      t2 = %Buffer{data: <<2::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
      t1 = Buffer.place_on_device(t1, client(), 0)

      assert [%Buffer{data: <<3::32-native>>}] =
               run([t1, t2], fn b, x, y -> Op.tuple(b, [Op.add(x, y)]) end)
    end

    test "suceeds with tuple return" do
      t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
      t2 = %Buffer{data: <<2::32-native>>, shape: Shape.make_shape({:s, 32}, {})}

      assert [%Buffer{data: <<1::32-native>>}, %Buffer{data: <<2::32-native>>}] =
               run([t1, t2], fn b, x, y -> Op.tuple(b, [x, y]) end)
    end

    test "succeeds with tuple return and keep_on_device true" do
      t1 = %Buffer{data: <<1::32-native>>, shape: Shape.make_shape({:s, 32}, {})}
      t2 = %Buffer{data: <<2::32-native>>, shape: Shape.make_shape({:s, 32}, {})}

      assert [a = %Buffer{}, b = %Buffer{}, c = %Buffer{}] =
               run([t1, t2], [keep_on_device: true], fn b, x, y ->
                 Op.tuple(b, [x, y, Op.add(x, y)])
               end)

      assert <<1::32-native>> == Buffer.read(a.ref)
      assert <<2::32-native>> == Buffer.read(b.ref)
      assert <<3::32-native>> == Buffer.read(c.ref)
    end
  end
end
