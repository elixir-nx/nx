defmodule Nx.BinaryBackend.PadTest do
  use ExUnit.Case, async: true
  alias Nx.BinaryBackend.Pad
  
  doctest Pad

  defp build_tensor(shape, {_, sizeof} = type, data) do
    shape_size = Nx.size(shape)
    data_sizeof = bit_size(data)
    data_size = div(data_sizeof, sizeof)

    assert data_size == shape_size, "mismatch" <>
           " between shape size (#{shape_size}) and data size" <>
           " (#{data_size}) with type #{inspect(type)} for shape" <>
           " #{inspect(shape)} and data #{inspect(data)}"
    t = Nx.iota(shape, type: type)
    %{t | data: %{t.data | state: data}}
  end

  defp run_padding(type, shape, padding_configs, expected_list, opts \\ []) do
  
    t = Nx.iota(shape, type: type)
    pad_value = opts[:pad_value] || Nx.tensor(100, type: type)

    good_pad = Nx.pad(t, pad_value, padding_configs)

    expected = Nx.tensor(expected_list, type: type)

    assert good_pad == expected

    expected_shape = Nx.Shape.pad(shape, padding_configs)
    assert expected_shape == Nx.shape(expected)
    data = Pad.run(t, pad_value, padding_configs)
    
    insp_data = inspect(data, binaries: :as_binaries)
    insp_expt = inspect(expected.data.state, binaries: :as_binaries)
    assert insp_data == insp_expt, """
    data mismatch -

      want:  #{insp_expt}
      got:   #{insp_data}
      
      
      tensor: #{inspect(t)}
      pad_value: #{inspect(pad_value)}
      padding_configs: #{inspect(padding_configs)}
    """


    t2 = build_tensor(expected_shape, type, data)

    assert t2 == expected

    t2
  end

  describe "run/3" do
    test "works for simplest case" do
      expected = [0]
      run_padding({:u, 8}, {1}, [{0, 0, 0}], expected)
    end

    test "works for rank 2 simplest case" do
      expected = [[0, 1], [2, 3]]
      run_padding({:u, 8}, {2, 2}, [{0, 0, 0}, {0, 0, 0}], expected)
    end

    test "works for rank 1 pad lo" do
      expected = [100, 0, 1]
      run_padding({:u, 8}, {2}, [{1, 0, 0}], expected)
    end
  
    test "works for rank 1 pad hi" do
      expected = [0, 1, 100]
      run_padding({:u, 8}, {2}, [{0, 1, 0}], expected)
    end

    test "works for rank 1 pad middle" do
      expected = [0, 100, 1]
      run_padding({:u, 8}, {2}, [{0, 0, 1}], expected)
    end

    test "works for rank 1 pad remove lo" do
      expected = [1]
      run_padding({:u, 8}, {2}, [{-1, 0, 0}], expected)
    end

    test "works for rank 1 pad remove hi" do
      expected = [0]
      run_padding({:u, 8}, {2}, [{0, -1, 0}], expected)
    end

    test "works for rank 1 dims > 1 with pad mid" do
      expected = [0, 100, 100, 1]
      run_padding({:u, 8}, {2}, [{0, 0, 2}], expected)
    end

    test "works for rank 2 with pad lo on axis 0" do
      expected = [[100, 100], [0, 1], [2, 3]]
      run_padding({:u, 8}, {2, 2}, [{1, 0, 0}, {0, 0, 0}], expected)
    end

    test "works for rank 2 with pad hi on axis 0" do
      expected = [[0, 1], [2, 3], [100, 100]]
      run_padding({:u, 8}, {2, 2}, [{0, 1, 0}, {0, 0, 0}], expected)
    end

    test "works for rank 2 with pad mid on axis 0" do
      expected = [[0, 1], [100, 100], [2, 3]]
      run_padding({:u, 8}, {2, 2}, [{0, 0, 1}, {0, 0, 0}], expected)
    end

    test "works for rank 2 with remove lo on axis 0" do
      expected = [[2, 3]]
      run_padding({:u, 8}, {2, 2}, [{-1, 0, 0}, {0, 0, 0}], expected)
    end

    test "works for rank 2 with remove hi on axis 0" do
      expected = [[0, 1]]
      padding_config = [{0, -1, 0}, {0, 0, 0}]
      run_padding({:u, 8}, {2, 2}, padding_config, expected)
    end


    test "works for rank 2 with remove lo and hi and add mid on axis 0" do
      expected = [[100, 100]]
      padding_config = [{-1, -1, 1}, {0, 0, 0}]
      run_padding({:u, 8}, {2, 2}, padding_config, expected)
    end


    test "works for padding config with {-1, 1, 1} in last-dim position" do
      expected = [
        [100, 1, 100],
        [100, 3, 100]
      ]
      padding_config = [{0, 0, 0}, {-1, 1, 1}]
      run_padding({:u, 8}, {2, 2}, padding_config, expected)
    end

    #     test "works for padding config with {-1, 1, 1} in last-dim position" do
    #   expected = [
    #     [100, 1, 100],
    #     [100, 3, 100]
    #   ]
    #   padding_config = [{0, 0, 0}, {-1, 1, 1}]
    #   run_padding({:u, 8}, {2, 2}, padding_config, expected)
    # end


    test "works for padding config with {-1, 1, 1} in before-last-dim position" do
      expected = [
        [100, 100, 100, 100],
        [100,   2,   3, 100],
        [100, 100, 100, 100]
      ]
      padding_config = [{-1, 1, 1}, {1, 1, 0}]
      run_padding({:u, 8}, {2, 2}, padding_config, expected)
    end

    test "works for {1, 1, 0} in before-last dim" do
      expected = [
        [100, 100, 100, 100],
        [100, 0,   1,  100],
        [100, 2,   3,  100],
        [100, 100, 100, 100]
      ]
      padding_config = [{1, 1, 0}, {1, 1, 0}]
      run_padding({:u, 8}, {2, 2}, padding_config, expected)
    end

    test "works with Nx.pad/3 docs example" do
      expected = [
        [1, 2, 3, 100],
        [5, 6, 7, 100]
      ]
      padding_config = [{0, 0, 0}, {-1, 1, 0}]
      run_padding({:u, 8}, {2, 4}, padding_config, expected)
    end

    test "works for rank 3 tensor" do
      t = Nx.tensor([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
      ], type: {:u, 8})
      assert Nx.shape(t) == {2, 2, 2}
      pad_value = Nx.tensor(0, type: {:u, 8})
      padding_config = [{0, 2, 0}, {1, 1, 0}, {1, 0, 0}]
      data = Pad.run(t, pad_value, padding_config)
      expected_t = Nx.tensor([
        [
          [0, 0, 0],
          [0, 1, 2],
          [0, 3, 4],
          [0, 0, 0]
        ],
        [
          [0, 0, 0],
          [0, 5, 6],
          [0, 7, 8],
          [0, 0, 0]
        ],
        [
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]
        ],
        [
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]
        ]
      ], type: {:u, 8})

      assert to_charlist(data) == Nx.to_flat_list(expected_t)
    end

    test "works for float tensor" do
      t = Nx.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], type: {:f, 64})
      pad_value = Nx.tensor(0.0, type: {:f, 64})
      padding_config = [{1, 2, 0}, {1, 0, 0}, {0, 1, 0}]
      
      expected_t = Nx.tensor([
        [
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0]
        ],
        [
          [0.0, 0.0, 0.0],
          [1.0, 2.0, 0.0],
          [3.0, 4.0, 0.0]
        ],
        [
          [0.0, 0.0, 0.0],
          [5.0, 6.0, 0.0],
          [7.0, 8.0, 0.0]
        ],
        [
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0]
        ],
        [
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0]
        ]
      ], type: {:f, 64})
      out = Nx.pad(t, pad_value, padding_config)
      assert out == expected_t
      data = Pad.run(t, pad_value, padding_config)
      out2 = %{out | data: %{out.data | state: data}}
      assert out2 == out
    end

    test "works for interior padding" do
      expected = [
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [100.0,   4.0, 100.0, 100.0,   5.0, 100.0, 100.0,   6.0, 100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
      ]
      t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      padding_config = [{-2, 1, 4}, {1, 3, 2}]
      assert Nx.pad(t, 100.0, padding_config) == Nx.tensor(expected)
    end


    test "works for wtfisit?" do
      expected = [
        [100.0,   3.0,   4.0],
        [100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0]
      ]
      padding_config = [{-1, 2, 0}, {1, -1, 0}]
      run_padding({:f, 32}, {2, 3}, padding_config, expected)
    end

    test "works for wtfisit?2" do
      tensor = Nx.tensor([[0, 1, 2], [3, 4, 5]], type: {:f, 32})
      pad_value = 0
      padding_config = [{-1, 2, 0}, {1, -1, 0}]
      
      expected = Nx.tensor([
        [0.0, 3.0, 4.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
      ], type: {:f, 32})
      
      out = Nx.pad(tensor, pad_value, padding_config)

      assert out == expected
    end
  end
end