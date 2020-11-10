defmodule ClientTest do
  use ExUnit.Case
  alias Exla.Client, as: Client

  test "start_link/1 starts up on host device" do
    assert {:ok, pid} = Client.start_link(:host)
  end
end