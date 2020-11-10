defmodule ClientTest do
  use ExUnit.Case
  alias Exla.Client

  test "create_client/1 succeeds on host device with default args" do
    assert {:ok, %Client{}} = Client.create_client()
  end
end
