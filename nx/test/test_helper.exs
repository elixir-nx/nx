ExUnit.start(assert_receive_timeout: 1000)

cond do
  :distributed in Keyword.get(ExUnit.configuration(), :exclude, []) ->
    :ok

  Code.ensure_loaded?(:peer) and {"", 0} == System.cmd("epmd", ["-daemon"]) and
      match?({:ok, _}, Node.start(:"primary@127.0.0.1", :longnames)) ->
    {:ok, _pid, node2} = :peer.start(%{name: :"secondary@127.0.0.1"})
    {:ok, _pid, node3} = :peer.start(%{name: :"tertiary@127.0.0.1", args: ~w(-hidden)c})

    for node <- [node2, node3] do
      true = :erpc.call(node, :code, :set_path, [:code.get_path()])
      {:ok, _} = :erpc.call(node, :application, :ensure_all_started, [:nx])
    end

  true ->
    ExUnit.configure(exclude: [:distributed])
end
