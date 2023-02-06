ExUnit.start(assert_receive_timeout: 1000)

cond do
  :distributed in Keyword.get(ExUnit.configuration(), :exclude, []) ->
    :ok

  Code.ensure_loaded?(:peer) and match?({:ok, _}, Node.start(:"primary@127.0.0.1", :longnames)) ->
    {:ok, _pid, node} = :peer.start(%{name: :"secondary@127.0.0.1"})
    true = :erpc.call(node, :code, :set_path, [:code.get_path()])
    {:ok, _} = :erpc.call(node, :application, :ensure_all_started, [:nx])

  true ->
    ExUnit.configure(exclude: [:distributed])
end
