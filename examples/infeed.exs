client = Exla.Client.fetch!(:default)

builder = Exla.Builder.new("infeed")
shape = Exla.Shape.make_shape({:s, 64}, {})

token = Exla.Op.create_token(builder)

{x, token} = Exla.Op.infeed(token, shape)
{y, token} = Exla.Op.infeed(token, shape)
z = Exla.Op.add(x, y)

intermediate_value = Exla.Op.outfeed(z, token, shape)

ast = Exla.Op.add(z, x)

comp = Exla.Builder.build(ast)

exec = Exla.Client.compile(client, comp, [])

b1 = Exla.Buffer.buffer(<<1::64-native>>, shape)
b2 = Exla.Buffer.buffer(<<1::64-native>>, shape)

run = Task.async(fn -> Exla.Executable.run(exec, []) end)
Exla.Buffer.to_infeed(b1, client, 0)
Process.sleep(3000)
Exla.Buffer.to_infeed(b2, client, 0)
IO.inspect Exla.Buffer.from_outfeed(client, shape, 0)
IO.inspect Task.await(run, :infinity)