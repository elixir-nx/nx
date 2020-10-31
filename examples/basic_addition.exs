# This API doesn't feel very Elixir-y because it's really stateful right now.

# Initialize the local client, this just runs on CPU
Exla.get_or_create_local_client()

# Create 2 1D xla constants, it's basically a list of length 5 filled with 5's
t1 = Exla.constant_r1(5, 5)
t2 = Exla.constant_r1(5, 5)

# Add them together
Exla.add(t1, t2)

# Run the result, this will return a string for now
IO.inspect Exla.run()