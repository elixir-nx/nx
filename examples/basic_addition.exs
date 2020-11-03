# Initialize the local client, this just runs on CPU
Exla.get_or_create_local_client()

# Create 2 1D xla constants, it's basically a list of length 5 filled with 5's
t1 = Exla.constant_r1(5, 5)
t2 = Exla.constant_r1(5, 5)

# Add them together
root = Exla.add(t1, t2)

# Build computation from root
comp = Exla.build(root)

# Compile to a local executable, no arguments or options for now
exec = Exla.compile(comp, {}, %{})

# Run the executable, no arguments or options for now
Exla.run(hd(exec), {}, %{})