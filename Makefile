setup:
	cd nx && mix deps.get
	cd exla && mix deps.get

test:
	 cd nx && mix test
	 cd exla && mix test
