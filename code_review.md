review notes:

- Nx.Defn Expr added a third out_template argument, but the template can be inferred from the expression itself.
- exla: elixir_call_test and elixir_call_exla_test are redundant with each other. we can combine them in a single file.

- callback server and the rest of the code seem to assume that tensor arguments are always at the beginning of the function. We should enforce this more clearly and document this.
- given that the callback server is named, enif_whereis_pid(https://www.erlang.org/doc/apps/erts/erl_nif.html#enif_whereis_pid) could be used to fetch the current pid for the function.
- EXLA.CallbackServer decode_args uses from_binary without options. We should keep track of the backend options such as the device used to allocate an EXLA tensor. Ideally, we shouldn't even be copying data back and forth. shape should already be passed as a tuple from the NIF.
- EXLA.CallbackServer does encode_reply/encode_outputs really need to "Nx.to_binary" the results? It seems like we should be able to pass EXLA Buffer refs back and forth.

- EXLA.Defn operands = call_args ++ [callback_id_value] we should prepend the id instead of append id
- What is api_version in EXLA.MLIR.Value.elixir_call?
- Should the callback_id instead be an attribute given that it should not change during execution?

----

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    exla_elixir_callback, exla_elixir_callback_impl,
    ffi::Ffi::Bind()
        .RemainingArgs()
        .RemainingRets());

This could receive the id in the first argument, and the tensors in the second.

exla.cc:550 I think there is already another function for mapping exla to nx types.

exla.cc:648 FINE_NIF(elixir_callback_reply, 0) I think is missing an IO-bound attr