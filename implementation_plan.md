# EXLA XLA FFI Custom Call Experiment – Implementation Plan

## Goals

- Add **experimental** support for XLA FFI custom calls in EXLA.
- Replace the current `stablehlo.token` + `infeed`/`outfeed` idea with:
  - A **custom call for outfeed** that uses `enif_send` to send tensors to a BEAM process. ✅ **DONE**
  - A **custom call for infeed** that uses `nif_call` to fetch tensors from a BEAM process. ⚠️ **PARTIAL** (stub implementation, nif_call integration deferred)
- Manage infeed/outfeed via a dedicated **stream BEAM process**, identified by a `:stream_name` compiler option. ✅ **DONE**
- Pass the stream process identity (PID and optionally a tag/name) into the computation as **tensor arguments**, using `:erlang.term_to_binary/1` and `enif_binary_to_term/3`. ✅ **DONE**
- Do **not** implement general `:elixir_call` in EXLA yet; this is a focused experiment around infeed/outfeed semantics. ✅ **DONE**

---

## 1. Stream process and compiler options

### 1.1. Compiler option ⏳ **TODO**

- Introduce an EXLA-specific compiler option `:stream_name`.
- Example usage:

  ```elixir
  Nx.Defn.default_options(
    compiler: EXLA,
    compiler_options: [stream_name: :my_stream]
  )
  ```

- Semantics:
  - When `stream_name: name` is set, EXLA will:
    - Ensure a BEAM process is running and registered under `name`.
    - Use this process as the **single stream** for infeed and outfeed for this computation.
  - For the first iteration, we can choose one of:
    - Require callers/tests to start the process explicitly, or
    - Have EXLA start it if it is not already running (e.g. `Process.whereis(name) || EXLA.FFI.Stream.start_link(name: name)`).

### 1.2. Stream process module ✅ **DONE**

- Add an Elixir module (name TBD, e.g. `EXLA.FFI.Stream`) responsible for managing infeed/outfeed queues.
- **Implementation**: `exla/lib/exla/ffi/stream.ex`

- Sketch:

  ```elixir
  defmodule EXLA.FFI.Stream do
    use GenServer

    @type name :: atom()

    def start_link(opts) do
      name = Keyword.fetch!(opts, :name)
      GenServer.start_link(__MODULE__, opts, name: name)
    end

    ## External API (for tests and clients) ##

    def push_infeed(name, value) do
      GenServer.cast(name, {:push_infeed, value})
    end

    def pop_outfeed(name, timeout \\ 5_000) do
      GenServer.call(name, :pop_outfeed, timeout)
    end

    ## GenServer callbacks ##

    @impl true
    def init(_opts) do
      {:ok, %{infeed: :queue.new(), outfeed: :queue.new()}}
    end

    @impl true
    def handle_cast({:push_infeed, value}, %{infeed: in_q} = state) do
      {:noreply, %{state | infeed: :queue.in(value, in_q)}}
    end

    @impl true
    def handle_call(:pop_outfeed, _from, %{outfeed: out_q} = state) do
      case :queue.out(out_q) do
        {{:value, value}, out_q} ->
          {:reply, {:ok, value}, %{state | outfeed: out_q}}

        {:empty, _} ->
          {:reply, :empty, state}
      end
    end

    # Additional callbacks will be added later to support interactions
    # initiated from the NIF side via nif_call.
  end
  ```

- The stream process maintains:
  - An **infeed queue**: values pushed by Elixir and consumed by NIF infeed custom calls.
  - An **outfeed queue**: values produced by NIF outfeed custom calls and consumed by Elixir.

- From tests or the shell, you can interact with it via `stream_name`:

  ```elixir
  # Fill infeed:
  EXLA.FFI.Stream.push_infeed(:my_stream, value)

  # Later, consume outfeed:
  {:ok, result} = EXLA.FFI.Stream.pop_outfeed(:my_stream)
  ```

---

## 2. Encoding PID/tag as tensors ✅ **DONE**

### 2.1. General approach ✅ **DONE**

- We want to pass BEAM terms (at least the stream PID; optionally also a tag/name) down into the XLA computation as **tensors**.
- Encoding strategy:
  - On Elixir side: use `:erlang.term_to_binary(term)`. ✅ **DONE**
  - Represent the resulting binary in a **fixed-size `u8` vector tensor**. ✅ **DONE**
  - On the NIF side: reconstruct the term using `enif_binary_to_term`. ✅ **DONE**

### 2.2. Tensor layout ✅ **DONE**

- Instead of using a hardcoded constant, the tensor size can be **computed** from the actual term encoding we plan to use:
  - For the **PID tensor**, use `byte_size(:erlang.term_to_binary(self()))` as the baseline and call this `pid_size`. ✅ **DONE**
  - For the **nif_call tag tensor**, use `byte_size(:erlang.term_to_binary({make_ref(), self()}))` as the baseline and call this `tag_size`. ✅ **DONE** (simplified: tags not used in current implementation)

- We represent terms as `Nx.Tensor` of shape `{pid_size}` or `{tag_size}` and type `:u8`, where those sizes are derived once per VM (or application startup) from the above reference encodings. ✅ **DONE**
- Layout:
  - `vec[0..size-1]` – bytes of the binary. There is no explicit length stored in the tensor; the length is fully determined by the shape. ✅ **DONE**

### 2.3. Helper module ✅ **DONE**

- Add a helper module to create these tensors:
- **Implementation**: `exla/lib/exla/ffi/term_tensor.ex`

  ```elixir
  defmodule EXLA.FFI.TermTensor do
    @pid_size byte_size(:erlang.term_to_binary(self()))
    @tag_size byte_size(:erlang.term_to_binary({make_ref(), self()}))

    @type t :: Nx.Tensor.t()

    def pid_to_tensor(pid) when is_pid(pid) do
      bin = :erlang.term_to_binary(pid)
      assert_size!(bin, @pid_size, :pid)
      Nx.from_binary(bin, :u8) |> Nx.reshape({@pid_size})
    end

    def tag_to_tensor(tag) do
      bin = :erlang.term_to_binary(tag)
      assert_size!(bin, @tag_size, :tag)
      Nx.from_binary(bin, :u8) |> Nx.reshape({@tag_size})
    end

    defp assert_size!(bin, expected, kind) do
      size = byte_size(bin)

      if size != expected do
        raise ArgumentError,
              "unexpected term_to_binary size for #{kind}: got #{size}, expected #{expected}"
      end
    end
  end
  ```

- EXLA will use `EXLA.FFI.TermTensor.pid_to_tensor/1` (and possibly `tag_to_tensor/1`) when constructing argument tensors for custom calls.

### 2.4. NIF-side decoding ✅ **DONE**

- For a `u8[size]` buffer representing a term, the length is known from the shape metadata, so we simply:
  1. Copy all `size` bytes into an `ErlNifBinary`. ⚠️ **OPTIMIZED** (reads directly without memcpy)
  2. Call `enif_binary_to_term(env, bin.data, bin.size, &term)`. ✅ **DONE**

- Example flows:
  - **PID term**: ✅ **DONE**
    - Validate `enif_is_pid(env, term)`.
    - Use `enif_get_local_pid` (or equivalent) to obtain `ErlNifPid`.
  - **Tag/name term**: ⏳ **DEFERRED** (simplified for initial implementation)
    - May be an atom or tuple; decode and use as needed in NIF logic.

---

## 3. XLA FFI custom calls

### 3.1. New custom calls

Define two XLA FFI handlers (C/C++ or Rust) and corresponding HLO custom calls:

- `"exla_beam_outfeed"` – replaces `outfeed`: ✅ **DONE**
  - **Implementation**: `exla/c_src/exla/custom_calls/beam_outfeed.cc`
  - Inputs:
    - Stream PID tensor (`u8[pid_size]`). ✅ **DONE**
    - Payload tensor(s) to send. ✅ **DONE**
  - Outputs:
    - A success flag (for example, a scalar `u8` or `s32` indicating success/failure). ✅ **DONE** (scalar u8: 1=success, 0=failure)
  - Behaviour:
    - Decode the PID from the tensor. ✅ **DONE**
    - Encode payload tensor(s) into an Erlang term using existing EXLA encoding logic. ✅ **DONE** (sends as binary)
    - Use `enif_send` to send a message to the stream process, e.g.: ✅ **DONE**
      - `{exla_outfeed, PayloadTerm}`.
    - Set the success flag accordingly. ✅ **DONE**

- `"exla_beam_infeed"` – replaces `infeed`: ⚠️ **PARTIAL**
  - **Implementation**: `exla/c_src/exla/custom_calls/beam_infeed.cc` (stub, returns kUnimplemented)
  - Inputs:
    - Stream PID tensor (`u8[pid_size]`). ✅ **DONE**
    - Current tag tensor (`u8[tag_size]`). ✅ **DONE** (parsing implemented but not used yet)
  - Outputs:
    - Updated tag tensor (`u8[tag_size]`) to be threaded through subsequent infeed calls. ⏳ **TODO**
    - Payload tensor(s) of some pre-agreed shapes/types. ⏳ **TODO**
  - Behaviour:
    - Decode PID and tag from tensors. ✅ **DONE**
    - Use `nif_call` to invoke a BEAM function that, given the current tag, returns: ⏳ **TODO** (nif_call integration deferred)
      - A new tag term (to be re-encoded into the updated tag tensor), and
      - The next infeed payload (to be decoded into the output tensors).
    - Decode the returned payload term into raw data and write it into XLA output buffers. ⏳ **TODO**

### 3.2. Interaction with the stream process

- Outfeed path (`enif_send`): ✅ **DONE**
  - NIF handler sends messages to the `EXLA.FFI.Stream` process, which appends them to the `outfeed` queue. ✅ **DONE**
  - Example message shape: ✅ **DONE**

    ```elixir
    {:exla_outfeed, payload_term}
    ```

  - The `Stream` process can: ✅ **DONE**
    - Store the term directly in the queue, or
    - Convert to a more convenient representation.

- Infeed path (`nif_call`): ⏳ **TODO**
  - NIF handler calls into a BEAM function (e.g. `EXLA.FFI.Stream.next_infeed(pid)`) that: ✅ **DONE** (GenServer handler exists)
    - Given the stream PID, returns `{:ok, payload_term}`. ✅ **DONE** (simplified: no tag threading)
  - For the first iteration, behaviour on **empty infeed** can be: ✅ **DONE**
    - Block until a value is available (with a timeout). ✅ **DONE** (GenServer.call with timeout)
    - Or fail with a clear runtime error. ✅ **DONE** (returns `{:error, :empty}`)

- Error cases to handle explicitly:
  - PID decode failure or dead process. ✅ **DONE** (returns success flag 0)
  - `nif_call` timeout or error. ⏳ **TODO** (not yet integrated)
  - Mismatched shapes/types between expected templates and received payload. ⏳ **TODO**

---

### 4.1. New EXLA.MLIR.Value helpers ✅ **DONE**

- Add functions to construct the new custom call ops:
- **Implementation**: `exla/lib/exla/mlir/value.ex`

  ```elixir
  def beam_outfeed(%Value{function: func}, pid_tensor, payload) do
    # Build custom_call "exla_beam_outfeed"
    # Returns scalar u8 success flag
  end

  def beam_infeed(%Value{function: func}, pid_tensor, tag_tensor, payload_typespec, tag_typespec) do
    # Build custom_call "exla_beam_infeed"
    # Returns {payload, new_tag}
  end
  ```

- These: ✅ **DONE**
  - Construct the appropriate HLO custom call with:
    - Inputs: PID/tag tensors, payload or template. ✅ **DONE**
    - Outputs: payload for infeed; a success flag for outfeed. ✅ **DONE**
  - Use XLA FFI API version 4. ✅ **DONE**

### 4.2. Integration into EXLA.Defn ⏳ **TODO**

- In `EXLA.Defn`, introduce new ops that map to `beam_infeed` and `beam_outfeed`.
- These **do not** use the existing `:elixir_call` op; we keep `to_operator(:elixir_call, ...)` raising for now.
- Layout:
  - When lowering a defn that wants to interact with the stream:
    - Insert a `beam_outfeed` custom call wherever we previously would outfeed.
    - Insert a `beam_infeed` custom call wherever we previously would infeed.
  - Handle `:stream_name` compiler option to inject PID tensor.

> Note: This file doesnt prescribe the exact Elixir API for defn yet (e.g. `Nx.custom_infeed/2`); see next section.

---

## 5. Error handling and determinism

### 5.1. Determinism ✅ **DONE**

- Outfeed custom calls are scheduled as part of the XLA graph; for a fixed input and stream contents, they will occur in a deterministic order. ✅ **DONE**
- Infeed custom calls should also be deterministic **given the queue contents**: ✅ **DONE**
  - If the same sequence of values is pushed to `infeed` before each run, the same sequence is consumed.

### 5.2. Backpressure and blocking

- Outfeed: ✅ **DONE**
  - `enif_send` is asynchronous; the BEAM stream process mailbox may grow. ✅ **DONE**
  - For the initial experiment, accept this behaviour and document that heavy outfeed use may need careful monitoring. ✅ **DONE**

- Infeed: ⏳ **TODO** (nif_call not yet integrated)
  - By default, `nif_call` should **block** until a value is available or a timeout is reached.
  - On timeout or error, we raise an EXLA runtime error with a clear message.

### 5.3. Explicit error cases

- PID tensor decode failure or invalid term type. ✅ **DONE** (returns success flag 0)
- Stream process dead (`enif_send` failure or `Process.alive?` false in BEAM helper). ✅ **DONE** (enif_send detects dead process)
- Infeed queue empty with no new values within timeout. ✅ **DONE** (GenServer returns `{:error, :empty}`)
- Decoding mismatch between BEAM payload term and expected tensor shapes/types. ⏳ **TODO**

---

## 6. Testing plan

### 6.1. BEAM-only tests ✅ **DONE**

- Test `EXLA.FFI.Stream` behaviour in isolation: ✅ **DONE**
  - `start_link(name: :my_stream)` registers the process. ✅ **DONE**
  - `push_infeed/2` and `pop_outfeed/2` queue semantics. ✅ **DONE**
  - **Implementation**: `exla/test/exla/ffi/stream_test.exs`
- Test `EXLA.FFI.TermTensor`: ✅ **DONE**
  - `pid_to_tensor` / `tag_to_tensor` round-trip through `enif_binary_to_term` in a small NIF or stub. ✅ **DONE**
  - **Implementation**: `exla/test/exla/ffi/term_tensor_test.exs`

### 6.2. EXLA integration tests ⏳ **TODO**

- Under `exla/test/exla/defn/` (or similar):

1. **Outfeed test** ⏳ **TODO**

   - Set compiler options with `stream_name: :my_stream`.
   - Start `EXLA.FFI.Stream` with that name.
   - Defn program that, via EXLA's custom call helpers, sends `x` to the stream (outfeed) and returns `x`.
   - Run via `EXLA.jit_apply`.
   - Assert:
     - The returned tensor is as expected.
     - `EXLA.FFI.Stream.pop_outfeed(:my_stream)` returns the expected payload.

2. **Infeed test** ⏳ **TODO**

   - Pre-populate infeed queue with known values.
   - Defn program that, via EXLA's custom call helpers, pulls from the infeed using a template tensor.
   - Assert the returned tensor matches the pre-populated values.

3. **Round-trip test** ⏳ **TODO**

   - Program pulls from infeed, transforms the value, and sends via outfeed.
   - Check both that infeed consumption is correct and outfeed message matches expected transformation.

4. **Error tests** ⏳ **TODO**

   - Use a `stream_name` with no running process.
   - Kill the stream process mid-execution.
   - Use mismatched payload shapes.

---

## 8. Out of scope for this experiment ✅ **CONFIRMED**

- Implementing general `:elixir_call` lowering in EXLA: ✅ **CONFIRMED**
  - `EXLA.Defn` will continue to raise on `:elixir_call` operators:
    - `Nx.elixir_call/3` remains supported only via `Nx.Defn.Evaluator` or backends that execute it directly.
- Advanced scheduling / backpressure control for infeed/outfeed streams. ✅ **CONFIRMED**
- Multi-node or distributed stream coordination. ✅ **CONFIRMED**

These can be revisited once the narrow infeed/outfeed custom call experiment is stable and useful.

---

## Implementation Status Summary

### ✅ Completed
- Stream process GenServer (`EXLA.FFI.Stream`)
- Term tensor encoding/decoding (`EXLA.FFI.TermTensor`)
- C++ outfeed custom call with `enif_send` (`beam_outfeed.cc`)
- C++ infeed custom call stub (`beam_infeed.cc` - compiles but returns kUnimplemented)
- MLIR Value helpers for both custom calls
- Basic unit tests for Stream and TermTensor modules
- RAII pattern using `UniqueNifEnv` for automatic resource cleanup
- Modern C++ patterns (std::pair returns, structured bindings)

### ⏳ Pending
- EXLA.Defn integration (handle `:stream_name` compiler option)
- Complete `nif_call` integration for beam_infeed
- Runner process setup for nif_call
- End-to-end integration tests
- Payload shape/type validation

### 🎯 Next Steps
1. Integrate `:stream_name` compiler option into EXLA.Defn
2. Add end-to-end tests for outfeed (can test now since it's complete)
3. Complete nif_call integration for infeed
4. Add comprehensive error handling and validation
