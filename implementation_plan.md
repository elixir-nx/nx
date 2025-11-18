# EXLA XLA FFI Custom Call Experiment – Implementation Plan

## Goals

- Add **experimental** support for XLA FFI custom calls in EXLA.
- Replace the current `stablehlo.token` + `infeed`/`outfeed` idea with:
  - A **custom call for outfeed** that uses `enif_send` to send tensors to a BEAM process.
  - A **custom call for infeed** that uses `nif_call` to fetch tensors from a BEAM process.
- Manage infeed/outfeed via a dedicated **stream BEAM process**, identified by a `:stream_name` compiler option.
- Pass the stream process identity (PID and optionally a tag/name) into the computation as **tensor arguments**, using `:erlang.term_to_binary/1` and `enif_binary_to_term/3`.
- Do **not** implement general `:elixir_call` in EXLA yet; this is a focused experiment around infeed/outfeed semantics.

---

## 1. Stream process and compiler options

### 1.1. Compiler option

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

### 1.2. Stream process module

- Add an Elixir module (name TBD, e.g. `EXLA.FFI.Stream`) responsible for managing infeed/outfeed queues.

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

## 2. Encoding PID/tag as tensors

### 2.1. General approach

- We want to pass BEAM terms (at least the stream PID; optionally also a tag/name) down into the XLA computation as **tensors**.
- Encoding strategy:
  - On Elixir side: use `:erlang.term_to_binary(term)`.
  - Represent the resulting binary in a **fixed-size `u8` vector tensor**.
  - On the NIF side: reconstruct the term using `enif_binary_to_term`.

### 2.2. Tensor layout

- Instead of using a hardcoded constant, the tensor size can be **computed** from the actual term encoding we plan to use:
  - For the **PID tensor**, use `byte_size(:erlang.term_to_binary(self()))` as the baseline and call this `pid_size`.
  - For the **nif_call tag tensor**, use `byte_size(:erlang.term_to_binary({make_ref(), self()}))` as the baseline and call this `tag_size`.

- We represent terms as `Nx.Tensor` of shape `{pid_size}` or `{tag_size}` and type `:u8`, where those sizes are derived once per VM (or application startup) from the above reference encodings.
- Layout:
  - `vec[0..size-1]` – bytes of the binary. There is no explicit length stored in the tensor; the length is fully determined by the shape.

### 2.3. Helper module

- Add a helper module to create these tensors:

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

### 2.4. NIF-side decoding

- For a `u8[size]` buffer representing a term, the length is known from the shape metadata, so we simply:
  1. Copy all `size` bytes into an `ErlNifBinary`.
  2. Call `enif_binary_to_term(env, bin.data, bin.size, &term)`.

- Example flows:
  - **PID term**:
    - Validate `enif_is_pid(env, term)`.
    - Use `enif_get_local_pid` (or equivalent) to obtain `ErlNifPid`.
  - **Tag/name term**:
    - May be an atom or tuple; decode and use as needed in NIF logic.

---

## 3. XLA FFI custom calls

### 3.1. New custom calls

Define two XLA FFI handlers (C/C++ or Rust) and corresponding HLO custom calls:

- `"exla_beam_outfeed"` – replaces `outfeed`:
  - Inputs:
    - Stream PID tensor (`u8[pid_size]`).
    - Payload tensor(s) to send.
  - Outputs:
    - A success flag (for example, a scalar `u8` or `s32` indicating success/failure).
  - Behaviour:
    - Decode the PID from the tensor.
    - Encode payload tensor(s) into an Erlang term using existing EXLA encoding logic.
    - Use `enif_send` to send a message to the stream process, e.g.:
      - `{exla_outfeed, PayloadTerm}`.
    - Set the success flag accordingly.

- `"exla_beam_infeed"` – replaces `infeed`:
  - Inputs:
    - Stream PID tensor (`u8[pid_size]`).
    - Current tag tensor (`u8[tag_size]`).
  - Outputs:
    - Updated tag tensor (`u8[tag_size]`) to be threaded through subsequent infeed calls.
    - Payload tensor(s) of some pre-agreed shapes/types.
  - Behaviour:
    - Decode PID and tag from tensors.
    - Use `nif_call` to invoke a BEAM function that, given the current tag, returns:
      - A new tag term (to be re-encoded into the updated tag tensor), and
      - The next infeed payload (to be decoded into the output tensors).
    - Decode the returned payload term into raw data and write it into XLA output buffers.

### 3.2. Interaction with the stream process

- Outfeed path (`enif_send`):
  - NIF handler sends messages to the `EXLA.FFI.Stream` process, which appends them to the `outfeed` queue.
  - Example message shape:

    ```elixir
    {:exla_outfeed, payload_term}
    ```

  - The `Stream` process can:
    - Store the term directly in the queue, or
    - Convert to a more convenient representation.

- Infeed path (`nif_call`):
  - NIF handler calls into a BEAM function (e.g. `EXLA.FFI.Stream.next_infeed(tag, pid)`) that:
    - Given the current tag and stream PID, returns `{new_tag, payload_term}`.
  - For the first iteration, behaviour on **empty infeed** can be:
    - Block until a value is available (with a timeout).
    - Or fail with a clear runtime error.

- Error cases to handle explicitly:
  - PID decode failure or dead process.
  - `nif_call` timeout or error.
  - Mismatched shapes/types between expected templates and received payload.

---

### 4.1. New EXLA.Lib helpers

- Add functions to construct the new custom call ops:

  ```elixir
  defmodule EXLA.Lib do
    # Pseudo-signatures
    def beam_outfeed(builder, pid_tensor, payload_tensor, opts \\ []) do
      # Build custom_call "exla_beam_outfeed"
    end

    def beam_infeed(builder, pid_tensor, template_tensor, opts \\ []) do
      # Build custom_call "exla_beam_infeed" with output
      # shape/type taken from template_tensor
    end
  end
  ```

- These should:
  - Construct the appropriate HLO custom call with:
    - Inputs: PID/tag tensors, payload or template.
    - Outputs: payload for infeed; a success flag (or dummy/token) for outfeed.
  - Respect token-based ordering if EXLA already uses tokens to sequence side effects.

### 4.2. Integration into EXLA.Defn

- In `EXLA.Defn`, introduce new ops that map to `beam_infeed` and `beam_outfeed`.
- These **do not** use the existing `:elixir_call` op; we keep `to_operator(:elixir_call, ...)` raising for now.
- Layout:
  - When lowering a defn that wants to interact with the stream:
    - Insert a `beam_outfeed` custom call wherever we previously would outfeed.
    - Insert a `beam_infeed` custom call wherever we previously would infeed.

> Note: This file doesnt prescribe the exact Elixir API for defn yet (e.g. `Nx.custom_infeed/2`); see next section.

---

## 5. Error handling and determinism

### 5.1. Determinism

- Outfeed custom calls are scheduled as part of the XLA graph; for a fixed input and stream contents, they will occur in a deterministic order.
- Infeed custom calls should also be deterministic **given the queue contents**:
  - If the same sequence of values is pushed to `infeed` before each run, the same sequence is consumed.

### 5.2. Backpressure and blocking

- Outfeed:
  - `enif_send` is asynchronous; the BEAM stream process mailbox may grow.
  - For the initial experiment, accept this behaviour and document that heavy outfeed use may need careful monitoring.

- Infeed:
  - By default, `nif_call` should **block** until a value is available or a timeout is reached.
  - On timeout or error, we raise an EXLA runtime error with a clear message.

### 5.3. Explicit error cases

- PID tensor decode failure or invalid term type.
- Stream process dead (`enif_send` failure or `Process.alive?` false in BEAM helper).
- Infeed queue empty with no new values within timeout.
- Decoding mismatch between BEAM payload term and expected tensor shapes/types.

---

## 6. Testing plan

### 6.1. BEAM-only tests

- Test `EXLA.FFI.Stream` behaviour in isolation:
  - `start_link(name: :my_stream)` registers the process.
  - `push_infeed/2` and `pop_outfeed/2` queue semantics.
- Test `EXLA.FFI.TermTensor`:
  - `pid_to_tensor` / `tag_to_tensor` round-trip through `enif_binary_to_term` in a small NIF or stub.

### 6.2. EXLA integration tests

- Under `exla/test/exla/defn/` (or similar):

1. **Outfeed test**

- Set compiler options with `stream_name: :my_stream`.
- Start `EXLA.FFI.Stream` with that name.
- Defn program that, via EXLA's custom call helpers, sends `x` to the stream (outfeed) and returns `x`.
- Run via `EXLA.jit_apply`.
- Assert:
  - The returned tensor is as expected.
  - `EXLA.FFI.Stream.pop_outfeed(:my_stream)` returns the expected payload.

2. **Infeed test**

- Pre-populate infeed queue with known values.
- Defn program that, via EXLA's custom call helpers, pulls from the infeed using a template tensor.
- Assert the returned tensor matches the pre-populated values.

3. **Round-trip test**

- Program pulls from infeed, transforms the value, and sends via outfeed.
- Check both that infeed consumption is correct and outfeed message matches expected transformation.

4. **Error tests**

- Use a `stream_name` with no running process.
- Kill the stream process mid-execution.
- Use mismatched payload shapes.

---

## 8. Out of scope for this experiment

- Implementing general `:elixir_call` lowering in EXLA:
  - `EXLA.Defn` will continue to raise on `:elixir_call` operators:
    - `Nx.elixir_call/3` remains supported only via `Nx.Defn.Evaluator` or backends that execute it directly.
- Advanced scheduling / backpressure control for infeed/outfeed streams.
- Multi-node or distributed stream coordination.

These can be revisited once the narrow infeed/outfeed custom call experiment is stable and useful.
