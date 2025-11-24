## Design: `Nx.elixir_call/3` and EXLA Integration

### 1. Overview

This document describes a two-phase plan to implement safe, efficient support for calling arbitrary Elixir code from `defn` via `Nx.elixir_call/3`, with a focus on EXLA.

- **Phase 1**: CPU-only implementation using EXLA + XLA host `CustomCall`, with a safe bridge to Elixir (no `nif_call`-style reentry into BEAM).
- **Phase 2**: Graph segmentation in `Nx.Defn.Graph` so the compiler can:
  - Treat `elixir_call` as a boundary and split the computation into stages.
  - Enable cross-device execution (CPU/GPU) while preserving a single user API.
  - Eventually optimize some callbacks to be compiled away or lowered differently (e.g. pure functions expressible in Nx).

This work extends [PR #1627, “feat: Nx.elixir_call/3”](https://github.com/elixir-nx/nx/pull/1627), which currently implements `Nx.elixir_call/3` only in `Nx.Defn.Evaluator`.

---

### 2. Goals and Non-goals

- **Goals**
  - **G1**: Provide a **public API** (`Nx.elixir_call/3`) that allows calling user-provided Elixir code inside `defn`.
  - **G2**: Implement a **safe** EXLA backend for `elixir_call` on **CPU** using XLA host `CustomCall`.
  - **G3**: Ensure callbacks have **statically known shapes/dtypes** to keep compilation and gradients well-defined.
  - **G4**: Provide a **unified intermediate representation** in `Nx.Defn.Graph` so future backends (EXLA GPU, other compilers) can share the same abstraction.
  - **G5**: In Phase 2, support **graph segmentation** around `elixir_call` so that:
    - We can mix device computation (CPU/GPU) with Elixir callbacks.
    - The compiler can decide to either split or compile callbacks, depending on their structure.

- **Non-goals (for now)**
  - **NG1**: No direct, device-side callbacks for GPU/TPU in Phase 1 (no infeed/outfeed complexity yet).
  - **NG2**: No guarantees about **side-effect isolation** of callbacks (user is responsible), beyond not violating BEAM safety.
  - **NG3**: No attempt to automatically infer output shapes/dtypes of callbacks at runtime; shapes must be known at `defn`/compile time.

---

### 3. Terminology

- **`elixir_call` node**: The internal IR node representing a call to arbitrary Elixir code (backed by `Nx.elixir_call/3`).
- **Callback ID**: A stable identifier (string or integer) used to look up the Elixir function and output spec at compile/run time.
- **Output spec**: Shape and type description for all outputs of a callback.
- **Bridge thread**: A native (C/C++) thread that acts as a mediator between XLA/EXLA and BEAM, using message-passing only (no direct BEAM calls from arbitrary XLA threads).

---

### 4. Phase 1: CPU-only EXLA Backend (Host `CustomCall`)

#### 4.1 Public API: `Nx.elixir_call/3`

- **Goal**: Reuse and finalize the API introduced in [nx#1627](https://github.com/elixir-nx/nx/pull/1627).

- **Shape** (subject to minor refinement):

  - `Nx.elixir_call(args, fun_or_mfa, opts \\ [])`

- **Key options / metadata**:
  - **`id` or `name`**: A stable callback identifier (string or integer).
  - **`output_template`** (or equivalent): A value (or list/tuple of values) that describes the **shapes and dtypes** of the callback’s outputs:
    - Can be Nx tensors or a structured spec, but must be statically known at `defn` compile time.
  - Potentially an **`impure`** flag (or similar) in the future to guide compiler optimizations.

- **Constraints**:
  - `fun_or_mfa` is not executed at `defn` compile time (except possibly in the evaluator backend).
  - Output shape/type comes from `output_template`, not from running the function.

#### 4.2 Nx IR: Representing `elixir_call`

- **Extend** `Nx.Defn.Expr` / `Nx.Defn.Graph` to carry `elixir_call` nodes explicitly.

- Proposed internal form:

  - `{:elixir_call, meta, args}`

  Where:
  - **`meta`** includes:
    - `callback_id` (string/int).
    - `fun_or_mfa` or internal reference (for evaluator backend and dispatcher).
    - `output_spec` (shapes + dtypes).
    - Any flags required for compilation/grad.
  - **`args`**: list of argument expressions.

- **Requirements**:
  - Shape inference for `elixir_call` uses `output_spec`.
  - Optimizer must **not** fuse or eliminate `elixir_call`; it is a logical boundary and may be effectful.
  - The evaluator backend (as in nx#1627) already knows how to interpret it.

#### 4.3 EXLA Lowering: From `elixir_call` to HLO/StableHLO

- In the EXLA backend (Elixir side):

  - When encountering an `elixir_call` node while building HLO/StableHLO:

    - Lower `args` to HLO values.
    - Construct a `CustomCall` operation with:
      - **Operands**: those input HLO values.
      - **Result types**: from `output_spec`.
      - **Call target name**: e.g. `"exla_elixir_callback"`.
      - **Attributes**:
        - `callback_id` (string/int).
        - Optionally an encoded `output_spec` (if needed on the native side).

- **CPU-only restriction** (Phase 1):
  - If the active EXLA client is **CPU**, allow this lowering.
  - If the client is GPU (or other non-CPU), raise a **clear error**:
    - e.g. “`Nx.elixir_call/3` is currently only supported for EXLA CPU; please run on CPU or wait for Phase 2 segmentation support.”

#### 4.4 Native EXLA: Callback Registry and Bridge

- **Callback registry (Elixir → native)**:
  - At the time of building an EXLA executable, collect all callbacks:

    - Map: `callback_id → {fun_or_mfa, output_spec}`.

  - Pass this mapping down to the native side, associated with the executable or run context.

- **Native data structures** (C/C++ side):

  - `struct CallbackRequest { RunRef run_ref; CallbackId callback_id; std::vector<Tensor> args; ReplyTag reply_tag; std::promise<Result> promise; };`

  - `struct CallbackResult { ReplyTag reply_tag; std::vector<Tensor> outputs; Error error; };`

  - A **thread-safe queue** for `CallbackRequest`s.

  - A **map** `reply_tag → std::promise<Result>` guarded by a mutex.

- **Bridge thread**:

  - Started when the EXLA NIF is initialized (or when the first callback-capable executable is created).

  - Main loop:
    1. Pop `CallbackRequest` from the queue.
    2. Serialize `args` into a compact binary representation (shape metadata + flat data).
    3. Use `enif_send` to send a message to a **dedicated Elixir dispatcher process**:
       - Message format (conceptual):
         `{:exla_elixir_call, run_ref, callback_id, args_bin, reply_tag}`.
    4. Wait on the `std::promise`/`std::future` associated with `reply_tag` until `CallbackResult` is set:
       - **Important**: This wait uses only native primitives (no BEAM APIs, no `nif_call`), so it is safe w.r.t. BEAM scheduling.
    5. On success/failure, the handler (see next section) is unblocked.

#### 4.5 XLA Host `CustomCall` Handler (CPU Client)

- **Registration**:

  - For the EXLA CPU client, register a host call target with XLA:

    - Name: `"exla_elixir_callback"`.

- **Handler logic**:

  1. Extract:
     - `callback_id` from `CustomCall` attributes.
     - Operand buffers (inputs).
     - Output buffers and their shapes/dtypes.
  2. Convert operand buffers into host tensors and build a `CallbackRequest`:
     - Assign a fresh `reply_tag`.
     - Create a `std::promise<Result>` and `std::future<Result>`.
     - Insert `reply_tag → promise` into the map.
  3. Enqueue the `CallbackRequest` onto the native request queue.
  4. Block on the `future` until a result arrives (native wait).
  5. Once the `CallbackResult` is available:
     - On success:
       - Write returned tensor data into XLA’s output buffers.
       - Return `OK` to XLA.
     - On error or timeout:
       - Return an error `Status` so the XLA run fails with a descriptive error.

#### 4.6 Elixir Dispatcher Process

- Implement a **GenServer** in Nx/EXLA that acts as the BEAM-side dispatcher for callbacks.

- Responsibilities:

  - Maintain:
    - `callbacks: %{ {run_ref, callback_id} => {fun_or_mfa, output_spec} }`.

  - Handle messages from the bridge thread:

    ```elixir
    def handle_info({:exla_elixir_call, run_ref, callback_id, args_bin, reply_tag}, state) do
      {args, arg_specs} = deserialize_tensors(args_bin)
      {fun_or_mfa, output_spec} =
        Map.fetch!(state.callbacks, {run_ref, callback_id})

      # Execute user code (possibly in a Task for isolation)
      result =
        try do
          call_user_fun(fun_or_mfa, args)
        rescue
          exception -> {:error, {:exception, exception, __STACKTRACE__}}
        catch
          kind, reason -> {:error, {kind, reason}}
        end

      reply_payload =
        encode_result(result, output_spec) # either {:ok, tensors_bin} or {:error, reason}

      # One NIF call to signal back to native side (bridge thread sees this)
      send_reply_to_nif(reply_tag, reply_payload)

      {:noreply, state}
    end
    ```

  - Ensure:
    - Result shapes/dtypes match `output_spec`; otherwise return a structured error.
    - Optional: enforce configurable timeouts per callback and abort the run on timeout.

- **API considerations**:

  - A worker or supervisor module (e.g. `EXLA.CallbackServer`) could manage:
    - Registration of callbacks per `run_ref`.
    - Cleanup after run completion.

when registering the callback, there should be a fun/capture -> integer mapping (maybe use :counters for generating these integers) and the function should be registered with this id. The id should be returned so that the compiler can use it. This turns the callback server into the source of truth and the generator of ids

#### 4.7 Error Handling and Validation

- **Compile-time checks**:

  - Verify that:
    - `output_template` can be converted into a valid `output_spec`.
    - Grad rules (where applicable) can be defined; if not, error clearly or fallback.

- **Runtime checks**:

  - After Elixir callback returns:
    - Validate result shape/dtype vs `output_spec`.
    - On mismatch, generate a descriptive error and fail the XLA run.

- **Timeouts**:

  - Optional but recommended:
    - Per-callback timeout at the dispatcher level.
    - If timeout expires, reply with error; native side then aborts the run.

- **Safety**:

  - No calls from arbitrary XLA threads into BEAM functions.
  - All BEAM interaction uses `enif_send` from the bridge thread or explicit NIF calls from Elixir processes.

---

### 5. Phase 2: Graph Segmentation and Cross-Device Support

After Phase 1 is solid on CPU, we extend support to all EXLA devices (CPU/GPU) via **segmentation** in `Nx.Defn.Graph`. This aligns with [the discussion on nx#1627](https://github.com/elixir-nx/nx/pull/1627), where `elixir_call` and other “optional callback” mechanisms share a unified specification, and the compiler decides whether to split or to compile.

#### 5.1 Treat `elixir_call` as a Stage Boundary

- In `Nx.Defn.Graph`, treat each `elixir_call` as a **potential cut point**:

  - Find maximal subgraphs that:
    - Contain no `elixir_call`.
    - Are otherwise pure Nx computations.

- Build a sequence:

  - `stage_0` → `elixir_call_0` → `stage_1` → `elixir_call_1` → … → `stage_n`.

- Each `stage_i` will be compiled separately for a target device (CPU or GPU).

#### 5.2 Stage Compilation

- For each pure stage:

  - Infer shapes and types as usual.
  - Choose a device (matching the EXLA client or using more advanced heuristics later).
  - Compile to an EXLA executable.

- For each `elixir_call` between stages:

  - Reuse the **Phase 1 dispatcher + bridge**:
    - Inputs: outputs of the previous stage (converted to host tensors).
    - Outputs: inputs to the next stage (converted back and transferred as needed).

#### 5.3 Orchestration Runtime

- Implement an orchestrator (in Nx/EXLA) that performs:

  1. Run `stage_0` on its device → get outputs.
  2. Transfer these outputs to host (if needed).
  3. Invoke the Elixir callback via the dispatcher → get callback outputs.
  4. Transfer callback outputs to the device for `stage_1` (if needed).
  5. Repeat until all stages and callbacks are executed.

- This orchestration:

  - Provides **consistent semantics** across CPU and GPU.
  - Keeps the user API (`Nx.elixir_call/3`) unchanged.
  - Allows future optimizations where:
    - Some callbacks are compiled away (if expressible in pure Nx).
    - Some backends choose device-specific mechanisms (e.g. XLA GPU host-callbacks or infeed/outfeed) internally.

#### 5.4 Compiler Decisions (Future Work)

- Over time, the compiler can classify callbacks:

  - **Pure, shape-stable callbacks definable in Nx**:
    - Potentially inline/compile them, removing the runtime callback.

  - **Genuinely dynamic callbacks**:
    - Keep them as segmentation boundaries.

- This addresses the concern raised in [nx#1627](https://github.com/elixir-nx/nx/pull/1627) about having a **unified specification** for callbacks while allowing the compiler to choose between splitting and compiling.

---

### 6. Open Questions / Next Steps

- **Naming and API**:
  - Finalize `Nx.elixir_call/3` naming and argument order.
  - Decide whether to expose more advanced options (timeouts, impurity markers, etc.) in `opts`.

- **Gradients**:
  - For Phase 1, gradients may be:
    - Not supported for arbitrary callbacks (raise on use), or
    - Supported only when the callback is expressible in Nx and compiled away (future optimization).

- **Concurrency model**:
  - Decide how many bridge threads to run.
  - Understand the interaction with multiple concurrent EXLA runs and multiple callback-heavy computations.

- **Device-specific optimizations** (beyond segmentation):
  - Investigate XLA’s GPU host-callback support and whether to implement a more tightly integrated path for GPU (possibly involving infeed/outfeed under the hood) once segmentation version is stable.

---

### 7. Implementation Order Checklist

1. **Land / refine `Nx.elixir_call/3` API and IR node** (based on [nx#1627](https://github.com/elixir-nx/nx/pull/1627)).
2. **Add EXLA lowering** for CPU:
   - Map `elixir_call` → HLO/StableHLO `CustomCall` with target `"exla_elixir_callback"`.
3. **Implement native callback registry + bridge thread** in EXLA NIF.
4. **Register CPU host `CustomCall` handler** (`"exla_elixir_callback"`) and wire it to the bridge.
5. **Implement Elixir dispatcher process** for callbacks + error handling + sanity checks.
6. **Add tests** for CPU:
   - Simple callbacks.
   - Multiple callbacks in a single `defn`.
   - Error cases (shape mismatch, thrown exceptions).
7. **Introduce segmentation in `Nx.Defn.Graph`**:
   - Identify stages between `elixir_call` nodes.
   - Compile/orchestrate stages for CPU/GPU.
8. **Extend EXLA to allow callbacks under segmentation** when using GPU clients.
9. Iterate on compiler-side heuristics to decide when callbacks can be compiled away vs split.

### 8. Phase 1 – Intended vs Implemented (Status Summary)

This section records where the **current implementation** matches or intentionally diverges from the original Phase 1 plan, so future work can see what is done vs still open.

#### 8.1 Public API (`Nx.elixir_call/3`)

- **Intended (this doc, §4.1)**:
  - Shape: `Nx.elixir_call(args, fun_or_mfa, opts \\ [])`.
  - Explicit `output_template` / `output_spec` passed via `opts`.
- **Implemented (Nx 0.10 + EXLA 0.10)**:
  - Shape: `Nx.elixir_call(output_template, args, fun)`.
    - `output_template` is the **first argument**, not an option.
    - `args` is a list of runtime arguments (tensors + static values).
    - `fun` is a plain Elixir function; we don’t support MFA in Phase 1.
  - `Nx.Defn.Expr.elixir_call/3`:
    - For tensor output: stores `:elixir_call` node with args `[in_args, fun, out_template]`, where `out_template = Nx.to_template(output)`.
    - For tuple output: builds an internal tuple-shaped template (`tuple_out/1`) plus a **user template** (also passed as `out_template`) so EXLA sees the per-element shapes/dtypes.
  - Rationale: keeping `output_template` as a *value argument* made the IR and EXLA lowering simpler and closer to existing `defn` conventions.

#### 8.2 IR Representation (`Nx.Defn.Expr` / `Nx.Defn.Graph`)

- **Intended (§4.2)**:
  - `{:elixir_call, meta, args}` with `meta` carrying:
    - `callback_id`, `fun_or_mfa`, `output_spec`, etc.
- **Implemented**:
  - `Expr` op is still `:elixir_call`, but:
    - We **do not** store `callback_id` in `meta`; it is managed solely by EXLA.
    - Arguments are `[in_args, fun, out_template]`.
    - Shape/type inference for the node uses `out_template` via the existing template machinery.
  - `Nx.Defn.Tree.apply_args/4` and `Nx.Defn.Evaluator.compute_cache/4` / `eval_apply/4` were updated to be aware of the `out_template` third arg but largely **ignore it** at runtime (it is for compilation only).

#### 8.3 EXLA Lowering to StableHLO `CustomCall`

- **Intended (§4.3)**:
  - `CustomCall("exla_elixir_callback")` with:
    - Result types from `output_spec`.
    - Attributes:
      - `callback_id` (string/int).
      - Possibly encoded `output_spec`.
- **Implemented**:
  - We lower to a `stablehlo.custom_call` with:
    - `call_target_name = "exla_elixir_callback"`.
    - `api_version = 4` (typed FFI).
    - **No `backend_config` or dictionary attributes** for callback id.
  - Instead of encoding `callback_id` as an attribute, we:
    - Append a scalar S64 operand at the **end of the operand list** carrying `callback_id`.
    - Register a typed FFI handler that:
      - Interprets the last operand as the callback id.
      - Treats the remaining operands as regular tensor arguments.
  - Result types are derived from the `out_template`:
    - `container_to_typespecs(out_template)` produces one `EXLA.Typespec` per tensor (including tuple elements).

#### 8.4 Native Bridge & Callback Registry

- **Intended (§4.4–4.5)**:
  - Per-run mapping `callback_id → {fun_or_mfa, output_spec}`.
  - Bridge using `RunRef`, `CallbackRequest`, `CallbackResult`, per-run state.
- **Implemented**:
  - We use a **global, process-wide `EXLA.CallbackServer`**:
    - Maps `callback_id (integer)` → `{fun, out_template, static_args}`.
    - Reuses ids when the same `{fun, out_template, static_args}` triple is registered again.
  - The C++ bridge maintains:
    - A global `ElixirCallbackBridgeState` with:
      - `dispatcher_pid`, `next_tag`, and a `pending` map from `reply_tag` to a small `ElixirCallbackPending` object (`std::mutex`, `std::condition_variable`, `ElixirCallbackResult`).
  - The **bridge thread** concept is realized as:
    - The host `CustomCall` handler runs on an XLA-controlled thread.
    - It **blocks natively** on a `std::condition_variable` associated with a `reply_tag` until the BEAM side replies via `EXLA.NIF.elixir_callback_reply/2`.
  - There is currently **no per-run `RunRef`**; callbacks are effectively global to the VM.

#### 8.5 Elixir Dispatcher (`EXLA.CallbackServer`)

- **Intended (§4.6)**:
  - Dedicated dispatcher process keyed by `(run_ref, callback_id)`.
  - Messages like `{:exla_elixir_call, run_ref, callback_id, args_bin, reply_tag}`.
- **Implemented**:
  - `EXLA.CallbackServer` is a `GenServer` with:
    - `callbacks :: %{callback_id => {fun, out_template, static_args}}`.
  - Native side sends:
    - `{:exla_elixir_call, callback_id, args_spec, reply_tag}`.
    - `args_spec` is a list of `{bin, {type_atom, bits}, shape_list}` tuples.
  - Dispatcher logic:
    - Decodes `args_spec` into `Nx.Tensor`s (`Nx.from_binary/3` + `Nx.reshape/2`).
    - Appends `static_args` captured at registration time.
    - Executes the callback function with `[tensor_args ++ static_args]`.
    - Validates the result against `out_template`:
      - Tuple size check + per-element shape/dtype/names check.

#### 8.6 Error Handling & Mapping

- **Intended (§4.7)**:
  - Shape/dtype validation vs `output_spec`.
  - Clear errors; possibly mapping to `ArgumentError` or similar.
- **Implemented (Phase 1)**:
  - `EXLA.CallbackServer.ensure_compatible/2`:
    - Returns `{:ok, value}` on success.
    - Returns tagged errors:
      - `{:error, {:shape_mismatch, left, right}}`.
      - `{:error, {:invalid_result, left, right}}` for non-tensors/tuples.
  - `encode_reply/1` maps internal error tuples to **typed error payloads**:
    - Shape mismatch / invalid result:
      - Encoded as `{:error, {:argument_error, message_binary}}` where the message mirrors `Nx.ensure_call_compatible!/2`, e.g.
        - `"expected the elixir_call function to match the given output template ..., got: ..."`
    - Decode failures, invalid args spec, unknown callback id, user exceptions, throws, exits:
      - Encoded as `{:error, {:runtime_error, message_binary}}` with descriptive text (`"Elixir callback raised: ..."`, etc.).
  - Native `DeliverElixirCallbackReply`:
    - For `{:ok, binaries}`, fills result buffers.
    - For `{:error, {kind_atom, message_binary}}`:
      - Uses the **message** as `result.error` string returned to XLA.
  - As a result:
    - From the user’s point of view, `Nx.elixir_call/3` under EXLA now fails with:
      - `RuntimeError` carrying a **descriptive, Nx-style message** (e.g. shape mismatch) instead of the generic `"elixir callback returned error"`.
    - We do **not** currently raise `ArgumentError` directly from EXLA runs; everything surfaces as `RuntimeError` with a rich message, which tests explicitly assert.

#### 8.7 Timeouts & Robustness

- **Intended (§4.7, Timeouts)**:
  - Optional per-callback timeout.
  - Native side aborts run on timeout.
- **Implemented**:
  - **No timeouts yet**:
    - The host `CustomCall` waits indefinitely on `condition_variable` for the reply.
    - If the Elixir dispatcher never replies, the XLA run will hang.
  - This is an explicit **TODO** for future hardening; the design here still stands, but is not yet implemented.

#### 8.8 Phase 2 – Not Implemented Yet

All of §5 (segmentation and cross-device support) remains **design only**:

- `elixir_call` is **not yet used as a segmentation boundary** in `Nx.Defn.Graph`.
- There is no multi-stage orchestration for GPU/TPU plus CPU callbacks.
- Callbacks are only allowed on the **EXLA host (CPU) client**, and we eagerly raise if the client platform is not `:host`.

This section should be updated again once segmentation and GPU support are implemented.
