# Infeed/Outfeed Migration Plan: Removing StableHLO Tokens

**Goal:** Complete removal of `stablehlo.token`, `stablehlo.infeed`, and `stablehlo.outfeed` in favor of custom XLA FFI calls with variadic support.

**Branch:** `pv-feat/infeed-outfeed`

---

## Current State Analysis

### ✅ Already Implemented
- [x] Custom FFI handlers in C++ (`infeed.h/cc`, `outfeed.h/cc`) with variadic support
- [x] NIF call infrastructure for Erlang↔C++ communication (`nif_call.h`, `exla_nif_call.h/cc`)
- [x] Handler process per device (`exla_feed_process_#{device_id}`)
- [x] Session-based tagging system via `EXLA.NifCall.Runner`
- [x] Message-based communication (no XLA native queues)
- [x] Tests for new custom call APIs (various dtypes, variadic)

### 🔄 Still Uses Old API
- [ ] `Typespec.token()` in 4 places (typespec.ex, defn.ex lines 548/1503/1718)
- [ ] Token threading in EXLA.Defn compilation pipeline
- [ ] Old `transfer_to_infeed` / `transfer_from_outfeed` NIFs
- [ ] Old tests using token-based patterns (lines 177-242 in executable_test.exs)

---

## Key Architecture Decisions

### Handler Process Design
- **One registered process per device**: `exla_feed_process_#{device_id}`
- **Maintains**: infeed queue, session tags, outfeed routing
- **Receives**: outfeed messages directly from C++ custom calls
- **Manages**: NIF callback registration for infeed requests

### Session Tags (Not Tokens)
- **Position**: Last argument (not first like tokens)
- **Format**: `{:u, 8}` tensor of 65 bytes
- **Purpose**: Encodes callback function via `EXLA.NifCall.Runner`
- **Benefit**: Enables multiple concurrent sessions per device

### Variadic FFI Calls
- **Infeed**: Uses `RemainingRets` to return multiple tensors
- **Outfeed**: Uses `RemainingArgs` to accept multiple tensors
- **Efficiency**: Single custom call handles all tensors

### PID Threading
- **Explicit**: `outfeed_cpu_custom_call` - PID as last argument
- **Implicit**: `outfeed_main_custom_call` - looks up `exla_feed_process_#{device_id}`

---

## Phase 1: Architecture & Current State Analysis ✅

- [x] Analyze current implementation
- [x] Identify all token usages
- [x] Understand handler process architecture
- [x] Review custom FFI implementations
- [x] Document design decisions

---

## Phase 2: Elixir-Level Token Removal

### Task 2.1: Replace Typespec.token() with Tag Buffer Typespec
**Location:** `exla/lib/exla/typespec.ex`

- [ ] Deprecate or remove `Typespec.token()` function (line 29-31)
- [ ] Create `Typespec.tag_buffer(size \\ 65)` helper
- [ ] Document standard tag size (65 bytes)
- [ ] Update module documentation

### Task 2.2: Update EXLA.Defn Module Token Handling
**Location:** `exla/lib/exla/defn.ex`

**Line 548 (cached_recur_operator with function calls):**
- [ ] Replace `Typespec.token()` with session tag typespec
- [ ] Pass tag as last argument instead of first
- [ ] Update `Value.call` to thread tag through

**Lines 1503-1510 (optional_computation):**
- [ ] Replace token typespec with tag typespec
- [ ] Pass tag as last argument instead of first
- [ ] Update function signatures

**Lines 1718-1732 (conditional/if operations):**
- [ ] Replace token in result typespecs with tag typespec
- [ ] Thread tag through both branches
- [ ] Update result unpacking (tag from last position, not first)

### Task 2.3: Update EXLA.MLIR.Value Token Functions
**Location:** `exla/lib/exla/mlir/value.ex`

**`create_token/2`:**
- [ ] Accept and encode NIF call tag instead of creating dummy constant
- [ ] Serialize tag to binary
- [ ] Create constant from bytes

**`infeed/2` functions:**
- [ ] Ensure all signatures work without tokens
- [ ] Remove any remaining token parameter requirements
- [ ] Verify tag-based infeed is default path

**`outfeed/2` functions:**
- [ ] Remove any token parameter requirements
- [ ] Ensure all paths use custom FFI calls
- [ ] Verify no `stablehlo.outfeed` operations remain

**Cleanup:**
- [ ] Remove `type_token/0` private function (already commented out)
- [ ] Add documentation to `typespec_to_mlir_type/1` for tag handling

---

## Phase 3: Handler Process & PID Threading

### Task 3.1: Ensure Handler Process is Properly Initialized
**Location:** `exla/lib/exla/defn/outfeed.ex`

- [ ] Verify process starts before any infeed/outfeed operations
- [ ] Confirm process maintains infeed queue
- [ ] Verify session tag management
- [ ] Test `:begin_session` message handling

### Task 3.2: Infeed Callback PID Resolution
**Location:** `exla/lib/exla/defn/outfeed.ex` - `infeed_callback/2`

- [ ] Verify callback registration via `EXLA.NifCall.Runner.register/2`
- [ ] Test tag serialization to C++ layer
- [ ] Verify C++ can decode tag and invoke callback
- [ ] Test `:pop_infeed` message flow

### Task 3.3: Outfeed Handler PID Usage
**Location:** `exla/c_src/exla/custom_calls/outfeed.h`

- [ ] Test `outfeed_cpu_custom_call_impl` with explicit PID
- [ ] Test `outfeed_main_custom_call_impl` with registered process
- [ ] Verify process receives binary messages
- [ ] Test multiple tensor outfeeds are properly grouped

### Task 3.4: Update Handler Process Loop for Pure Custom Calls
**Location:** `exla/lib/exla/defn/outfeed.ex` - `loop/7`

- [ ] Verify no calls to `EXLA.Client.from_outfeed`
- [ ] Verify no calls to `EXLA.Client.to_infeed`
- [ ] Test message passing works correctly
- [ ] Test `:pop_infeed` request handling
- [ ] Test tensor accumulation for pending hooks

---

## Phase 4: Remove Old NIF Infrastructure

### Task 4.1: Remove transfer_to_infeed NIF

- [ ] `exla/lib/exla/nif.ex` - Line 65: Remove stub
- [ ] `exla/c_src/exla/exla.cc` - Lines 350-360: Remove implementation
- [ ] `exla/c_src/exla/exla_client.h` - Lines 107-109: Remove declaration
- [ ] `exla/c_src/exla/exla_client.cc` - Lines 426-473: Remove `TransferToInfeed` method

### Task 4.2: Remove transfer_from_outfeed NIF

- [ ] `exla/lib/exla/nif.ex` - Line 66: Remove stub
- [ ] `exla/c_src/exla/exla.cc` - Lines 362-377: Remove implementation
- [ ] `exla/c_src/exla/exla_client.h` - Lines 111-112: Remove declaration
- [ ] `exla/c_src/exla/exla_client.cc` - Lines 475-489: Remove `TransferFromOutfeed` method

### Task 4.3: Update EXLA.Client Module
**Location:** `exla/lib/exla/client.ex`

**Option A: Remove methods**
- [ ] Remove `to_infeed/3` (lines 90-95)
- [ ] Remove `from_outfeed/5` (lines 100-108)

**Option B: Redirect to handler process**
- [ ] Update `to_infeed/3` to send `{:infeed_data, data}` to handler
- [ ] Keep `from_outfeed/5` for backward compat
- [ ] Add deprecation warnings

---

## Phase 5: Test Updates

### Task 5.1: Update Old Infeed/Outfeed Tests
**Location:** `exla/test/exla/executable_test.exs` - Lines 177-242

**Test: "successfully sends to/from device asynchronously" (line 177)**
- [ ] Replace `Value.create_token(b)` with NIF call tag creation
- [ ] Replace `Value.infeed(token, [typespec])` with `Value.infeed_custom(tag, [typespec])`
- [ ] Replace `Value.outfeed(val, token)` with `Value.outfeed([val], builder)`
- [ ] Update `Client.to_infeed` calls
- [ ] Update `from_outfeed` calls

**Test: "successfully sends to/from device asynchronously in a loop" (line 199)**
- [ ] Update while loop to use tag threading instead of token
- [ ] Replace token in condition region with tag
- [ ] Replace token in body region with tag
- [ ] Update infeed/outfeed calls in loop body

### Task 5.2: Ensure New Custom Call Tests Pass
**Location:** `exla/test/exla/executable_test.exs` - Lines 244-430

- [ ] Run "infeed custom call dtypes" tests
- [ ] Run "outfeed custom call" tests
- [ ] Run "variadic infeed/outfeed custom calls" tests
- [ ] Fix any failures

### Task 5.3: Add Integration Tests

- [ ] Test handler process lifecycle
- [ ] Test session tag creation and passing
- [ ] Test infeed callback invocation
- [ ] Test outfeed message reception
- [ ] Test multiple concurrent sessions
- [ ] Test different session tags
- [ ] Test correct data routing

---

## Phase 6: MLIR Compilation Pipeline Updates

### Task 6.1: Remove Token Creation from Function Compilation
**Location:** `exla/lib/exla/defn.ex` - Module compilation flow

- [ ] Remove automatic token prepending to arguments
- [ ] When streaming enabled, append tag argument at end
- [ ] Tag should be `{:u, 8}` tensor of size 65 bytes
- [ ] Update argument indexing throughout

### Task 6.2: Update Conditional (if/else) Compilation
**Location:** `exla/lib/exla/defn.ex` - Lines 1700-1733

- [ ] Thread tag through both branches (if present)
- [ ] Tag should be last element in result list
- [ ] Update `wrap_tuple_result` to handle tag correctly
- [ ] Test conditionals with streaming enabled

### Task 6.3: Update While Loop Compilation

- [ ] Identify where while loops use tokens
- [ ] Replace token threading with tag threading
- [ ] Make tag part of loop state
- [ ] Update condition region to handle tag
- [ ] Update body region to handle tag

### Task 6.4: Update Function Regions

- [ ] Check `Function.push_region` usage
- [ ] Ensure regions handle tag arguments if streaming enabled
- [ ] Verify tag is always last argument/result
- [ ] Test nested regions

---

## Phase 7: C++ Custom Call Refinements

### Task 7.1: Ensure Infeed Custom Call is Fully Functional
**Location:** `exla/c_src/exla/custom_calls/infeed.h`

- [ ] Verify `infeed_cpu_custom_call_impl` decodes tag correctly
- [ ] Verify `exla_nif_call_make` calls work with `:next_variadic`
- [ ] Verify list of binaries received correctly
- [ ] Verify binaries copied to result buffers correctly
- [ ] Test buffer count mismatch handling
- [ ] Test single and multiple tensors (variadic)

### Task 7.2: Ensure Outfeed Custom Call is Fully Functional
**Location:** `exla/c_src/exla/custom_calls/outfeed.h`

- [ ] Test `outfeed_cpu_custom_call_impl` with explicit PID
- [ ] Test `outfeed_main_custom_call_impl` with registered process
- [ ] Verify variadic arguments handled correctly
- [ ] Verify binaries sent to Erlang process correctly
- [ ] Verify multiple tensors grouped in list

### Task 7.3: Error Handling in Custom Calls

- [ ] Handle missing handler process gracefully
- [ ] Add timeout handling for infeed requests
- [ ] Add clear error messages for debugging
- [ ] Consider adding logging for failures (via XLA logging)

### Task 7.4: Optimize Memory Copying

- [ ] Review infeed/outfeed for unnecessary copies
- [ ] Investigate zero-copy techniques
- [ ] Verify ErlNifBinary allocations are efficient
- [ ] Profile memory usage

---

## Phase 8: Session Tag Management

### Task 8.1: Tag Creation and Serialization
**Location:** `exla/lib/exla/defn/outfeed.ex` - `start_child/4`

- [ ] Verify tag is unique per session
- [ ] Verify tag can be deserialized in C++
- [ ] Verify tag persists for entire execution
- [ ] Implement tag cleanup after execution

### Task 8.2: Tag Threading Through Compilation
**Location:** `exla/lib/exla/defn.ex` - Function argument handling

- [ ] When outfeed present (streaming enabled), add tag argument
- [ ] Tag should be last argument in function signature
- [ ] Tag should be threaded through all operations
- [ ] Tag should be in return values if needed for next call

### Task 8.3: Tag Buffer Size Standardization

- [ ] Document why 65 bytes is chosen
- [ ] Verify size is sufficient for all tag encodings
- [ ] Ensure consistency across all usages
- [ ] Add constants for tag sizes

---

## Phase 9: Backward Compatibility & Migration

### Task 9.1: Deprecation Warnings

- [ ] Add warning for `Typespec.token()` usage
- [ ] Add warning for direct `transfer_to_infeed` usage
- [ ] Add warning for direct `transfer_from_outfeed` usage
- [ ] Add warning for old test patterns using tokens

### Task 9.2: Documentation Updates

- [ ] Update EXLA.Defn.Outfeed module docs
- [ ] Document new infeed/outfeed architecture
- [ ] Document handler process lifecycle
- [ ] Document session tags and NIF callbacks
- [ ] Add examples of new patterns
- [ ] Update API documentation for `Value.infeed_custom/2`
- [ ] Update API documentation for `Value.outfeed_custom/2`

### Task 9.3: Migration Guide

- [ ] Create before/after code examples
- [ ] Document common migration patterns
- [ ] Add troubleshooting section
- [ ] Document performance characteristics
- [ ] Add debugging guide

---

## Phase 10: Verification & Cleanup

### Task 10.1: Remove All StableHLO Token References

- [ ] Search for `stablehlo.token` in all files
- [ ] Search for `stablehlo.create_token`
- [ ] Search for `stablehlo.infeed` (should only be in comments/docs)
- [ ] Search for `stablehlo.outfeed` (should only be in comments/docs)
- [ ] Verify none exist in production code

### Task 10.2: Run Full Test Suite

- [ ] Run `mix test` in exla directory
- [ ] Verify all streaming/hook tests pass
- [ ] Verify all infeed/outfeed tests pass
- [ ] Verify integration tests pass
- [ ] Fix any failures

### Task 10.3: Code Cleanup

- [ ] Remove commented-out token code
- [ ] Remove unused helper functions
- [ ] Remove debug logging added during development
- [ ] Remove temporary workarounds
- [ ] Run code formatter
- [ ] Check for linter warnings

### Task 10.4: Performance Testing

- [ ] Compare performance with old implementation
- [ ] Check for memory leaks in handler processes
- [ ] Check for resource leaks in NIF calls
- [ ] Test concurrent sessions work correctly
- [ ] Profile under load

---

## Phase 11: Edge Cases & Advanced Scenarios

### Task 11.1: Nested Function Calls with Streaming

- [ ] Test function A calling function B, both with streaming
- [ ] Verify tag threaded through call stack correctly
- [ ] Verify each function can access handler process
- [ ] Verify no tag collisions

### Task 11.2: Multi-Device Streaming

- [ ] Test multiple devices, each with own handler
- [ ] Verify each device has unique handler process
- [ ] Verify tags are device-specific
- [ ] Verify no cross-device interference

### Task 11.3: Handler Process Crash Recovery

- [ ] Test handler process crash during execution
- [ ] Implement detection of crashed handler
- [ ] Implement graceful error reporting
- [ ] Implement proper resource cleanup

### Task 11.4: Large Tensor Infeed/Outfeed

- [ ] Test very large tensors (GB scale)
- [ ] Verify no buffer overflows
- [ ] Verify memory efficiency
- [ ] Verify no timeout issues

---

## Implementation Priority Order

### Critical Path (Must be done first)
1. ✅ Task 1 - Architecture understanding
2. Task 2.1 - Replace Typespec.token()
3. Task 2.2 - Update EXLA.Defn token handling
4. Task 2.3 - Update Value token functions
5. Task 3.4 - Verify handler process loop
6. Task 8.2 - Tag threading through compilation

### Medium Priority (Cleanup & Testing)
7. Task 5.1 - Update old tests
8. Task 4.1 - Remove transfer_to_infeed NIF
9. Task 4.2 - Remove transfer_from_outfeed NIF
10. Task 4.3 - Update Client module
11. Task 10.2 - Run full test suite

### Low Priority (Nice to have)
12. Task 7.3 - Error handling improvements
13. Task 7.4 - Memory optimization
14. Task 9.2 - Documentation updates
15. Task 11.* - Edge case testing

---

## Testing Strategy

### Unit Tests
- [ ] Individual custom call functions
- [ ] Tag serialization/deserialization
- [ ] Handler process message handling
- [ ] Callback registration and invocation

### Integration Tests
- [ ] Full infeed→computation→outfeed pipeline
- [ ] Multiple concurrent sessions
- [ ] Multi-device scenarios
- [ ] Error cases (missing handler, invalid tag, etc.)

### Performance Tests
- [ ] Throughput vs old implementation
- [ ] Memory usage over time
- [ ] Handler process overhead
- [ ] Large tensor handling

---

## Documentation Needs

### Architecture Overview
- [ ] Diagram: BEAM → Handler Process → Custom FFI Call → XLA
- [ ] Explain session tags and how they work
- [ ] Explain handler process role
- [ ] Document design decisions

### API Documentation
- [ ] `Value.infeed_custom/2`
- [ ] `Value.outfeed_custom/2`
- [ ] Handler process interface
- [ ] Session tag management

### Migration Guide
- [ ] Before/after code examples
- [ ] Common patterns
- [ ] Troubleshooting guide
- [ ] Performance debugging

---

## Notes & Considerations

### Why This Migration?
- **Better Control:** Erlang-native semantics instead of XLA queues
- **Better Debugging:** Message passing is easier to trace
- **Variadic Support:** Handle multiple tensors efficiently
- **Concurrent Sessions:** Multiple executions can share same device

### Key Technical Insights
- Tags are 65 bytes because that's sufficient for Erlang term encoding
- Handler process per device simplifies routing (vs passing PIDs everywhere)
- NIF call infrastructure provides safe Erlang↔C++ callbacks
- Last argument position for tags is more natural than first

### Common Pitfalls to Avoid
- Don't mix old token-based and new tag-based code
- Always thread tags through all operations when streaming is enabled
- Handler process must be started before execution
- Session tags must be unique per execution

---

## Progress Tracking

**Overall Progress:** `___%` complete

- Phase 1: ✅ 100%
- Phase 2: ☐ 0%
- Phase 3: ☐ 0%
- Phase 4: ☐ 0%
- Phase 5: ☐ 0%
- Phase 6: ☐ 0%
- Phase 7: ☐ 0%
- Phase 8: ☐ 0%
- Phase 9: ☐ 0%
- Phase 10: ☐ 0%
- Phase 11: ☐ 0%

---

**Last Updated:** October 14, 2025
**Branch:** `pv-feat/infeed-outfeed`
**Status:** Planning Complete, Ready for Implementation

