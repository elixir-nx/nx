// Experimental BEAM infeed FFI handler using nif_call.
//
// This custom call fetches infeed data from a BEAM stream process via nif_call,
// enabling Elixir-side control of infeed values without the traditional
// token-based mechanism.

#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"
#include <cstring>
#include <erl_nif.h>
#include <memory>
#include <utility>

namespace ffi = xla::ffi;

namespace exla {

// Custom deleter for ErlNifEnv to ensure proper cleanup
struct NifEnvDeleter {
  void operator()(ErlNifEnv *env) const {
    if (env) {
      enif_free_env(env);
    }
  }
};

using UniqueNifEnv = std::unique_ptr<ErlNifEnv, NifEnvDeleter>;

// Helper to decode a BEAM term from a u8 buffer
// Returns {success, term} pair
static std::pair<bool, ERL_NIF_TERM>
decode_term_from_buffer(const uint8_t *data, size_t size, ErlNifEnv *env) {
  ERL_NIF_TERM term;
  int result = enif_binary_to_term(env, data, size, &term, 0);
  return {result > 0, term};
}

// Helper to encode a BEAM term into a u8 buffer
// Returns {success, actual_size} pair
static std::pair<bool, size_t> encode_term_to_buffer(ERL_NIF_TERM term,
                                                     uint8_t *data,
                                                     size_t max_size,
                                                     ErlNifEnv *env) {
  ErlNifBinary binary;

  if (!enif_term_to_binary(env, term, &binary)) {
    return {false, 0};
  }

  if (binary.size > max_size) {
    enif_release_binary(&binary);
    return {false, binary.size};
  }

  memcpy(data, binary.data, binary.size);
  size_t actual_size = binary.size;
  enif_release_binary(&binary);

  return {true, actual_size};
}

// Experimental: fetch payload from BEAM stream via nif_call.
static ffi::Error
exla_beam_infeed_impl(ffi::Buffer<ffi::U8> pid_bytes,
                      ffi::Buffer<ffi::U8> tag_bytes,
                      ffi::ResultBuffer<ffi::U8> out_payload,
                      ffi::ResultBuffer<ffi::U8> out_tag_bytes) {
  // Step 1: Create NIF environment for this call (RAII with automatic cleanup)
  UniqueNifEnv env(enif_alloc_env());
  if (!env) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "Failed to allocate NIF environment");
  }

  // Step 2: Decode PID from pid_bytes
  auto pid_dimensions = pid_bytes.dimensions();
  if (pid_dimensions.size() != 1) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "PID buffer must be 1D");
  }

  size_t pid_size = pid_dimensions[0];
  const uint8_t *pid_data =
      static_cast<const uint8_t *>(pid_bytes.untyped_data());

  auto [pid_success, pid_term] =
      decode_term_from_buffer(pid_data, pid_size, env.get());
  if (!pid_success) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Failed to decode PID");
  }

  ErlNifPid pid;
  if (!enif_get_local_pid(env.get(), pid_term, &pid)) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "PID term is not a valid PID");
  }

  // Step 3: Decode tag from tag_bytes
  auto tag_dimensions = tag_bytes.dimensions();
  if (tag_dimensions.size() != 1) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "Tag buffer must be 1D");
  }

  size_t tag_size = tag_dimensions[0];
  const uint8_t *tag_data =
      static_cast<const uint8_t *>(tag_bytes.untyped_data());

  auto [tag_success, tag_term] =
      decode_term_from_buffer(tag_data, tag_size, env.get());
  if (!tag_success) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Failed to decode tag");
  }

  // Step 4: Build arguments for EXLA.FFI.Stream.next_infeed/3
  // We need to call: EXLA.FFI.Stream.next_infeed(pid, tag, timeout)
  // For now, use a hardcoded timeout of 5000ms
  ERL_NIF_TERM timeout_term = enif_make_int(env.get(), 5000);

  // Build argument list: [pid_term, tag_term, timeout_term]
  ERL_NIF_TERM args[3] = {pid_term, tag_term, timeout_term};

  // Step 5: Call EXLA.FFI.Stream.next_infeed using nif_call
  // Create a unique tag for this nif_call
  ERL_NIF_TERM call_tag = enif_make_ref(env.get());

  // Build the callback: {EXLA.FFI.Stream, :next_infeed, [pid, tag, timeout]}
  ERL_NIF_TERM module_atom = enif_make_atom(env.get(), "Elixir.EXLA.FFI.Stream");
  ERL_NIF_TERM function_atom = enif_make_atom(env.get(), "next_infeed");

  // TODO: Integrate with nif_call - this requires:
  // 1. Setting up the nif_call infrastructure (runner process, etc.)
  // 2. Making the actual call
  // 3. Waiting for the response
  //
  // For now, return an error indicating this is not yet implemented
  return ffi::Error(
      ffi::ErrorCode::kUnimplemented,
      "nif_call integration not yet complete - requires runner process setup");

  // TODO: Once nif_call is working, the rest would be:
  // Step 6: Parse the response {:ok, new_tag, payload_term}
  // Step 7: Encode new_tag into out_tag_bytes
  // Step 8: Decode payload_term and write to out_payload
  // Step 9: Clean up happens automatically via UniqueNifEnv destructor
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(exla_beam_infeed, exla_beam_infeed_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::U8>>()
                                  .Arg<ffi::Buffer<ffi::U8>>()
                                  .Ret<ffi::Buffer<ffi::U8>>()
                                  .Ret<ffi::Buffer<ffi::U8>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "exla_beam_infeed", "Host",
                         exla_beam_infeed);

} // namespace exla
