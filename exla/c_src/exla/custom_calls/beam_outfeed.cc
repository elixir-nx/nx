// Experimental BEAM outfeed FFI handler.
//
// This custom call sends tensor data to a BEAM process (EXLA.FFI.Stream)
// using enif_send, replacing the token-based outfeed mechanism.

#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"
#include <erl_nif.h>
#include <memory>

namespace ffi = xla::ffi;

namespace exla {

// Custom deleter for ErlNifEnv to use with std::unique_ptr (RAII)
struct NifEnvDeleter {
  void operator()(ErlNifEnv *env) const {
    if (env)
      enif_free_env(env);
  }
};

using UniqueNifEnv = std::unique_ptr<ErlNifEnv, NifEnvDeleter>;

// Helper: Decode PID from u8 buffer using enif_binary_to_term
std::pair<ErlNifPid, bool> decode_pid_from_buffer(const uint8_t *data,
                                                  size_t size) {
  ErlNifPid pid;
  // Create a temporary environment for decoding (RAII with automatic cleanup)
  UniqueNifEnv temp_env(enif_alloc_env());
  if (!temp_env) {
    return {pid, false};
  }

  ERL_NIF_TERM term;
  if (enif_binary_to_term(temp_env.get(), data, size, &term, 0) == 0) {
    return {pid, false};
  }

  // Extract PID from the term
  if (!enif_get_local_pid(temp_env.get(), term, &pid)) {
    return {pid, false};
  }
  return {pid, true};
}

// Experimental: send payload to BEAM stream process via enif_send.
static ffi::Error
exla_beam_outfeed_impl(ffi::Buffer<ffi::U8> pid_bytes, ffi::AnyBuffer payload,
                       ffi::Result<ffi::BufferR0<ffi::U8>> success_flag) {
  // Step 1: Decode the PID from pid_bytes
  auto pid_data = pid_bytes.untyped_data();
  auto pid_dims = pid_bytes.dimensions();
  size_t pid_size = 1;
  for (auto dim : pid_dims) {
    pid_size *= dim;
  }

  auto [stream_pid, pid_success] =
      decode_pid_from_buffer((const uint8_t *)pid_data, pid_size);

  if (!pid_success) {
    // Write failure flag
    auto success_data = success_flag->untyped_data();
    static_cast<uint8_t *>(success_data)[0] = 0;
    return ffi::Error::Success();
  }

  // Step 2: Create a message environment for sending (RAII with automatic
  // cleanup)
  UniqueNifEnv msg_env(enif_alloc_env());
  if (!msg_env) {
    // Write failure flag
    auto success_data = success_flag->untyped_data();
    static_cast<uint8_t *>(success_data)[0] = 0;
    return ffi::Error::Success();
  }

  // Step 3: Convert payload to binary term
  auto payload_data = payload.untyped_data();
  auto payload_dims = payload.dimensions();
  size_t payload_size = 1;
  for (auto dim : payload_dims) {
    payload_size *= dim;
  }

  // Calculate bytes - payload is already in bytes for AnyBuffer
  // For typed buffers, we'd need element_size, but AnyBuffer gives us raw bytes
  size_t total_bytes = payload_size;

  ERL_NIF_TERM binary_term;
  unsigned char *binary_data =
      enif_make_new_binary(msg_env.get(), total_bytes, &binary_term);
  if (!binary_data) {
    // Write failure flag
    auto success_data = success_flag->untyped_data();
    static_cast<uint8_t *>(success_data)[0] = 0;
    return ffi::Error::Success();
  }

  memcpy(binary_data, payload_data, total_bytes);

  // Step 4: Create message tuple {:exla_outfeed, payload_binary}
  ERL_NIF_TERM atom_outfeed = enif_make_atom(msg_env.get(), "exla_outfeed");
  ERL_NIF_TERM message =
      enif_make_tuple2(msg_env.get(), atom_outfeed, binary_term);

  // Step 5: Send message to stream process
  auto success_data = success_flag->untyped_data();
  if (!enif_send(nullptr, &stream_pid, msg_env.get(), message)) {
    // Write failure flag
    static_cast<uint8_t *>(success_data)[0] = 0;
  } else {
    // Write success flag
    static_cast<uint8_t *>(success_data)[0] = 1;
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(exla_beam_outfeed, exla_beam_outfeed_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::U8>>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::BufferR0<ffi::U8>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "exla_beam_outfeed", "Host",
                         exla_beam_outfeed);

} // namespace exla
