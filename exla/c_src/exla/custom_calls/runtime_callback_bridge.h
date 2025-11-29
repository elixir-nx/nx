#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "../exla_nif_util.h"
#include "xla/ffi/api/ffi.h"
#include <erl_nif.h>
#include <fine.hpp>

namespace exla {

namespace callback_bridge {

struct Arg {
  xla::ffi::DataType dtype;
  std::vector<int64_t> dims;
  const uint8_t *data = nullptr;
  size_t size_bytes = 0;
};

// Result of an Elixir callback. On success, data has already been copied into
// the pre-registered output buffers held by Pending, so we only
// need to track success or an error message here.
struct Result {
  bool ok = false;
  std::string error;
};

// Host-side description of an output buffer that should receive the callback
// result for a given output index.
struct OutputBuffer {
  uint8_t *data = nullptr;
  size_t size = 0;
};

// Per-callback pending state used to synchronize between the XLA host thread
// and the Elixir-side dispatcher. This is exposed as a Fine resource so we
// can pass it as an opaque handle in messages instead of using integer tags.
struct Pending {
  // Constructor used on the host callback path where we pre-register the
  // destination buffers for each output.
  explicit Pending(std::vector<OutputBuffer> outputs)
      : outputs(std::move(outputs)) {}

  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  Result result;
  std::vector<OutputBuffer> outputs;
};

// Called from the Elixir side to deliver a reply for a given pending handle.
// We receive the reply as a status atom (e.g. :ok or :error) and a result
// term. For the :ok case the result is a list of binaries that we decode as
// RuntimeCallbackTensor outputs via Fine's decoding machinery.
void deliver_reply(ErlNifEnv *env, fine::ResourcePtr<Pending> pending,
                   fine::Atom status, fine::Term result);

// Synchronously calls the Elixir callback identified by `callback_id` with the
// given tensor arguments. This function:
//
//   * Allocates a unique Pending resource
//   * Sends a message to the dispatcher via enif_send/3
//   * Blocks the calling native thread until the reply arrives via
//     deliver_reply/3
//
// It returns a Result that either indicates success (data has
// been written into the registered output buffers) or an error message.
Result InvokeRuntimeCallback(
    xla::ffi::Span<const int64_t> callback_id_words, uint64_t callback_id_size,
    xla::ffi::Span<const int64_t> callback_server_pid_words,
    uint64_t callback_server_pid_size, const std::vector<Arg> &inputs,
    const std::vector<OutputBuffer> &outputs);

fine::Ok<> start_runtime_callback_bridge(ErlNifEnv *env,
                                         ErlNifPid dispatcher_pid);

fine::Ok<> runtime_callback_reply(ErlNifEnv *env,
                                  fine::ResourcePtr<Pending> pending,
                                  fine::Atom status, fine::Term result);

fine::Ok<> clear_runtime_callback_bridge(ErlNifEnv *env,
                                         ErlNifPid dispatcher_pid);

} // namespace callback_bridge

} // namespace exla

namespace fine {

// Define encoding for {ffi_dtype, dims} into %EXLA.Typespec{} term. This is
// used by the Elixir callback bridge to surface type and shape information
// about callback arguments to the Elixir side.
template <> struct Encoder<xla::ffi::DataType> {
  static ERL_NIF_TERM encode(ErlNifEnv *env, const xla::ffi::DataType &dtype) {
    using DT = xla::ffi::DataType;

    // Tokens are encoded as the atom :token with empty shape.
    if (dtype == DT::TOKEN) {
      return fine::encode(env, exla::atoms::token);
    }

    std::optional<fine::Atom> type_name;
    std::optional<uint64_t> type_size;

    switch (dtype) {
    case DT::PRED:
      type_name = exla::atoms::pred;
      type_size = 8;
      break;

    case DT::U8:
      type_name = exla::atoms::u;
      type_size = 8;
      break;
    case DT::U16:
      type_name = exla::atoms::u;
      type_size = 16;
      break;
    case DT::U32:
      type_name = exla::atoms::u;
      type_size = 32;
      break;
    case DT::U64:
      type_name = exla::atoms::u;
      type_size = 64;
      break;

    case DT::S8:
      type_name = exla::atoms::s;
      type_size = 8;
      break;
    case DT::S16:
      type_name = exla::atoms::s;
      type_size = 16;
      break;
    case DT::S32:
      type_name = exla::atoms::s;
      type_size = 32;
      break;
    case DT::S64:
      type_name = exla::atoms::s;
      type_size = 64;
      break;

    case DT::F16:
      type_name = exla::atoms::f;
      type_size = 16;
      break;
    case DT::F32:
      type_name = exla::atoms::f;
      type_size = 32;
      break;
    case DT::F64:
      type_name = exla::atoms::f;
      type_size = 64;
      break;

    case DT::BF16:
      type_name = exla::atoms::bf;
      type_size = 16;
      break;

    case DT::C64:
      type_name = exla::atoms::c;
      type_size = 64;
      break;
    case DT::C128:
      type_name = exla::atoms::c;
      type_size = 128;
      break;

    default:
      break;
    }

    if (type_name && type_size) {
      return fine::encode(
          env, std::make_tuple(type_name.value(), type_size.value()));
    }

    throw std::invalid_argument("encode failed, unexpected xla::ffi::DataType");
  }
};

} // namespace fine


