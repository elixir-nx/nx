#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <erl_nif.h>
#include <fine.hpp>
#include "xla/ffi/api/ffi.h"

namespace exla {

struct ElixirCallbackArg {
  xla::ffi::DataType dtype;
  std::vector<int64_t> dims;
  const uint8_t *data = nullptr;
  size_t size_bytes = 0;
};

// Result of an Elixir callback. On success, data has already been copied into
// the pre-registered output buffers held by ElixirCallbackPending, so we only
// need to track success or an error message here.
struct ElixirCallbackResult {
  bool ok = false;
  std::string error;
};

// Host-side description of an output buffer that should receive the callback
// result for a given output index.
struct ElixirCallbackOutputBuffer {
  uint8_t *data = nullptr;
  size_t size = 0;
};

// Per-callback pending state used to synchronize between the XLA host thread
// and the Elixir-side dispatcher. This is exposed as a Fine resource so we
// can pass it as an opaque handle in messages instead of using integer tags.
struct ElixirCallbackPending {
  // Constructor used on the host callback path where we pre-register the
  // destination buffers for each output.
  explicit ElixirCallbackPending(
      std::vector<ElixirCallbackOutputBuffer> outputs)
      : outputs(std::move(outputs)) {}

  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  ElixirCallbackResult result;
  std::vector<ElixirCallbackOutputBuffer> outputs;
};

// Called from the Elixir side to deliver a reply for a given pending handle.
// We receive the reply as a status atom (e.g. :ok or :error) and a result
// term. For the :ok case the result is a list of binaries that we decode as
// ElixirCallbackTensor outputs via Fine's decoding machinery.
void DeliverElixirCallbackReply(
    ErlNifEnv *env, fine::ResourcePtr<ElixirCallbackPending> pending,
    fine::Atom status, fine::Term result);

// Synchronously calls the Elixir callback identified by `callback_id` with the
// given tensor arguments. This function:
//
//   * Allocates a unique reply_tag
//   * Sends a message to the dispatcher via enif_send/3
//   * Blocks the calling native thread until the reply arrives via
//     DeliverElixirCallbackReply/3
//
// It returns an ElixirCallbackResult that either indicates success (data has
// been written into the registered output buffers) or an error message.
ElixirCallbackResult
CallElixirCallback(int64_t callback_id,
                   const std::vector<ElixirCallbackArg> &inputs,
                   const std::vector<ElixirCallbackOutputBuffer> &outputs);

} // namespace exla

namespace fine {

// Decode a binary term into a raw byte vector. We only care about the payload
// bytes; dtype and shape are validated on the Elixir side.
template <> struct Decoder<std::vector<uint8_t>> {
  static std::vector<uint8_t> decode(ErlNifEnv *env, const ERL_NIF_TERM &term) {
    ErlNifBinary bin;
    if (!enif_inspect_binary(env, term, &bin)) {
      throw std::invalid_argument(
          "decode failed, expected binary for callback output");
    }

    std::vector<uint8_t> bytes;
    bytes.assign(bin.data, bin.data + bin.size);
    return bytes;
  }
};

} // namespace fine
