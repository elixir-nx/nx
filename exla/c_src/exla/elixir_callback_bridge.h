#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <erl_nif.h>
#include <fine.hpp>
#include "xla/ffi/api/ffi.h"

namespace exla {

// Lightweight tensor payload used to transfer arguments and results between
// the XLA host CustomCall handler and the Elixir dispatcher.
struct ElixirCallbackTensor {
  xla::ffi::DataType dtype;
  std::vector<int64_t> dims;
  std::vector<uint8_t> data;
};

struct ElixirCallbackResult {
  bool ok = false;
  std::string error;
  std::vector<ElixirCallbackTensor> outputs;
};

// Called from the Elixir side to deliver a reply for a given callback tag.
void DeliverElixirCallbackReply(ErlNifEnv *env, int64_t reply_tag,
                                fine::Term payload);

// Synchronously calls the Elixir callback identified by `callback_id` with the
// given tensor arguments. This function:
//
//   * Allocates a unique reply_tag
//   * Sends a message to the dispatcher via enif_send/3
//   * Blocks the calling native thread until the reply arrives via
//     DeliverElixirCallbackReply/3
//
// It returns an ElixirCallbackResult that either contains a list of output
// tensors (on success) or an error message.
ElixirCallbackResult CallElixirCallback(int64_t callback_id,
                                        const std::vector<ElixirCallbackTensor> &inputs);

} // namespace exla


