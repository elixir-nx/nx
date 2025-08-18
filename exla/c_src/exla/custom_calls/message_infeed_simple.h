#pragma once

#include <iostream>
#include <vector>
#include <string>

#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"
#include <erl_nif.h>

namespace ffi = xla::ffi;

// Simple message-based infeed/outfeed without complex state management

// Helper function to send a message to an Elixir process
inline void send_tensor_message(ErlNifPid &pid, const std::string &type,
                                const void *data, size_t data_size,
                                const std::vector<uint64_t> &shape) {
  ErlNifEnv *env = enif_alloc_env();

  // Create the data binary
  ErlNifBinary data_binary;
  enif_alloc_binary(data_size, &data_binary);
  memcpy(data_binary.data, data, data_size);

  // Create shape tuple
  ERL_NIF_TERM shape_terms[shape.size()];
  for (size_t i = 0; i < shape.size(); i++) {
    shape_terms[i] = enif_make_uint64(env, shape[i]);
  }
  ERL_NIF_TERM shape_tuple =
      enif_make_tuple_from_array(env, shape_terms, shape.size());

  // Create the message: {type, data_binary, shape}
  ERL_NIF_TERM message =
      enif_make_tuple3(env, enif_make_atom(env, type.c_str()),
                       enif_make_binary(env, &data_binary), shape_tuple);

  enif_send(NULL, &pid, env, message);
  enif_free_env(env);
}

// Helper function to receive tensor data from Elixir (simplified approach)
inline std::vector<uint8_t>
receive_tensor_data(ErlNifPid &pid, const std::vector<uint64_t> &shape) {
  // For now, return zeros - in a full implementation this would need
  // to coordinate with Elixir to receive the actual data
  size_t total_elements = 1;
  for (auto dim : shape) {
    total_elements *= dim;
  }

  // Assuming float32 for simplicity (4 bytes per element)
  size_t data_size = total_elements * 4;
  return std::vector<uint8_t>(data_size, 0);
}