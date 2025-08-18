#pragma once

#include <iostream>
#include <vector>

#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"
#include <erl_nif.h>

namespace ffi = xla::ffi;

template <typename DataType, typename BufferType>
ffi::Error pid_cpu_custom_call_impl(BufferType operand,
                                    ffi::Result<BufferType> result) {
  auto operand_dims = operand.dimensions();

  // For a 29xu8 tensor, we expect dimensions [29]
  uint64_t total_size = 1;
  for (auto dim : operand_dims) {
    total_size *= dim;
  }

  std::cout << "PID bytes (" << total_size << " bytes): ";

  // Print all bytes in the array
  const DataType *data = operand.typed_data();
  for (uint64_t i = 0; i < total_size; i++) {
    std::cout << static_cast<int>(data[i]);
    if (i < total_size - 1) {
      std::cout << " ";
    }
  }
  std::cout << std::endl;

  ErlNifEnv *env = enif_alloc_env();
  ERL_NIF_TERM pid_term;
  ErlNifPid pid;
  enif_binary_to_term(
      env, reinterpret_cast<const unsigned char *>(operand.untyped_data()),
      total_size, &pid_term, 0);
  enif_get_local_pid(env, pid_term, &pid);

  ERL_NIF_TERM msg = enif_make_atom(env, "pid_cpu_custom_call_result");
  enif_send(NULL, &pid, env, msg);

  // Return a single byte (value 1) as confirmation
  DataType *result_data = result->typed_data();
  result_data[0] = static_cast<DataType>(1);

  return ffi::Error::Success();
}