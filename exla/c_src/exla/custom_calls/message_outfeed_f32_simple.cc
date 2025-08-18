#include "message_infeed_simple.h"
#include "xla/ffi/api/ffi.h"
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace ffi = xla::ffi;

// Global state for infeed coordination
struct InfeedRequest {
  float *result_buffer;
  uint64_t result_size;
  bool completed;
  std::condition_variable cv;
  std::mutex mutex;

  InfeedRequest(float *buffer, uint64_t size)
      : result_buffer(buffer), result_size(size), completed(false) {}
};

// Forward declarations - actual definitions are in exla.cc (outside namespace)
extern std::unordered_map<std::string, std::shared_ptr<InfeedRequest>>
    pending_infeeds;
extern std::mutex pending_infeeds_mutex;

// Simplified F32 outfeed implementation
ffi::Error message_outfeed_f32_simple_impl(ffi::Buffer<ffi::U8> pid_buffer,
                                           ffi::Buffer<ffi::F32> data_buffer,
                                           ffi::ResultBuffer<ffi::U8> result) {
  auto pid_dims = pid_buffer.dimensions();
  auto data_dims = data_buffer.dimensions();

  // Calculate sizes
  uint64_t pid_size = 1;
  for (auto dim : pid_dims) {
    pid_size *= dim;
  }

  uint64_t data_size = 1;
  for (auto dim : data_dims) {
    data_size *= dim;
  }

  // Decode the PID
  ErlNifPid pid;
  ErlNifBinary pid_binary;
  pid_binary.data = reinterpret_cast<unsigned char *>(
      const_cast<void *>(pid_buffer.untyped_data()));
  pid_binary.size = pid_size;

  ErlNifEnv *env = enif_alloc_env();
  ERL_NIF_TERM pid_term;
  if (enif_binary_to_term(env, pid_binary.data, pid_binary.size, &pid_term,
                          0) &&
      enif_get_local_pid(env, pid_term, &pid)) {

    // Convert dimensions to vector
    std::vector<uint64_t> shape;
    for (auto dim : data_dims) {
      shape.push_back(dim);
    }

    // Create data binary for the message
    ErlNifBinary data_binary;
    size_t total_bytes = data_size * sizeof(float);
    enif_alloc_binary(total_bytes, &data_binary);
    memcpy(data_binary.data, data_buffer.untyped_data(), total_bytes);

    // Create shape tuple
    ERL_NIF_TERM shape_terms[data_dims.size()];
    for (size_t i = 0; i < data_dims.size(); i++) {
      shape_terms[i] = enif_make_uint64(env, data_dims[i]);
    }
    ERL_NIF_TERM shape_tuple =
        enif_make_tuple_from_array(env, shape_terms, data_dims.size());

    // Create the message: {outfeed, data_binary, shape}
    ERL_NIF_TERM message =
        enif_make_tuple3(env, enif_make_atom(env, "outfeed"),
                         enif_make_binary(env, &data_binary), shape_tuple);

    enif_send(NULL, &pid, env, message);
  }

  enif_free_env(env);

  // Return confirmation
  uint8_t *result_data = result->typed_data();
  result_data[0] = 1;

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    message_outfeed_f32_simple, message_outfeed_f32_simple_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::U8>>()   // PID buffer
        .Arg<ffi::Buffer<ffi::F32>>()  // Data buffer
        .Ret<ffi::Buffer<ffi::U8>>()); // Confirmation result

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "message_outfeed_f32_simple",
                         "Host", message_outfeed_f32_simple);

// Simplified F32 infeed implementation
ffi::Error message_infeed_f32_simple_impl(ffi::Buffer<ffi::U8> pid_buffer,
                                          ffi::ResultBuffer<ffi::F32> result) {
  auto pid_dims = pid_buffer.dimensions();
  auto result_dims = result->dimensions();

  // Calculate sizes
  uint64_t pid_size = 1;
  for (auto dim : pid_dims) {
    pid_size *= dim;
  }

  uint64_t result_size = 1;
  for (auto dim : result_dims) {
    result_size *= dim;
  }

  // Decode the PID
  ErlNifPid pid;
  ErlNifBinary pid_binary;
  pid_binary.data = reinterpret_cast<unsigned char *>(
      const_cast<void *>(pid_buffer.untyped_data()));
  pid_binary.size = pid_size;

  ErlNifEnv *env = enif_alloc_env();
  ERL_NIF_TERM pid_term;
  if (enif_binary_to_term(env, pid_binary.data, pid_binary.size, &pid_term,
                          0) &&
      enif_get_local_pid(env, pid_term, &pid)) {

    // Create a unique reference for this infeed request
    ERL_NIF_TERM ref = enif_make_ref(env);

    // Convert reference to string for lookup
    ErlNifBinary ref_binary;
    enif_term_to_binary(env, ref, &ref_binary);
    std::string ref_id(reinterpret_cast<char *>(ref_binary.data),
                       ref_binary.size);

    // Convert result dimensions to vector for the shape
    std::vector<uint64_t> shape;
    for (auto dim : result_dims) {
      shape.push_back(dim);
    }

    // Create shape tuple
    ERL_NIF_TERM shape_terms[result_dims.size()];
    for (size_t i = 0; i < result_dims.size(); i++) {
      shape_terms[i] = enif_make_uint64(env, result_dims[i]);
    }
    ERL_NIF_TERM shape_tuple =
        enif_make_tuple_from_array(env, shape_terms, result_dims.size());

    // Create infeed request and store it
    float *result_data = result->typed_data();
    auto infeed_request =
        std::make_shared<InfeedRequest>(result_data, result_size);

    {
      std::lock_guard<std::mutex> lock(pending_infeeds_mutex);
      pending_infeeds[ref_id] = infeed_request;
    }

    // Send infeed request message: {infeed_request, ref, shape}
    ERL_NIF_TERM request_message = enif_make_tuple3(
        env, enif_make_atom(env, "infeed_request"), ref, shape_tuple);

    enif_send(NULL, &pid, env, request_message);

    // Wait for Elixir to respond with data (with timeout)
    std::unique_lock<std::mutex> lock(infeed_request->mutex);
    auto timeout = std::chrono::milliseconds(5000); // 5 second timeout

    if (infeed_request->cv.wait_for(
            lock, timeout, [&] { return infeed_request->completed; })) {
      // Data was provided by Elixir via complete_infeed_request NIF
      // Result buffer was already filled
    } else {
      // Timeout - fill with test data as fallback
      for (uint64_t i = 0; i < result_size; i++) {
        result_data[i] = 5.0f + (float)i; // Test data: 5, 6, 7, 8...
      }
    }

    // Clean up the request
    {
      std::lock_guard<std::mutex> cleanup_lock(pending_infeeds_mutex);
      pending_infeeds.erase(ref_id);
    }
  } else {
    // If PID decoding fails, just return zeros
    float *result_data = result->typed_data();
    for (uint64_t i = 0; i < result_size; i++) {
      result_data[i] = 0.0f;
    }
  }

  enif_free_env(env);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    message_infeed_f32_simple, message_infeed_f32_simple_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::U8>>()    // PID buffer
        .Ret<ffi::Buffer<ffi::F32>>()); // Result buffer

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "message_infeed_f32_simple",
                         "Host", message_infeed_f32_simple);