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

namespace callback_bridge {

// Opaque handle type used only so Elixir can keep the bridge alive via a
// Fine resource. It carries no data; the real bridge state is stored
// internally in the bridge implementation.
struct ElixirCallbackBridgeHandle {};

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
//   * Allocates a unique ElixirCallbackPending resource
//   * Sends a message to the dispatcher via enif_send/3
//   * Blocks the calling native thread until the reply arrives via
//     DeliverElixirCallbackReply/3
//
// It returns an ElixirCallbackResult that either indicates success (data has
// been written into the registered output buffers) or an error message.
ElixirCallbackResult InvokeElixirCallback(
    int64_t callback_id, const std::vector<ElixirCallbackArg> &inputs,
    const std::vector<ElixirCallbackOutputBuffer> &outputs);

fine::Ok<> start_elixir_callback_bridge(ErlNifEnv *env,
                                        ErlNifPid dispatcher_pid);

fine::Ok<> elixir_callback_reply(
    ErlNifEnv *env, fine::ResourcePtr<ElixirCallbackPending> pending,
    fine::Atom status, fine::Term result);

fine::Ok<> clear_elixir_callback_bridge(ErlNifEnv *env,
                                        ErlNifPid dispatcher_pid);

fine::ResourcePtr<ElixirCallbackBridgeHandle>
acquire_elixir_callback_bridge(ErlNifEnv *env);

} // namespace callback_bridge

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

// Define encoding for {ffi_dtype, dims} into %EXLA.Typespec{} term. This is
// used by the Elixir callback bridge to surface type and shape information
// about callback arguments to the Elixir side.
template <>
struct Encoder<std::tuple<xla::ffi::DataType, std::vector<int64_t>>> {
  static ERL_NIF_TERM
  encode(ErlNifEnv *env,
         const std::tuple<xla::ffi::DataType, std::vector<int64_t>> &spec) {
    const xla::ffi::DataType &dtype = std::get<0>(spec);
    const std::vector<int64_t> &dims = std::get<1>(spec);

    ERL_NIF_TERM keys[] = {fine::encode(env, exla::atoms::__struct__),
                           fine::encode(env, exla::atoms::type),
                           fine::encode(env, exla::atoms::shape)};

    ERL_NIF_TERM values[] = {fine::encode(env, exla::atoms::ElixirEXLATypespec),
                             encode_type(env, dtype),
                             encode_shape(env, dtype, dims)};

    ERL_NIF_TERM map;
    if (!enif_make_map_from_arrays(env, keys, values, 3, &map)) {
      throw std::runtime_error("encode: failed to make a map");
    }

    return map;
  }

private:
  static ERL_NIF_TERM encode_type(ErlNifEnv *env, xla::ffi::DataType dtype) {
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

    throw std::invalid_argument("encode failed, unexpected ffi::DataType");
  }

  static ERL_NIF_TERM encode_shape(ErlNifEnv *env, xla::ffi::DataType dtype,
                                   const std::vector<int64_t> &dims) {
    if (dtype == xla::ffi::DataType::TOKEN) {
      return enif_make_tuple(env, 0);
    }

    std::vector<ERL_NIF_TERM> dim_terms;
    dim_terms.reserve(dims.size());

    for (auto d : dims) {
      dim_terms.push_back(fine::encode<int64_t>(env, d));
    }

    return enif_make_tuple_from_array(env, dim_terms.data(), dim_terms.size());
  }
};

} // namespace fine


