#include "runtime_callback_bridge.h"

#include <cstring>

namespace exla {

namespace callback_bridge {

struct BridgeState {
  ErlNifPid dispatcher_pid;
  bool dispatcher_set = false;
};

BridgeState *GetBridgeState() {
  static BridgeState *state = new BridgeState();
  return state;
}

fine::Ok<> start_runtime_callback_bridge(ErlNifEnv *env,
                                         ErlNifPid dispatcher_pid) {
  (void)env;
  auto state = GetBridgeState();
  state->dispatcher_pid = dispatcher_pid;
  state->dispatcher_set = true;
  return fine::Ok();
}

fine::Ok<> runtime_callback_reply(ErlNifEnv *env,
                                  fine::ResourcePtr<Pending> pending,
                                  fine::Atom status, fine::Term result) {
  deliver_reply(env, pending, status, result);
  return fine::Ok();
}

fine::Ok<> clear_runtime_callback_bridge(ErlNifEnv *env,
                                         ErlNifPid dispatcher_pid) {
  (void)env;
  auto state = GetBridgeState();

  if (state->dispatcher_set &&
      std::memcmp(&state->dispatcher_pid, &dispatcher_pid, sizeof(ErlNifPid)) ==
          0) {
    state->dispatcher_set = false;
  }

  return fine::Ok();
}

void deliver_reply(ErlNifEnv *env, fine::ResourcePtr<Pending> pending,
                   fine::Atom status, fine::Term result_term) {
  Result cb_result;

  if (status == "ok") {
    // Successful reply: result_term is a list of binaries that we decode into
    // raw byte vectors via Fine and copy directly into the registered output
    // buffers.
    try {
      auto payloads = fine::decode<std::vector<ErlNifBinary>>(env, result_term);

      std::lock_guard<std::mutex> lock(pending->mu);

      if (payloads.size() != pending->outputs.size()) {
        cb_result.ok = false;
        cb_result.error =
            "mismatched number of callback outputs vs registered buffers";
      } else {
        cb_result.ok = true;

        for (size_t i = 0; i < payloads.size(); ++i) {
          const ErlNifBinary &bytes = payloads[i];
          auto &out_buf = pending->outputs[i];

          if (bytes.size != out_buf.size) {
            cb_result.ok = false;
            cb_result.error =
                "callback returned binary of unexpected size for result buffer";
            break;
          }

          if (out_buf.size > 0) {
            std::memcpy(out_buf.data, bytes.data, out_buf.size);
          }
        }
      }
    } catch (const std::exception &e) {
      cb_result.ok = false;
      cb_result.error =
          std::string("failed to decode Elixir callback outputs: ") + e.what();
    }
  } else {
    // Error reply: result_term is expected to be {kind_atom, message :: binary}
    cb_result.ok = false;

    try {
      auto decoded =
          fine::decode<std::tuple<fine::Atom, ErlNifBinary>>(env, result_term);
      fine::Atom kind = std::get<0>(decoded);
      ErlNifBinary msg_bin = std::get<1>(decoded);

      cb_result.error =
          "elixir callback returned " + kind.to_string() + ": " +
          std::string(reinterpret_cast<const char *>(msg_bin.data),
                      msg_bin.size);
    } catch (const std::exception &) {
      cb_result.error = "elixir callback returned error";
    }
  }

  {
    std::lock_guard<std::mutex> lock(pending->mu);
    pending->result = std::move(cb_result);
    pending->done = true;
  }

  pending->cv.notify_one();
}

Result InvokeRuntimeCallback(
    xla::ffi::Span<const int64_t> callback_id_words, uint64_t callback_id_size,
    xla::ffi::Span<const int64_t> callback_server_pid_words,
    uint64_t callback_server_pid_size, const std::vector<Arg> &inputs,
    const std::vector<OutputBuffer> &outputs) {
  auto state = GetBridgeState();

  if (!state->dispatcher_set) {
    Result res;
    res.ok = false;
    res.error = "EXLA elixir callback dispatcher is not set";
    return res;
  }

  auto pending = fine::make_resource<Pending>(outputs);

  ErlNifEnv *msg_env = enif_alloc_env();

  // Reinterpret the 64-bit words as a contiguous byte buffer and use the
  // original (unpadded) sizes when decoding the callback id and callback
  // server pid terms.
  if (callback_id_size > callback_id_words.size() * sizeof(int64_t)) {
    Result res;
    res.ok = false;
    res.error = "inconsistent callback id size";
    return res;
  }

  if (callback_server_pid_size >
      callback_server_pid_words.size() * sizeof(int64_t)) {
    Result res;
    res.ok = false;
    res.error = "inconsistent callback server pid size";
    return res;
  }

  const unsigned char *id_bytes =
      reinterpret_cast<const unsigned char *>(callback_id_words.begin());

  ERL_NIF_TERM callback_id_term;
  if (!enif_binary_to_term(msg_env, id_bytes, callback_id_size,
                           &callback_id_term, 0)) {
    Result res;
    res.ok = false;
    res.error = "failed to decode callback id term";
    return res;
  }

  const unsigned char *pid_bytes = reinterpret_cast<const unsigned char *>(
      callback_server_pid_words.begin());

  ERL_NIF_TERM callback_server_pid_term;
  if (!enif_binary_to_term(msg_env, pid_bytes, callback_server_pid_size,
                           &callback_server_pid_term, 0)) {
    Result res;
    res.ok = false;
    res.error = "failed to decode callback server pid term";
    return res;
  }

  ErlNifPid callback_server_pid;
  if (!enif_get_local_pid(msg_env, callback_server_pid_term,
                          &callback_server_pid)) {
    Result res;
    res.ok = false;
    res.error = "failed to decode callback server pid";
    return res;
  }

  // Encode arguments as [{bin, %EXLA.Typespec{}}, ...]. We currently send
  // plain binaries because the BEAM callback needs to own the data lifetime.
  std::vector<std::tuple<fine::Term,
                         std::tuple<xla::ffi::DataType, std::vector<int64_t>>>>
      args_terms;
  args_terms.reserve(inputs.size());

  for (const auto &tensor : inputs) {
    fine::Term bin_term = fine::make_new_binary(
        msg_env, reinterpret_cast<const char *>(tensor.data),
        tensor.size_bytes);

    // Build an %EXLA.Typespec{} directly from the ffi::DataType and dims via
    // Fine's encoder defined in exla_nif_util.h.
    auto arg_tuple =
        std::make_tuple(bin_term, std::make_tuple(tensor.dtype, tensor.dims));

    args_terms.push_back(arg_tuple);
  }

  auto msg = std::make_tuple(fine::Atom("exla_runtime_call"),
                             fine::Term(callback_id_term), args_terms, pending);

  // Use the dispatcher pid registered via start_runtime_callback_bridge/1.
  // We still are within the NIF thread that started the computation,
  // but we don't know its env, therefore we cannot use enif_whereis_pid.
  // enif_whereis_pid can be called with NULL, but only from non-ERTS
  // threads, and doing so here results in a segfault.
  enif_send(msg_env, &callback_server_pid, msg_env, fine::encode(msg_env, msg));
  enif_free_env(msg_env);

  std::unique_lock<std::mutex> lock(pending->mu);
  pending->cv.wait(lock, [&pending] { return pending->done; });

  return pending->result;
}

} // namespace callback_bridge

} // namespace exla


