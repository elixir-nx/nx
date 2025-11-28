#include "elixir_callback_bridge.h"

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

fine::Ok<> start_elixir_callback_bridge(ErlNifEnv *env,
                                        ErlNifPid dispatcher_pid) {
  (void)env;
  auto state = GetBridgeState();
  state->dispatcher_pid = dispatcher_pid;
  state->dispatcher_set = true;
  return fine::Ok();
}

fine::Ok<> elixir_callback_reply(ErlNifEnv *env,
                                 fine::ResourcePtr<Pending> pending,
                                 fine::Atom status, fine::Term result) {
  deliver_reply(env, pending, status, result);
  return fine::Ok();
}

fine::Ok<> clear_elixir_callback_bridge(ErlNifEnv *env,
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
      ErlNifBinary msg_bin = std::get<1>(decoded);
      cb_result.error.assign(reinterpret_cast<const char *>(msg_bin.data),
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

Result InvokeElixirCallback(int64_t callback_id, const std::vector<Arg> &inputs,
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

  // Encode arguments as [{bin, %EXLA.Typespec{}}, ...]. We currently send
  // plain binaries because the BEAM callback needs to own the data lifetime.
  std::vector<ERL_NIF_TERM> args_terms;
  args_terms.reserve(inputs.size());

  for (const auto &tensor : inputs) {
    ERL_NIF_TERM bin_term;
    unsigned char *bin_data =
        enif_make_new_binary(msg_env, tensor.size_bytes, &bin_term);
    if (tensor.size_bytes > 0) {
      memcpy(bin_data, tensor.data, tensor.size_bytes);
    }

    // Build an %EXLA.Typespec{} directly from the ffi::DataType and dims via
    // Fine's encoder defined in exla_nif_util.h.
    ERL_NIF_TERM typespec_term =
        fine::encode(msg_env, std::make_tuple(tensor.dtype, tensor.dims));

    ERL_NIF_TERM arg_tuple = enif_make_tuple2(msg_env, bin_term, typespec_term);

    args_terms.push_back(arg_tuple);
  }

  ERL_NIF_TERM args_list =
      enif_make_list_from_array(msg_env, args_terms.data(), args_terms.size());

  ERL_NIF_TERM pending_term = fine::encode(msg_env, pending);
  ERL_NIF_TERM cb_term = enif_make_int64(msg_env, callback_id);

  ERL_NIF_TERM msg =
      enif_make_tuple4(msg_env, enif_make_atom(msg_env, "exla_elixir_call"),
                       cb_term, args_list, pending_term);

  // Use the dispatcher pid registered via start_elixir_callback_bridge/1.
  // Calling enif_whereis_pid from this non-scheduler thread is unsafe and
  // was causing a segfault.
  ErlNifPid dispatcher_pid = state->dispatcher_pid;
  enif_send(msg_env, &dispatcher_pid, msg_env, msg);
  enif_free_env(msg_env);

  std::unique_lock<std::mutex> lock(pending->mu);
  pending->cv.wait(lock, [&pending] { return pending->done; });

  return pending->result;
}

} // namespace callback_bridge

} // namespace exla


