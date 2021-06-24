#include "tensorflow/compiler/xla/exla/exla_host_callback.h"
#include "tensorflow/compiler/xla/exla/exla_nif_util.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "absl/base/casts.h"

namespace exla {

ExlaCallback::ExlaCallback(ErlNifPid registry_pid,
                           std::string name,
                           xla::Shape& shape,
                           size_t size) : registry_pid_(registry_pid),
                                          name_(name),
                                          input_shape_(shape),
                                          value_size_(size) {
  caller_env_ = enif_alloc_env();
}

void ExlaCallback::Call(void ** inputs) {
  ErlNifBinary result_binary;
  enif_alloc_binary(value_size_, &result_binary);
  std::memcpy(result_binary.data, inputs[0], value_size_);

  ERL_NIF_TERM run_atom = nif::atom(caller_env_, "run");
  ERL_NIF_TERM name = nif::make(caller_env_, name_);
  ERL_NIF_TERM data = nif::make(caller_env_, result_binary);
  ERL_NIF_TERM shape = nif::make<xla::Shape>(caller_env_, input_shape_);

  ERL_NIF_TERM send_term = enif_make_tuple4(caller_env_, run_atom, name, data, shape);

  LOG(INFO) << "HERE";

  if (!enif_send(caller_env_, &registry_pid_, NULL, send_term)) {
    LOG(ERROR) << "Unable to send message to registry.";
  }
}

extern "C" void XlaNIFCpuCallback(void * output, void ** inputs) {
  ExlaCallback * callback = absl::bit_cast<ExlaCallback*>(*static_cast<uintptr_t*>(inputs[0]));
  callback->Call(inputs + 1);
  output = inputs[1];
}

XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("xla_nif_cpu_callback", &XlaNIFCpuCallback);

}