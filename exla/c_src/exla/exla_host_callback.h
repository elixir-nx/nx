#ifndef EXLA_HOST_CALLBACK_H_
#define EXLA_HOST_CALLBACK_H_

#include "tensorflow/compiler/xla/exla/exla_nif_util.h"
#include "tensorflow/compiler/xla/shape.h"

namespace exla {

class ExlaCallback {
 public:

  ExlaCallback(ErlNifPid registry_pid,
               std::string name,
               xla::Shape& input_shape,
               size_t size_bytes);

  ~ExlaCallback() { enif_free_env(caller_env_); }

  void Call(void ** inputs);

 private:
  ErlNifEnv* caller_env_;
  ErlNifPid registry_pid_;
  std::string name_;
  xla::Shape input_shape_;
  size_t value_size_;
};

}
#endif