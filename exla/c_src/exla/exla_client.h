#ifndef EXLA_CLIENT_H_
#define EXLA_CLIENT_H_

#include <memory>
#include <vector>
#include <utility>

#include "tensorflow/compiler/xla/exla/exla_nif_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"

// The implementations in this module are designed after implementations
// in the XLA runtime, PjRt. Deviations are made where it makes sense
// to work better with the VM.


namespace exla {

class ExlaClient;

class ExlaBuffer {
 public:
  explicit ExlaBuffer(std::unique_ptr<xla::PjRtBuffer> buffer);

  xla::PjRtBuffer* buffer() { buffer_.get(); }

  xla::StatusOr<ERL_NIF_TERM> ToBinary(ErlNifEnv* env);

  void Deallocate();

  // static xla::StatusOr<ERL_NIF_TERM>
  // DecomposeBufferToTerm(ErlNifEnv* env,
  //                       ExlaBuffer* buffer,
  //                       bool keep_on_device);

 private:
  std::unique_ptr<xla::PjRtBuffer> buffer_;
}

class ExlaExecutable {
 public:
  explicit ExlaExecutable(std::unique_ptr<xla::PjRtExecutable> executable);

  xla::PjRtExecutable* executable() { executable_.get(); }

  xla::StatusOr<ERL_NIF_TERM> ExlaExecutable::Run(ErlNifEnv* env,
                                                ERL_NIF_TERM arguments,
                                                bool keep_on_device);

 private:
  std::unique_ptr<xla::PjRtExecutable> executable_;
}

class ExlaClient {
 public:
  explicit ExlaClient(std::unique_ptr<xla::PjRtClient> client);

  virtual ~ExlaClient() = default;

  xla::PjRtClient* client() { client_.get(); }

  // Compiles the given computation with the given compile
  // options
  xla::StatusOr<ExlaExecutable*> Compile(const xla::XlaComputation&,
                                         std::vector<xla::Shape*> argument_layouts,
                                         xla::ExecutableBuildOptions& options,
                                         bool compile_portable_executable);

  xla::StatusOr<ExlaBuffer*> BufferFromBinary(const ErlNifBinary& binary,
                                              xla::Shape& shape,
                                              int device_id);

 private:
  std::unique_ptr<xla::PjRtClient> client_;
}

xla::StatusOr<ExlaClient*> GetCpuClient();

xla::StatusOr<ExlaClient*> GetGpuClient(double memory_fraction,
                                        bool preallocate,
                                        xla::GpuAllocatorConfig::Kind kind);

xla::StatusOr<ExlaClient*> GetTpuClient();
} // namespace exla

#endif
