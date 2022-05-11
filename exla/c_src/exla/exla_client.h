#ifndef EXLA_CLIENT_H_
#define EXLA_CLIENT_H_

#include <memory>
#include <vector>
#include <utility>

#include "exla_nif_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/gpu_device.h"

// The implementations in this module are designed after implementations
// in the XLA runtime, PjRt. Deviations are made where it makes sense
// to work better with the VM.


namespace exla {

class ExlaClient;

class ExlaBuffer {
 public:
  ExlaBuffer(std::unique_ptr<xla::PjRtBuffer> buffer,
             bool can_be_released_after_run_ = false);

  // Make sure the transfer has completed before we deallocate the buffer.
  ~ExlaBuffer() {
    buffer_->BlockHostUntilReady();
  }

  bool release_after_run() { return can_be_released_after_run_; }
  xla::PjRtBuffer* buffer() { return buffer_.get(); }
  xla::StatusOr<ExlaBuffer*> CopyToDevice(xla::PjRtDevice * dst_device);
  xla::StatusOr<ERL_NIF_TERM> ToBinary(ErlNifEnv* env, exla::int64 size);
  xla::Status BlockHostUntilReady();
  xla::Status Deallocate();

 private:
  std::unique_ptr<xla::PjRtBuffer> buffer_;
  bool can_be_released_after_run_;
};

class ExlaExecutable {
 public:
  ExlaExecutable(std::unique_ptr<xla::PjRtExecutable> executable,
                 absl::optional<std::string> fingerprint,
                 ExlaClient* client);

  xla::PjRtExecutable* executable() { return executable_.get(); }

  xla::StatusOr<ERL_NIF_TERM> Run(ErlNifEnv* env,
                                  ERL_NIF_TERM arguments,
                                  int device_id);

 private:
  std::unique_ptr<xla::PjRtExecutable> executable_;
  absl::optional<std::string> fingerprint_;
  ExlaClient* client_;
};

class ExlaClient {
 public:
  explicit ExlaClient(std::shared_ptr<xla::PjRtClient> client);

  virtual ~ExlaClient() = default;

  xla::PjRtClient* client() { return client_.get(); }

  // Compiles the given computation with the given compile options
  xla::StatusOr<ExlaExecutable*> Compile(const xla::XlaComputation&,
                                         std::vector<xla::Shape*> argument_layouts,
                                         xla::ExecutableBuildOptions& options,
                                         bool compile_portable_executable);

  xla::StatusOr<ExlaBuffer*> BufferFromBinary(const ErlNifBinary& binary,
                                              xla::Shape& shape,
                                              int device_id,
                                              bool can_be_released_after_run);

  // TODO(seanmor5): This is device logic and should be refactored
  xla::Status TransferToInfeed(ErlNifEnv* env,
                               ERL_NIF_TERM data,
                               const xla::Shape& shape,
                               int device_id);

  xla::StatusOr<ERL_NIF_TERM> TransferFromOutfeed(ErlNifEnv* env, int device_id, xla::Shape& shape);

 private:
  std::shared_ptr<xla::PjRtClient> client_;
};

xla::StatusOr<ExlaClient*> GetHostClient();

xla::StatusOr<ExlaClient*> GetGpuClient(double memory_fraction,
                                        bool preallocate,
                                        xla::GpuAllocatorConfig::Kind kind);

xla::StatusOr<ExlaClient*> GetTpuClient();
} // namespace exla

#endif
