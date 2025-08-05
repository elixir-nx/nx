#ifndef EXLA_CLIENT_H_
#define EXLA_CLIENT_H_

#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "exla_types.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"
#include <erl_nif.h>
#include <fine.hpp>

// The implementations in this module are designed after implementations
// in the XLA runtime, PjRt. Deviations are made where it makes sense
// to work better with the VM.

namespace exla {

class ExlaClient;

class ExlaBuffer {
public:
  ExlaBuffer(std::unique_ptr<xla::PjRtBuffer> buffer);

  int device_id() { return buffer_->device()->id(); }
  xla::PjRtBuffer *buffer() { return buffer_.get(); }
  tsl::StatusOr<fine::ResourcePtr<ExlaBuffer>>
  CopyToDevice(xla::PjRtDevice *dst_device);
  tsl::StatusOr<ERL_NIF_TERM> ToBinary(ErlNifEnv *env, exla::int64 size);
  tsl::Status Deallocate();

  tsl::StatusOr<std::uintptr_t> GetDevicePointer(xla::PjRtClient *client) {
    return client->UnsafeBufferPointer(buffer_.get());
  }

  tsl::StatusOr<size_t> GetOnDeviceSizeInBytes() {
    return buffer_.get()->GetOnDeviceSizeInBytes();
  }

  void SetClient(ExlaClient *client) { client_ = client; }

  ~ExlaBuffer();

private:
  std::unique_ptr<xla::PjRtBuffer> buffer_;
  ExlaClient *client_ = nullptr;

  void TrackDeallocation();
};

class ExlaExecutable {
public:
  using ReplicaArgument = std::variant<fine::ResourcePtr<ExlaBuffer>,
                                       std::tuple<fine::Term, xla::Shape>>;
  using RunArguments = std::vector<std::vector<ReplicaArgument>>;

  using RunReplicaResult =
      std::tuple<std::vector<fine::ResourcePtr<ExlaBuffer>>, int64_t>;
  using RunResult = std::vector<RunReplicaResult>;

  ExlaExecutable(std::unique_ptr<xla::PjRtLoadedExecutable> executable,
                 absl::optional<std::string> fingerprint, ExlaClient *client);

  xla::PjRtLoadedExecutable *executable() { return executable_.get(); }

  tsl::StatusOr<RunResult> Run(ErlNifEnv *env, RunArguments arguments,
                               int device_id);

  tsl::StatusOr<std::string> SerializeExecutable() {
    return executable_->SerializeExecutable();
  }

private:
  std::unique_ptr<xla::PjRtLoadedExecutable> executable_;
  absl::optional<std::string> fingerprint_;
  ExlaClient *client_;
};

class ExlaClient {
public:
  explicit ExlaClient(std::shared_ptr<xla::PjRtClient> client);

  virtual ~ExlaClient() = default;

  xla::PjRtClient *client() { return client_.get(); }

  // Compiles the given computation with the given compile options

  tsl::StatusOr<fine::ResourcePtr<ExlaExecutable>>
  Compile(mlir::ModuleOp computation, std::vector<xla::Shape> argument_layouts,
          xla::ExecutableBuildOptions &options,
          bool compile_portable_executable);

  tsl::StatusOr<fine::ResourcePtr<ExlaBuffer>>
  BufferFromBinary(ERL_NIF_TERM binary_term, xla::Shape &shape, int device_id);

  tsl::StatusOr<fine::ResourcePtr<ExlaExecutable>>
  DeserializeExecutable(std::string serialized_executable);

  // TODO(seanmor5): This is device logic and should be refactored
  tsl::Status TransferToInfeed(ErlNifEnv *env,
                               std::vector<ErlNifBinary> buffer_bins,
                               std::vector<xla::Shape> shapes, int device_id);

  tsl::StatusOr<ERL_NIF_TERM> TransferFromOutfeed(ErlNifEnv *env, int device_id,
                                                  xla::Shape &shape);

  // Memory tracking methods
  void TrackBufferAllocated(int device_id, size_t size);
  void TrackBufferDeallocated(int device_id, size_t size);
  size_t GetAllocatedMemory() const;
  size_t GetPeakMemory() const;
  void ResetPeakMemory();
  std::map<int, size_t> GetPerDeviceMemory() const;

private:
  std::shared_ptr<xla::PjRtClient> client_;

  // Memory tracking state
  mutable std::mutex memory_mutex_;
  std::map<int, size_t> device_memory_;
  size_t total_memory_ = 0;
  size_t peak_memory_ = 0;
};

tsl::StatusOr<fine::ResourcePtr<ExlaClient>> GetHostClient();

tsl::StatusOr<fine::ResourcePtr<ExlaClient>>
GetGpuClient(double memory_fraction, bool preallocate,
             xla::GpuAllocatorConfig::Kind kind);

tsl::StatusOr<fine::ResourcePtr<ExlaClient>> GetTpuClient();

tsl::StatusOr<fine::ResourcePtr<ExlaClient>>
GetCApiClient(std::string device_type);
} // namespace exla

#endif
