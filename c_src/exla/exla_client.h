#ifndef EXLA_CLIENT_H_
#define EXLA_CLIENT_H_

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/exla/exla_device.h"
#include "tensorflow/compiler/xla/exla/exla_nif_util.h"

#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/core/framework/allocator.h"

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/status.h"

namespace exla {

namespace se = tensorflow::se;

class ExlaClient;

/*
 * Wraps a ScopedShapedBuffer.
 */
class ExlaBuffer {
 public:
  ExlaBuffer(xla::ScopedShapedBuffer* buffer,
             ExlaDevice* device,
             bool zero_copy) : buffer_(buffer),
                               device_(device),
                               zero_copy_(zero_copy) {}

  ~ExlaBuffer() { Deallocate(); }

  xla::Status Deallocate();

  xla::StatusOr<std::vector<ExlaBuffer*>> DecomposeTuple();

  bool empty() { return buffer_ == nullptr; }


  const xla::Shape on_host_shape() { return buffer_->on_host_shape(); }
  const xla::Shape on_device_shape() { return buffer_->on_device_shape(); }

  xla::ScopedShapedBuffer* buffer() { return buffer_; }

  ExlaDevice* device() { return device_; }

  bool is_tuple() { return !empty() && buffer_->on_host_shape().IsTuple(); }

  xla::Status PopulateBuffer(xla::ScopedShapedBuffer& donated_buffer, ExlaClient* client);

 private:
  // Used for donating this buffer to another function, like `Run`
  xla::ScopedShapedBuffer* buffer_;
  // Was the bool created with a zero-copy transfer
  bool zero_copy_;
  // Buffer's device
  ExlaDevice* device_;
};

/*
 * Wraps an xla::LocalExecutable
 */
class ExlaExecutable {
 public:
  ExlaExecutable(std::vector<std::unique_ptr<xla::LocalExecutable>> executables,
                 std::shared_ptr<xla::DeviceAssignment> device_assignment,
                 std::vector<std::pair<int, int>> local_logical_device_ids,
                 std::vector<ExlaDevice*> local_devices,
                 ExlaClient* client);

  ExlaClient* client() { return client_; }

  int num_replicas() const { return executables_.at(0)->build_options().num_replicas(); }

  int num_partitions() const { return executables_.at(0)->build_options().num_replicas(); }

  const std::vector<std::shared_ptr<xla::LocalExecutable>>& executables() const { return executables_; }

  const xla::DeviceAssignment& device_assignment() const { return *device_assignment_; }

  const std::vector<std::pair<int, int>>& local_logical_device_ids() const { return local_logical_device_ids_; }

  const std::vector<ExlaDevice*> local_devices() { return local_devices_; }

  void Delete() { executables_.clear(); }

  xla::StatusOr<ERL_NIF_TERM> Run(ErlNifEnv* env,
                                  ERL_NIF_TERM arguments,
                                  xla::Shape& output_shape,
                                  int replica,
                                  int partition,
                                  int run_id,
                                  int rng_seed,
                                  int launch_id,
                                  ExlaDevice* device,
                                  bool keep_on_device);

 private:
  ExlaClient* client_;
  std::vector<std::shared_ptr<xla::LocalExecutable>> executables_;
  std::shared_ptr<xla::DeviceAssignment> device_assignment_;
  std::vector<std::pair<int, int>> local_logical_device_ids_;
  std::vector<ExlaDevice*> local_devices_;
};

class ExlaClient {
 public:
  explicit ExlaClient(xla::LocalClient* client,
                      int host_id,
                      std::vector<std::unique_ptr<ExlaDevice>> devices,
                      std::unique_ptr<se::DeviceMemoryAllocator> allocator,
                      std::unique_ptr<tensorflow::Allocator> host_memory_allocator,
                      std::unique_ptr<xla::gpu::GpuExecutableRunOptions> gpu_run_options);


  virtual ~ExlaClient() = default;

  xla::StatusOr<ExlaExecutable*> Compile(const xla::XlaComputation&,
                                           std::vector<xla::Shape*> argument_layouts,
                                           xla::ExecutableBuildOptions& build_options,
                                           bool compile_portable_executable);

  xla::StatusOr<ExlaBuffer*> BufferFromErlBin(const ErlNifBinary binary,
                                              const xla::Shape& shape,
                                              ExlaDevice* device,
                                              bool transfer_for_run);

  xla::StatusOr<ERL_NIF_TERM> DecomposeBuffer(ErlNifEnv* env,
                                              ExlaBuffer* buffer);

  xla::StatusOr<ErlNifBinary> ErlBinFromBuffer(ExlaBuffer* buffer);

  xla::StatusOr<ERL_NIF_TERM> ErlListFromBuffer(ErlNifEnv* env,
                                                ExlaBuffer* buffer);

  xla::StatusOr<xla::DeviceAssignment> GetDefaultDeviceAssignment(int num_replicas, int num_partitions);

  xla::LocalClient* client() { return client_; }

  tensorflow::Allocator* host_memory_allocator() { return host_memory_allocator_.get(); }

  int host_id() { return host_id_; }

  se::DeviceMemoryAllocator* allocator() { return allocator_; }

  xla::gpu::GpuExecutableRunOptions* gpu_run_options() {
    return gpu_run_options_.get();
  }

  int device_count() const { return devices_.size(); }

  const std::vector<std::unique_ptr<ExlaDevice>>& devices() const {
    return devices_;
  }

  exla::ExlaDevice* device(int id) { return devices_.at(id).get(); }

 private:
  xla::LocalClient* client_;
  std::unique_ptr<tensorflow::Allocator> host_memory_allocator_;
  int host_id_;
  se::DeviceMemoryAllocator* allocator_;
  std::unique_ptr<se::DeviceMemoryAllocator> owned_allocator_;
  std::unique_ptr<xla::gpu::GpuExecutableRunOptions> gpu_run_options_;
  std::vector<std::unique_ptr<ExlaDevice>> devices_;
};

// TODO(seanmor5): Separate into different device classes similar to PjRt
xla::StatusOr<ExlaClient*> GetHostClient(int num_replicas,
                                         int intra_op_parallelism_threads);
xla::StatusOr<ExlaClient*> GetGpuClient(int num_replicas,
                                        int intra_op_parallelism_threads,
                                        const char* platform_name);
}  // namespace exla

#endif
