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
 * Representation of an on-device buffer used during computations.
 */
class ExlaBuffer {
 public:

  // Similar to PjRt, we attach semantics to each type of buffer so we know
  // how to handle the buffer during different operations. The differences
  // between each are mainly in how the buffer was constructed.
  enum class BufferType {
    // The buffer was created using a zero-copy transfer from the
    // VM. Internally, the buffer points to a binary owned by
    // the VM. This is only possible in certain circumstances
    // on a CPU device. Because the VM will eventually garbage
    // collect the underlying binary, we need to release ownership
    // back to the VM without modifying the underlying buffer
    kZeroCopy,

    // The buffer was created using an explicit device transfer
    // and therefore the VM holds a reference to the underlying
    // buffer. Usage of this buffer needs to be immutable because
    // the reference can be used multiple times. The difference
    // between a kZeroCopy and kImmutable is that kImmutable buffers
    // are allowed to deallocate their underyling device buffers.
    kReference,

    // The buffer was created during a call to run or another
    // operation and therefore the VM does not need to know of
    // it's existence. We can "donate" the underlying buffer
    // to XLA and allow it to destruct when it goes out of
    // scope.
    kTemporary
  };

  // The current state of this buffer
  enum class BufferState {
    // The buffer is in a valid and useable state
    kValid,

    // The buffer has already been deallocated
    kDeallocated,

    // The buffer is waiting on it's definition event
    kWaiting,

    // The buffer is in an error state
    kError
  };

  ExlaBuffer(absl::Span<se::DeviceMemoryBase const> device_memory,
             ExlaDevice* device,
             ExlaClient* client,
             BufferType type) : device_memory_(device_memory.begin(), device_memory.end()),
                                device_(device),
                                client_(client),
                                type_(type),
                                state_(BufferState::kValid) {}

  ExlaBuffer(se::Stream* creation_stream,
             std::unique_ptr<se::Event> definition_event,
             std::unique_ptr<xla::ScopedShapedBuffer> buffer,
             ExlaDevice* device,
             ExlaClient* client,
             BufferType type) : creation_stream_(creation_stream),
                                definition_event_(std::move(definition_event)),
                                buffer_(std::move(buffer)),
                                device_(device),
                                client_(client),
                                type_(type) {}

  ~ExlaBuffer() { Deallocate(); }

  xla::Status Deallocate();

  xla::StatusOr<std::vector<ExlaBuffer*>> DecomposeTuple();

  bool empty() { return state_ == BufferState::kDeallocated; }

  const xla::Shape on_host_shape() { return on_host_shape_; }

  const xla::Shape on_device_shape() { return on_device_shape_; }

  ExlaDevice* device() { return device_; }

  se::Event* definition_event() const { return definition_event_.get(); }

  se::Stream* creation_stream() const { return creation_stream_; }

  bool is_tuple() { return !empty() && buffer_->on_host_shape().IsTuple(); }

  void AddToInput(xla::ShapeTree<xla::MaybeOwningDeviceMemory>::iterator* iterator,
                  xla::ShapeTree<xla::MaybeOwningDeviceMemory>::iterator& end,
                  xla::ExecutionInput* input);

  xla::StatusOr<ErlNifBinary> ToBinary();

  xla::Status BlockHostUntilReady();

 private:
  // The buffer's underlying device memory, we follow PjRt
  // and use an inlined vector. Inlined vectors behave exactly
  // the same as std::vector, but small sequences are stored
  // inline.
  absl::InlinedVector<se::DeviceMemoryBase, 1> device_memory_;

  // Buffer's underlying host/device shapes
  xla::Shape on_host_shape_;
  xla::Shape on_device_shape_;

  // Buffer semantics, see discussion above
  BufferType type_;

  // Buffer's device and client
  ExlaClient* client_;
  ExlaDevice* device_;

  // TODO(seanmor5): PjRt uses an event pool, internally a stack
  // of events. Research more
  // Buffer's definition event, we use events to immediately
  // return to the VM on executions and device transfers.
  std::unique_ptr<se::Event> definition_event_;

  // Buffer's creation stream
  se::Stream* creation_stream_;

  // Buffer's current state
  BufferState state_;

  // Used in AddToInput, depending on the buffer type
  void AddToInputAsDonated(xla::ShapeTree<xla::MaybeOwningDeviceMemory>::iterator* iterator,
                           xla::ShapeTree<xla::MaybeOwningDeviceMemory>::iterator& end,
                           xla::ExecutionInput* input);
  void AddToInputAsImmutable(xla::ShapeTree<xla::MaybeOwningDeviceMemory>::iterator* iterator,
                             xla::ShapeTree<xla::MaybeOwningDeviceMemory>::iterator& end);
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

  xla::StatusOr<std::vector<xla::ExecutionInput>> PopulateInputBuffers(absl::Span<ExlaBuffer* const> argument_handles);

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
