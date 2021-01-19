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

  // The current state of this buffer, used to check if this
  // buffer can be written to, deallocated, or used in another
  // computation.
  enum class BufferState {
    // The buffer is in a valid and useable state
    kValid,

    // The buffer has already been deallocated
    kDeallocated,

    // The buffer is waiting on it's definition event
    kWaiting,

    // The buffer has been donated
    kDonated,

    // The buffer is in an error state
    kError
  };

  ExlaBuffer(absl::Span<se::DeviceMemoryBase const> device_memory,
             xla::Shape& on_host_shape,
             xla::Shape& on_device_shape,
             ExlaDevice* device,
             ExlaClient* client,
             BufferType type) : device_memory_(device_memory.begin(), device_memory.end()),
                                on_host_shape_(std::move(on_host_shape)),
                                on_device_shape_(std::move(on_device_shape)),
                                device_(device),
                                client_(client),
                                type_(type),
                                state_(BufferState::kValid) {}

  ~ExlaBuffer() {
    Deallocate();
  }

  // Returns true if the underlying buffer is empty. The buffer is considered
  // empty if (1) it no longer has ownership of it's underlying device memory,
  // or (2) the underlying device memory has not yet been written to. This is
  // the case if the buffer is in a deallocated or donated (case 1), or it is
  // in a waiting state (case 2).
  bool empty() { return state_ == BufferState::kDeallocated || state_ == BufferState::kWaiting || state_ == BufferState::kDonated; }

  // Returns the underlying host shape of the buffer.
  const xla::Shape on_host_shape() { return on_host_shape_; }

  // Returns the underlying device shape of the buffer.
  const xla::Shape on_device_shape() { return on_device_shape_; }

  // Returns this buffer's device.
  ExlaDevice* device() { return device_; }

  // Returns this buffer's client.
  ExlaClient* client() { return client_; }

  // Returns the buffer's type.
  BufferType type() { return type_; }

  // Returns the underlying vector of device memory.
  const absl::InlinedVector<se::DeviceMemoryBase, 1> device_memory() const { return device_memory_; }

  // Returns true if the underlying memory has a tuple shape.
  bool is_tuple() { return on_host_shape_.IsTuple(); }

  // Adds this buffer as an input to a computation. Inputs can
  // either be donated or immutable. In the case of an immutable input,
  // the caller is responsible for deallocating the buffer at the appropriate
  // time. Donated inputs are deallocated when the computation finishes.
  void AddToInput(xla::ShapeTree<xla::MaybeOwningDeviceMemory>::iterator* iterator,
                  xla::ShapeTree<xla::MaybeOwningDeviceMemory>::iterator& end,
                  xla::ExecutionInput* input);

  // Converts the underlying buffer to a VM binary to be returned back
  // to the VM. This is a non-destructive operation. The buffer either
  // has to be explicitly deallocated, or deallocated when the object
  // goes out of scope.
  xla::StatusOr<ErlNifBinary> ToBinary();

  // Deallocates the underlying device memory and returns a success
  // status or an error status. Only temporary and reference tensors
  // can be explicitly deallocated. Zero-copy deallocation releases
  // the underlying device memory to the VM to garbage collect. If the
  // tensor is already deallocated, or waiting for it's buffers to be
  // populated, returns an error.
  xla::Status Deallocate();

  // Releases ownership of the underlying device memory. The underlying
  // memory is considered "deallocated" becasue it is no longer assumed
  // as safe to use. It is the responsibility of the caller to deallocate
  // the underlying memory. This is used mostly when creating buffers
  // from zero-copy transfers to allow the VM to garbage collect the
  // underlying device memory.
  void ReleaseMemoryOwnership();

  // Returns a view of the buffer as a shaped buffer. This is a convenience
  // for performing transfers between device and host. Underlying device
  // memory is still owned by this buffer, and must be deallocated either
  // explicitly or when the buffer is destructed.
  xla::ShapedBuffer AsShapedBuffer();

  // Transfers ownership of the underlying device memory from the scoped
  // shaped buffer to this buffer.
  void WriteToBuffer(xla::ScopedShapedBuffer* shaped_buffer);

  // Creates a new ExlaBuffer from the scoped shaped buffer with the given
  // device, client, and type. The ExlaBuffer takes ownership of the
  // underlying device memory and is responsible for deallocating the memory
  // upon destruction or with an explicit deallocation.
  static ExlaBuffer* FromScopedShapedBuffer(xla::ScopedShapedBuffer* shaped_buffer,
                                            ExlaDevice* device,
                                            ExlaClient* client,
                                            BufferType type);

  // Decomposes the given buffer to an Erlang VM term. The term is either
  // a binary if the buffer has an array-like shape or a list of the
  // buffer has a tuple shape. If `keep_on_device` is true, the term
  // will be a reference or a list of references to the underlying buffer(s).
  static xla::StatusOr<ERL_NIF_TERM> DecomposeBufferToTerm(ErlNifEnv* env,
                                                           ExlaBuffer* buffer,
                                                           bool keep_on_device);
 private:
  // Buffer's underlying device memory, we follow PjRt
  // and use an inlined vector. Inlined vectors behave exactly
  // the same as std::vector, but small sequences are stored
  // inline without heap allocation. We decompose tuples
  // into multiple buffers, so we can assume a default capacity
  // of 1.
  absl::InlinedVector<se::DeviceMemoryBase, 1> device_memory_;

  // Buffer's underlying host/device shapes
  xla::Shape on_host_shape_;
  xla::Shape on_device_shape_;

  // Buffer semantics, see discussion above
  BufferType type_;

  // Buffer's device and client
  ExlaClient* client_;
  ExlaDevice* device_;

  // Buffer's current state
  BufferState state_;

  // Donates this buffer to the given xla::ExecutionInput. The input takes
  // ownership of the underlying buffer, and is responsible for deallocating
  // the underlying device memory. Because of that, this buffer is no longer
  // considered valid.
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

  xla::StatusOr<ExlaBuffer*> BufferFromBinary(const ErlNifBinary& binary,
                                              xla::Shape& shape,
                                              ExlaDevice* device,
                                              bool transfer_for_run);

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
