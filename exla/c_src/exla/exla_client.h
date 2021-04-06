#ifndef EXLA_CLIENT_H_
#define EXLA_CLIENT_H_

#include <memory>
#include <vector>
#include <utility>

#include "tensorflow/compiler/xla/exla/exla_device.h"
#include "tensorflow/compiler/xla/exla/exla_nif_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/status.h"

// The implementations in this module are designed after implementations
// in the XLA runtime, PjRt. Deviations are made where it makes sense
// to work better with the VM.

namespace exla {

namespace se = tensorflow::se;

class ExlaClient;

// Representation of an on-device buffer used during computations.
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
             const xla::Shape& on_host_shape,
             const xla::Shape& on_device_shape,
             ExlaDevice* device,
             ExlaClient* client,
             BufferType type)
            : device_memory_(device_memory.begin(), device_memory.end()),
              on_host_shape_(on_host_shape),
              on_device_shape_(on_device_shape),
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
  bool empty() {
    return state_ == BufferState::kDeallocated ||
           state_ == BufferState::kWaiting ||
           state_ == BufferState::kDonated;
  }

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
  const absl::InlinedVector<se::DeviceMemoryBase, 1> device_memory() const {
    return device_memory_;
  }

  // Returns true if the underlying memory has a tuple shape.
  bool is_tuple() { return on_host_shape_.IsTuple(); }

  // Adds this buffer as an input to a computation. Inputs can
  // either be donated or immutable. In the case of an immutable input,
  // the caller is responsible for deallocating the buffer at the appropriate
  // time. Donated inputs are deallocated when the computation finishes.
  xla::Status AddToInput(xla::ShapeTree<xla::MaybeOwningDeviceMemory>::iterator* iterator,
                         const xla::ShapeTree<xla::MaybeOwningDeviceMemory>::iterator& end,
                         xla::ExecutionInput* input);

  // Converts the underlying buffer to a VM binary to be returned back
  // to the VM. This is a non-destructive operation. The buffer either
  // has to be explicitly deallocated, or deallocated when the object
  // goes out of scope.
  xla::StatusOr<ERL_NIF_TERM> ToBinary(ErlNifEnv* env);

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
  static ExlaBuffer*
  FromScopedShapedBuffer(xla::ScopedShapedBuffer* shaped_buffer,
                         ExlaDevice* device,
                         ExlaClient* client,
                         BufferType type);

  // Decomposes the given buffer to an Erlang VM term. The term is either
  // a binary if the buffer has an array-like shape or a list of the
  // buffer has a tuple shape. If `keep_on_device` is true, the term
  // will be a reference or a list of references to the underlying buffer(s).
  static xla::StatusOr<ERL_NIF_TERM>
  DecomposeBufferToTerm(ErlNifEnv* env,
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
                           const xla::ShapeTree<xla::MaybeOwningDeviceMemory>::iterator& end,
                           xla::ExecutionInput* input);

  // Adds this buffer to an input's buffers without transferring ownership
  // of the underlying memory to the input buffer.
  void AddToInputAsImmutable(xla::ShapeTree<xla::MaybeOwningDeviceMemory>::iterator* iterator,
                             const xla::ShapeTree<xla::MaybeOwningDeviceMemory>::iterator& end);
};


// Provides a convenient interface for working with executables. We wrap
// potentially multiple xla::LocalExecutables into a single interface for
// convenience. This class also keeps track of an executables device assignment,
// available devices, client, and logical device IDs.
class ExlaExecutable {
 public:
  ExlaExecutable(std::vector<std::unique_ptr<xla::LocalExecutable>> executables,
                 std::shared_ptr<xla::DeviceAssignment> device_assignment,
                 ExlaClient* client);

  // Returns executable's compiling client.
  ExlaClient* client() { return client_; }

  // Returns number of replicas specified in the executable.
  int num_replicas() const {
    return executables_.at(0)->build_options().num_replicas();
  }

  // Returns number of partition specified in the executable.
  int num_partitions() const {
    return executables_.at(0)->build_options().num_replicas();
  }

  // Returns a vector of underlying XLA executables.
  const std::vector<std::shared_ptr<xla::LocalExecutable>>& executables() const {
    return executables_;
  }

  // Returns the executable device assignment, if there is one
  const xla::DeviceAssignment& device_assignment() const {
    return *device_assignment_;
  }

  // Deletes the underlying executables
  void Delete() { executables_.clear(); }

  // Populates input buffers from the given ExlaBuffers. See the note in
  // the ExlaBuffer class for a discussion of how ownership is transferred
  // from the given argument handles to input buffers.
  xla::StatusOr<std::vector<xla::ExecutionInput>>
  PopulateInputBuffers(absl::Span<ExlaBuffer* const> argument_handles);

  // Runs the executable with the given configuration options. If
  // `keep_on_device` is true, the resulting term will be a reference
  // of a list of references to the underlying buffer(s). Otherwise,
  // the resulting buffer is decomposed to an Erlang term and the device
  // memory is deallocated.
  xla::StatusOr<ERL_NIF_TERM> Run(ErlNifEnv* env,
                                  ERL_NIF_TERM arguments,
                                  xla::Shape& output_shape,
                                  int replica,
                                  int partition,
                                  int run_id,
                                  int rng_seed,
                                  int launch_id,
                                  bool async_run,
                                  bool keep_on_device);

 private:
  ExlaClient* client_;
  std::vector<std::shared_ptr<xla::LocalExecutable>> executables_;
  std::shared_ptr<xla::DeviceAssignment> device_assignment_;
};

// Encapsulates an xla::LocalClient, which provides an interface for
// interacting with one or many devices.
class ExlaClient {
 public:
  explicit ExlaClient(xla::LocalClient* client,
                      int host_id,
                      std::vector<std::unique_ptr<ExlaDevice>> devices,
                      std::unique_ptr<se::DeviceMemoryAllocator> allocator,
                      std::unique_ptr<tensorflow::Allocator> host_memory_allocator,
                      std::unique_ptr<xla::gpu::GpuExecutableRunOptions> gpu_run_options);


  virtual ~ExlaClient() = default;

  // Compiles the given computation with the given argument layouts
  // and build options. If `compile_portable_executable` is set to
  // true, the resulting executable can be executed on any of the
  // local devices compatible with this client.
  xla::StatusOr<ExlaExecutable*>
  Compile(const xla::XlaComputation&,
          std::vector<xla::Shape*> argument_layouts,
          xla::ExecutableBuildOptions& build_options,
          bool compile_portable_executable);

  // Copies the underlying binary to the given device. `transfer_for_run`
  // is a flag used to indicate whether or not the resulting buffer should
  // be a temporary/zero copy buffer or a long-lived reference buffer. The
  // device transfer is non-destructive with respect to the binary because
  // the VM expects to be able to garbage collect the binary later on.
  xla::StatusOr<ExlaBuffer*> BufferFromBinary(const ErlNifBinary& binary,
                                              xla::Shape& shape,
                                              ExlaDevice* device,
                                              bool transfer_for_run,
                                              bool async_run);

  // Returns the client's default device assignment from the
  // given replica and partition account. This is used when
  // no device assignment is specified for compiling an executable.
  xla::StatusOr<xla::DeviceAssignment>
  GetDefaultDeviceAssignment(int num_replicas,
                             int num_partitions);

  // Returns a pointer to the underlying local client.
  xla::LocalClient* client() { return client_; }

  // Returns the client's host memory allocator. The host memory allocator
  // is used to stage host-to-device transfers. On CPU the default host-memory
  // allocator is the ExlaErtsAllocator which allocates directly in the VM.
  // On GPU we use the TF Device Host Allocator, with a plan to integrate
  // one that works well the VM.
  tensorflow::Allocator* host_memory_allocator() {
    return host_memory_allocator_.get();
  }

  // Returns the underlying platform ID.
  int host_id() { return host_id_; }

  // Returns the client's allocator. The allocator is used to allocate
  // memory on the device. We use the platform-default on Host platforms.
  // On GPU platforms, we use the same implementation as PjRt. The GPU
  // allocator is a multi-device allocator which wraps multiple TF BFC
  // allocators and streams into a single interface.
  se::DeviceMemoryAllocator* allocator() { return allocator_; }

  // Returns client's default GPU run options.
  xla::gpu::GpuExecutableRunOptions* gpu_run_options() {
    return gpu_run_options_.get();
  }

  // Returns the number of devices accessible to this client.
  int device_count() const { return devices_.size(); }

  // Returns a vector of all of the devices accessible to this client.
  const std::vector<std::unique_ptr<ExlaDevice>>& devices() const {
    return devices_;
  }

  // Returns a single device from the given `id`.
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
                                        const char* platform_name,
                                        double memory_fraction,
                                        bool preallocate);
}  // namespace exla

#endif
