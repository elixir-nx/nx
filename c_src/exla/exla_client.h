#ifndef EXLA_CLIENT_H_
#define EXLA_CLIENT_H_

#include "tensorflow/compiler/xla/exla/exla_device.h"
#include "tensorflow/compiler/xla/exla/erts/erl_nif.h"

#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/types.h"

namespace exla {
  namespace se = tensorflow::se;

  /*
   * Wraps a ScopedShapedBuffer in a unique_ptr so the ScopedShapedBuffer is automatically
   * destructed when this object is destructed. In the future, will provide a place to
   * implement device transfers, etc.
   */
  // TODO: Add device attribute
  class ExlaBuffer {

  public:
    ExlaBuffer(std::unique_ptr<xla::ScopedShapedBuffer> buffer, bool zero_copy) : owned_buffer_(std::move(buffer)),
                                                                                  zero_copy_(zero_copy) {}
    ~ExlaBuffer() {
      if(zero_copy_) {
        if(donated_) {
          donated_buffer_->release();
        } else {
          owned_buffer_->release();
        }
      } else if(!zero_copy_ && donated_) {
        delete donated_buffer_;
      } else {}
    }

    xla::ScopedShapedBuffer* donate() {
      donated_buffer_ = owned_buffer_.release();
      donated_ = true;
      return donated_buffer_;
    }

    xla::ScopedShapedBuffer* buffer() { return owned_buffer_.get(); }

  private:
    // Owned Buffer, guaranteed destructed when this class is destructed
    std::unique_ptr<xla::ScopedShapedBuffer> owned_buffer_;
    // Used for donating this buffer to another function, like `Run`
    xla::ScopedShapedBuffer* donated_buffer_ = nullptr;
    // Was the bool created with a zero-copy transfer
    bool zero_copy_;
    // Has the buffer been donated?
    bool donated_ = false;

  };

  /*
   * There are a lot of resources that we need to keep track of, and it doesn't make sense
   * to pass references to all of them back and forth between the BEAM and NIFs. To avoid this
   * we implement an `ExlaClient` in the same spirit of: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/pjrt/pjrt_client.h
   * which wraps an `xla::LocalClient` and also resources that we can initialize and hold on client
   * creation.
   */
  class ExlaClient {
  public:

    explicit ExlaClient(xla::LocalClient* client,
                        int host_id,
                        std::vector<std::unique_ptr<ExlaDevice>> devices,
                        std::unique_ptr<se::DeviceMemoryAllocator> allocator,
                        std::unique_ptr<tensorflow::Allocator> host_memory_allocator,
                        std::unique_ptr<xla::GpuExecutableRunOptions> gpu_run_options);

    virtual ~ExlaClient() = default;

    xla::StatusOr<xla::ScopedShapedBuffer> Run(xla::LocalExecutable* executable,
                                               std::vector<ExlaBuffer*>& buffers,
                                               xla::ExecutableRunOptions& options);

    xla::StatusOr<ExlaBuffer*> BufferFromErlBin(const ErlNifBinary binary,
                                                const xla::Shape& shape,
                                                ExlaDevice* device);

    xla::StatusOr<ErlNifBinary> ErlBinFromBuffer(xla::ScopedShapedBuffer& buffer, ExlaDevice* device);

    xla::LocalClient* client() { return client_; }

    tensorflow::Allocator* host_memory_allocator() { return host_memory_allocator_.get(); }

    int host_id() { return host_id_; }

    se::DeviceMemoryAllocator* allocator() { return allocator_; }

    xla::GpuExecutableRunOptions* gpu_run_options() { return gpu_run_options_.get(); }

    int device_count() const { return devices_.size(); }

    const std::vector<std::unique_ptr<ExlaDevice>>& devices() const { return devices_; }

    exla::ExlaDevice* device(int id) { return devices_.at(id).get(); }

    /* tensorflow::thread::ThreadPool* h2d_transfer_pool() { return &h2d_transfer_pool_; } */

  private:
    xla::LocalClient* client_;
    std::unique_ptr<tensorflow::Allocator> host_memory_allocator_;
    int host_id_;
    se::DeviceMemoryAllocator* allocator_;
    std::unique_ptr<se::DeviceMemoryAllocator> owned_allocator_;
    std::unique_ptr<xla::GpuExecutableRunOptions> gpu_run_options_;
    std::vector<std::unique_ptr<ExlaDevice>> devices_;
    /* tensorflow::thread::ThreadPool h2d_transfer_pool_ = nullptr; */
  };

  // TODO: Separate into different device classes similar to PjRt
  xla::StatusOr<ExlaClient*> getHostClient(int num_replicas, int intra_op_parallelism_threads);
  xla::StatusOr<ExlaClient*> getCUDAClient(int num_replicas, int intra_op_parallelism_threads);
}

#endif
