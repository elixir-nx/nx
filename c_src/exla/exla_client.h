#ifndef EXLA_CLIENT_H_
#define EXLA_CLIENT_H_

#include "tensorflow/compiler/xla/exla/exla_device.h"
#include "tensorflow/compiler/xla/exla/exla_nif_util.h"

#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/status.h"

namespace exla {
  namespace se = tensorflow::se;

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
    ~ExlaBuffer() {
      if(this->empty()) {
        return;
      } else {
        if(zero_copy_ && buffer_ != nullptr) {
          buffer_->release();
          buffer_ = nullptr;
        } else if(!zero_copy_ && buffer_ != nullptr) {
          delete buffer_;
          buffer_ = nullptr;
        } else {
          return;
        }
      }
    }

    xla::StatusOr<std::vector<ExlaBuffer*>> DecomposeTuple() {
      if(!is_tuple()) {
        return tensorflow::errors::FailedPrecondition("Buffer is not a Tuple.");
      }

      std::vector<ExlaBuffer*> buffers;
      long long int tuple_elements = xla::ShapeUtil::TupleElementCount(on_device_shape());
      buffers.reserve(tuple_elements);
      for(int i=0;i<tuple_elements;i++) {
        xla::ScopedShapedBuffer* sub_buffer = new xla::ScopedShapedBuffer(std::move(buffer_->TakeSubTree({i})));
        buffers.push_back(new ExlaBuffer(sub_buffer, device_, false));
      }

      return buffers;
    }

    bool empty() { return buffer_ == nullptr; }

    xla::Status deallocate() {
      if(this->empty()) {
        return tensorflow::errors::Aborted("Attempt to deallocate already deallocated buffer.");
      } else {
        if(zero_copy_ && buffer_ != nullptr) {
          buffer_->release();
          buffer_ = nullptr;
        } else if(!zero_copy_ && buffer_ != nullptr) {
          delete buffer_;
          buffer_ = nullptr;
        } else {
          return tensorflow::errors::Aborted("Attempt to deallocate already deallocated buffer.");
        }
      }

      return tensorflow::Status::OK();
    }

    const xla::Shape on_host_shape() { return buffer_->on_host_shape(); }
    const xla::Shape on_device_shape() { return buffer_->on_device_shape(); }

    xla::ScopedShapedBuffer* buffer() { return buffer_; }

    ExlaDevice* device() { return device_; }

    bool is_tuple() { return !empty() && buffer_->on_host_shape().IsTuple(); }

  private:
    // Used for donating this buffer to another function, like `Run`
    xla::ScopedShapedBuffer* buffer_;
    // Was the bool created with a zero-copy transfer
    bool zero_copy_;
    // Buffer's device
    ExlaDevice* device_;

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
                                               std::vector<std::pair<exla::ExlaBuffer*, exla::ExlaBuffer**>>& buffers,
                                               xla::ExecutableRunOptions& options);

    xla::StatusOr<ExlaBuffer*> BufferFromErlBin(const ErlNifBinary binary,
                                                const xla::Shape& shape,
                                                ExlaDevice* device);

    ERL_NIF_TERM DecomposeBuffer(ErlNifEnv* env, ExlaBuffer* buffer);

    xla::StatusOr<ErlNifBinary> ErlBinFromBuffer(ExlaBuffer* buffer);

    xla::StatusOr<ERL_NIF_TERM> ErlListFromBuffer(ErlNifEnv* env, ExlaBuffer* buffer);

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
