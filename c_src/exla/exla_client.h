#ifndef EXLA_CLIENT_H_
#define EXLA_CLIENT_H_

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
                        std::unique_ptr<se::DeviceMemoryAllocator> allocator,
                        std::unique_ptr<tensorflow::Allocator> host_memory_allocator,
                        std::unique_ptr<xla::GpuExecutableRunOptions> gpu_run_options);

    virtual ~ExlaClient() = default;

    xla::LocalClient* client() { return client_; }

    tensorflow::Allocator* host_memory_allocator() { return host_memory_allocator_.get(); }

    int host_id() { return host_id_; }

    se::DeviceMemoryAllocator* allocator() { return allocator_; }

    xla::GpuExecutableRunOptions* gpu_run_options() { return gpu_run_options_.get(); }

    /* tensorflow::thread::ThreadPool* h2d_transfer_pool() { return &h2d_transfer_pool_; } */

  private:
    xla::LocalClient* client_;
    std::unique_ptr<tensorflow::Allocator> host_memory_allocator_;
    int host_id_;
    se::DeviceMemoryAllocator* allocator_;
    std::unique_ptr<se::DeviceMemoryAllocator> owned_allocator_;
    std::unique_ptr<xla::GpuExecutableRunOptions> gpu_run_options_;
    /* tensorflow::thread::ThreadPool h2d_transfer_pool_ = nullptr; */
  };

  // TODO: Separate into different device classes similar to PjRt
  xla::StatusOr<ExlaClient*> GetCpuClient();
  xla::StatusOr<ExlaClient*> GetGpuClient();
}

#endif
