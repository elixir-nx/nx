#include "tensorflow/compiler/xla/exla/exla_client.h"

namespace exla {

  xla::StatusOr<xla::LocalClient*> GetCpuClient() {
    // TODO: Handle StatusOr
    stream_executor::Platform *platform = xla::PlatformUtil::GetPlatform("Host").ConsumeValueOrDie();
    if(platform->VisibleDeviceCount() <= 0){
      return xla::FailedPrecondition("CPU platform has no visible devices.");
    }

    xla::LocalClientOptions options;
    options.set_platform(platform);

    xla::StatusOr<xla::LocalClient*> client = xla::ClientLibrary::GetOrCreateLocalClient(options);

    // TODO: Individual device configuration similar to: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/pjrt/cpu_device.cc
    return client;
  }

  xla::StatusOr<xla::LocalClient*> GetGpuClient() {
    // TODO: Handle StatusOr
    stream_executor::Platform *platform = xla::PlatformUtil::GetPlatform("CUDA").ConsumeValueOrDie();
    if(platform->VisibleDeviceCount() <= 0){
      return xla::FailedPrecondition("CUDA Platform has no visible devices.");
    }

    xla::LocalClientOptions options;
    options.set_platform(platform);

    xla::StatusOr<xla::LocalClient*> client = xla::ClientLibrary::GetOrCreateLocalClient(options);

    return client;
  }
}
