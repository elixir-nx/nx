#include "tensorflow/compiler/xla/exla/exla_device.h"

namespace exla {

  ExlaDevice::ExlaDevice(int id,
                         se::StreamExecutor* executor,
                         xla::LocalClient* client) : id_(id),
                                                     executor_(executor),
                                                     client_(client) {
    compute_stream_ = std::make_unique<se::Stream>(executor);
    host_to_device_stream_ = std::make_unique<se::Stream>(executor);
    callback_stream_ = std::make_unique<se::Stream>(executor);
    device_to_host_stream_ = std::make_unique<se::Stream>(executor);
    compute_stream_->Init();
    host_to_device_stream_->Init();
    callback_stream_->Init();
    device_to_host_stream_->Init();
  }

  xla::Status ExlaDevice::SynchronizeAllActivity() {
    xla::Status status;
    status.Update(compute_stream_->BlockHostUntilDone());
    status.Update(callback_stream_->BlockHostUntilDone());
    bool ok = compute_stream_->parent()->SynchronizeAllActivity();
    if (!ok) {
      status.Update(xla::Unknown("SynchronizeAllActivity failed."));
    }
    return status;
  }

}  // namespace exla
