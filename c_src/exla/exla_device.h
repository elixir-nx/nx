#ifndef EXLA_DEVICE_H_
#define EXLA_DEVICE_H_

#include <memory>

#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace exla {

/*
 * Wrapper around a single XLA device. This is mainly used to manage computation streams in a multi-device
 * setting; however, it will prove useful later on for managing device transfers and more. This
 * class combines `PjRtDevice` and `LocalDeviceState` in the Python implementation of XLA. Devices
 * are owned by a single client object. Clients can have multiple devices.
 */

namespace se = tensorflow::se;

class ExlaDevice {
 public:
    explicit ExlaDevice(int id,
                        se::StreamExecutor* executor,
                        xla::LocalClient* client);

    virtual ~ExlaDevice() = default;

    int id() const {
      return id_;
    }

    int device_ordinal() const { return executor_->device_ordinal(); }

    se::StreamExecutor* executor() const { return executor_; }

    xla::LocalClient* client() const {
      return client_;
    }

    se::Stream* compute_stream() const {
      return compute_stream_.get();
    }

    se::Stream* host_to_device_stream() const {
      return host_to_device_stream_.get();
    }

    // TODO(seanmor5): Make multiple of these
    se::Stream* device_to_host_stream() const {
      return device_to_host_stream_.get();
    }

    se::Stream* callback_stream() const {
      return callback_stream_.get();
    }

 private:
    int id_;
    se::StreamExecutor* const executor_;
    xla::LocalClient* const client_;
    std::unique_ptr<se::Stream> compute_stream_;
    std::unique_ptr<se::Stream> host_to_device_stream_;
    std::unique_ptr<se::Stream> device_to_host_stream_;
    std::unique_ptr<se::Stream> callback_stream_;
};
}  // namespace exla

#endif
