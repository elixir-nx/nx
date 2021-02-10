#ifndef EXLA_DEVICE_H_
#define EXLA_DEVICE_H_

#include <memory>

#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace exla {

namespace se = tensorflow::se;

// Wrapper around a single XLA device.

class ExlaDevice {
 public:
    explicit ExlaDevice(int id,
                        se::StreamExecutor* executor,
                        xla::LocalClient* client);

    virtual ~ExlaDevice() = default;

    int id() const {
      return id_;
    }

    // Returns this device's device ordinal.
    int device_ordinal() const { return executor_->device_ordinal(); }

    // Returns this device's stream executor.
    se::StreamExecutor* executor() const { return executor_; }

    // Returns this device's client.
    xla::LocalClient* client() const {
      return client_;
    }

    // Returns this device's compute stream. Compute streams
    // are used for running computations.
    se::Stream* compute_stream() const {
      return compute_stream_.get();
    }

    // Returns this device's host-to-device stream. Host-to-device
    // streams are used for host-to-device transfers.
    se::Stream* host_to_device_stream() const {
      return host_to_device_stream_.get();
    }

    // Returns this device's device-to-host stream. Device-to-host
    // streams are used for device-to-host transfers.
    se::Stream* device_to_host_stream() const {
      return device_to_host_stream_.get();
    }

    // Returns this device's callback stream. Callback streams
    // are used for callbacks from host to device.
    se::Stream* callback_stream() const {
      return callback_stream_.get();
    }

    // See PjRt implementation: tensorflow/compiler/xla/pjrt/local_device_state.cc
    // This function synchronizes streams on this device
    xla::Status SynchronizeAllActivity();

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
