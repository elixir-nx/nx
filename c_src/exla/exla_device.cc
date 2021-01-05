#include "tensorflow/compiler/xla/exla/exla_device.h"
#include <memory>

namespace exla {

  ExlaDevice::ExlaDevice(int id,
                         se::StreamExecutor* executor,
                         xla::LocalClient* client) : id_(id),
                                                     executor_(executor),
                                                     client_(client) {
    compute_stream_ = absl::make_unique<se::Stream>(executor);
    host_to_device_stream_ = absl::make_unique<se::Stream>(executor);
    callback_stream_ = absl::make_unique<se::Stream>(executor);
    device_to_host_stream_ = absl::make_unique<se::Stream>(executor);
    compute_stream_->Init();
    host_to_device_stream_->Init();
    callback_stream_->Init();
    device_to_host_stream_->Init();
  }

  xla::Status ExlaDevice::TransferBinaryToInfeed(const ErlNifBinary bin, const xla::Shape& shape) const {
    xla::BorrowingLiteral literal(const_cast<char*>(reinterpret_cast<char*>(bin.data)), shape);
    return this->client()->TransferToInfeedLocal(literal, this->device_ordinal());
  }

  xla::StatusOr<ErlNifBinary> ExlaDevice::TransferBinaryFromOutfeed(const xla::Shape& shape) const {
    EXLA_ASSIGN_OR_RETURN(xla::Literal literal,
      this->client()->TransferFromOutfeedLocal(shape, this->device_ordinal()));

    // Allocate enough space for the binary
    int64 size = literal.size_bytes();
    ErlNifBinary binary;
    enif_alloc_binary(size, &binary);

    // No need to copy, just move to the underlying bytes in memory
    void *src_mem = const_cast<void*>(literal.untyped_data());
    std::memmove(binary.data, src_mem, size);

    return binary;
  }

}  // namespace exla
