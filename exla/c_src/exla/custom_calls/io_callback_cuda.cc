#include "runtime_callback_bridge.h"

#include <cstring>
#include <vector>

#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace ffi = xla::ffi;

#ifdef CUDA_ENABLED

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

ffi::Error exla_io_callback_cuda_impl(
    CUstream stream, ffi::RemainingArgs args,
    ffi::Span<const int64_t> callback_id_words, uint64_t callback_id_size,
    ffi::RemainingRets /* rets */) {
  if (args.size() == 0) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "io_callback missing callback server pid operand");
  }

  std::vector<std::vector<uint8_t>> host_input_buffers;
  host_input_buffers.reserve(args.size());

  std::vector<exla::callback_bridge::Arg> inputs;
  inputs.reserve(args.size() - 1);
  exla::callback_bridge::Arg callback_server_pid_arg;

  for (size_t i = 0; i < args.size(); ++i) {
    auto maybe_buf_or = args.get<ffi::AnyBuffer>(i);
    if (!maybe_buf_or) {
      return maybe_buf_or.error();
    }

    ffi::AnyBuffer buf = *maybe_buf_or;
    size_t size_bytes = buf.size_bytes();

    host_input_buffers.emplace_back(size_bytes);
    std::vector<uint8_t> &host_buf = host_input_buffers.back();

    cudaError_t err = cudaMemcpyAsync(host_buf.data(), buf.untyped_data(),
                                      size_bytes, cudaMemcpyDeviceToHost,
                                      stream);
    if (err != cudaSuccess) {
      return ffi::Error(ffi::ErrorCode::kInternal,
                        std::string("cudaMemcpyAsync D->H failed: ") +
                            cudaGetErrorString(err));
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
      return ffi::Error(ffi::ErrorCode::kInternal,
                        std::string("cudaStreamSynchronize failed: ") +
                            cudaGetErrorString(err));
    }

    exla::callback_bridge::Arg tensor;
    tensor.dtype = buf.element_type();
    auto dims = buf.dimensions();
    tensor.dims.assign(dims.begin(), dims.end());
    tensor.data = host_buf.data();
    tensor.size_bytes = size_bytes;

    if (i == 0) {
      callback_server_pid_arg = std::move(tensor);
    } else {
      inputs.push_back(std::move(tensor));
    }
  }

  // Result buffers are aliased to input buffers via output_operand_aliases —
  // we must NOT touch them. The Elixir callback is a pure side effect.
  exla::callback_bridge::Result result =
      exla::callback_bridge::InvokeIoCallback(
          callback_id_words, callback_id_size, callback_server_pid_arg, inputs);

  if (!result.ok) {
    return ffi::Error(ffi::ErrorCode::kInternal, result.error);
  }

  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    exla_io_callback_cuda, exla_io_callback_cuda_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<CUstream>>()
        .RemainingArgs()
        .Attr<ffi::Span<const int64_t>>("callback_id")
        .Attr<uint64_t>("callback_id_size")
        .RemainingRets());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "exla_io_callback", "CUDA",
                         exla_io_callback_cuda);

#else  // CUDA_ENABLED

namespace {

ffi::Error exla_io_callback_cuda_stub(
    ffi::RemainingArgs, ffi::Span<const int64_t>, uint64_t,
    ffi::RemainingRets) {
  return ffi::Error(ffi::ErrorCode::kUnimplemented,
                    "EXLA was not compiled with CUDA support. This error means your EXLA compilation is out of sync with your libexla.so NIF.");
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    exla_io_callback_cuda, exla_io_callback_cuda_stub,
    ffi::Ffi::Bind()
        .RemainingArgs()
        .Attr<ffi::Span<const int64_t>>("callback_id")
        .Attr<uint64_t>("callback_id_size")
        .RemainingRets());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "exla_io_callback", "CUDA",
                         exla_io_callback_cuda);

#endif  // CUDA_ENABLED
