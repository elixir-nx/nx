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

ffi::Error exla_runtime_callback_cuda_impl(
    CUstream stream, ffi::RemainingArgs args,
    ffi::Span<const int64_t> callback_id_words, uint64_t callback_id_size,
    ffi::Span<const int64_t> callback_server_pid_words,
    uint64_t callback_server_pid_size, ffi::RemainingRets rets) {

  // Keep host buffers alive for the duration of the callback.
  std::vector<std::vector<uint8_t>> host_input_buffers;
  host_input_buffers.reserve(args.size());

  std::vector<exla::callback_bridge::Arg> inputs;
  inputs.reserve(args.size());

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
    inputs.push_back(std::move(tensor));
  }

  // Outputs: host staging buffers; bridge will write into these, then we H→D.
  std::vector<std::vector<uint8_t>> host_output_buffers;
  host_output_buffers.reserve(rets.size());

  std::vector<exla::callback_bridge::OutputBuffer> outputs;
  outputs.reserve(rets.size());

  std::vector<void *> device_output_ptrs;
  device_output_ptrs.reserve(rets.size());

  for (size_t i = 0; i < rets.size(); ++i) {
    auto maybe_ret_or = rets.get<ffi::AnyBuffer>(i);
    if (!maybe_ret_or) {
      return maybe_ret_or.error();
    }

    ffi::Result<ffi::AnyBuffer> ret = *maybe_ret_or;
    ffi::AnyBuffer out = *ret;

    size_t size = ffi::ByteWidth(out.element_type()) *
                  static_cast<size_t>(out.element_count());

    host_output_buffers.emplace_back(size);
    std::vector<uint8_t> &host_out = host_output_buffers.back();

    exla::callback_bridge::OutputBuffer obuf;
    obuf.data = host_out.data();
    obuf.size = size;
    outputs.push_back(obuf);

    device_output_ptrs.push_back(out.untyped_data());
  }

  exla::callback_bridge::Result result =
      exla::callback_bridge::InvokeRuntimeCallback(
          callback_id_words, callback_id_size, callback_server_pid_words,
          callback_server_pid_size, inputs, outputs);

  if (!result.ok) {
    return ffi::Error(ffi::ErrorCode::kInternal, result.error);
  }

  for (size_t i = 0; i < rets.size(); ++i) {
    size_t size = outputs[i].size;
    cudaError_t err =
        cudaMemcpyAsync(device_output_ptrs[i], host_output_buffers[i].data(),
                        size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
      return ffi::Error(ffi::ErrorCode::kInternal,
                        std::string("cudaMemcpyAsync H->D failed: ") +
                            cudaGetErrorString(err));
    }
  }

  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    exla_runtime_callback_cuda, exla_runtime_callback_cuda_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<CUstream>>()
        .RemainingArgs()
        .Attr<ffi::Span<const int64_t>>("callback_id")
        .Attr<uint64_t>("callback_id_size")
        .Attr<ffi::Span<const int64_t>>("callback_server_pid")
        .Attr<uint64_t>("callback_server_pid_size")
        .RemainingRets());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "exla_runtime_callback", "CUDA",
                         exla_runtime_callback_cuda);

#else  // CUDA_ENABLED

namespace {

ffi::Error exla_runtime_callback_cuda_stub(
    ffi::RemainingArgs, ffi::Span<const int64_t>, uint64_t,
    ffi::Span<const int64_t>, uint64_t, ffi::RemainingRets) {
  return ffi::Error(ffi::ErrorCode::kUnimplemented,
                    "EXLA was not compiled with CUDA support. This error means your EXLA compilation is out of sync with your libexla.so NIF.");
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    exla_runtime_callback_cuda, exla_runtime_callback_cuda_stub,
    ffi::Ffi::Bind()
        .RemainingArgs()
        .Attr<ffi::Span<const int64_t>>("callback_id")
        .Attr<uint64_t>("callback_id_size")
        .Attr<ffi::Span<const int64_t>>("callback_server_pid")
        .Attr<uint64_t>("callback_server_pid_size")
        .RemainingRets());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "exla_runtime_callback", "CUDA",
                         exla_runtime_callback_cuda);

#endif  // CUDA_ENABLED
