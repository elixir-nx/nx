#include "runtime_callback_bridge.h"

#include <cstring>
#include <string>
#include <vector>

#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace ffi = xla::ffi;

namespace {

// Shared helper that builds the lightweight input descriptions used by the
// Elixir callback bridge. The `prepare` functor is responsible for populating
// the data pointer and size_bytes fields on the Arg and can perform any
// platform-specific bookkeeping (e.g. device->host copies). Returns `true` on
// success and `false` on error, in which case `error_out` is populated.
template <typename PrepareFn>
bool BuildInputs(ffi::RemainingArgs args, PrepareFn &&prepare,
                 std::vector<exla::callback_bridge::Arg> &inputs,
                 ffi::Error *error_out) {
  inputs.clear();
  inputs.reserve(args.size());

  for (size_t i = 0; i < args.size(); ++i) {
    auto maybe_buf_or = args.get<ffi::AnyBuffer>(i);
    if (!maybe_buf_or) {
      if (error_out) {
        *error_out = maybe_buf_or.error();
      }
      return false;
    }

    ffi::AnyBuffer buf = *maybe_buf_or;

    exla::callback_bridge::Arg tensor;
    tensor.dtype = buf.element_type();

    auto dims = buf.dimensions();
    tensor.dims.assign(dims.begin(), dims.end());

    std::string error;
    if (!prepare(buf, tensor, error)) {
      if (error_out) {
        *error_out = ffi::Error(ffi::ErrorCode::kInternal, std::move(error));
      }
      return false;
    }

    inputs.push_back(std::move(tensor));
  }

  return true;
}

// Shared helper that builds output buffer descriptors. The `prepare` functor
// is responsible for assigning the host-visible buffer pointer and size and
// can keep track of any associated device pointers when running on
// accelerators. Returns `true` on success and `false` on error, in which case
// `error_out` is populated.
template <typename PrepareFn>
bool BuildOutputs(ffi::RemainingRets rets, PrepareFn &&prepare,
                  std::vector<exla::callback_bridge::OutputBuffer> &outputs,
                  ffi::Error *error_out) {
  outputs.clear();
  outputs.reserve(rets.size());

  for (size_t i = 0; i < rets.size(); ++i) {
    auto maybe_ret_or = rets.get<ffi::AnyBuffer>(i);
    if (!maybe_ret_or) {
      if (error_out) {
        *error_out = maybe_ret_or.error();
      }
      return false;
    }

    ffi::Result<ffi::AnyBuffer> ret = *maybe_ret_or;
    ffi::AnyBuffer out = *ret;

    exla::callback_bridge::OutputBuffer buf;

    std::string error;
    if (!prepare(out, buf, error)) {
      if (error_out) {
        *error_out = ffi::Error(ffi::ErrorCode::kInternal, std::move(error));
      }
      return false;
    }

    outputs.push_back(buf);
  }

  return true;
}

ffi::Error exla_runtime_callback_impl(
    ffi::RemainingArgs args, ffi::Span<const int64_t> callback_id_words,
    uint64_t callback_id_size,
    ffi::Span<const int64_t> callback_server_pid_words,
    uint64_t callback_server_pid_size, ffi::RemainingRets rets) {
  // Host implementation: we can pass the underlying host buffers directly to
  // the Elixir callback bridge without any extra copies.
  std::vector<exla::callback_bridge::Arg> inputs;
  ffi::Error error = ffi::Error::Success();
  if (!BuildInputs(
          args,
          [](ffi::AnyBuffer &buf, exla::callback_bridge::Arg &tensor,
             std::string &error) {
            (void)error;
            tensor.data = reinterpret_cast<const uint8_t *>(buf.untyped_data());
            tensor.size_bytes = buf.size_bytes();
            return true;
          },
          inputs, &error)) {
    return error;
  }

  // Prepare output buffer descriptors so the callback bridge can write results
  // directly into the final destination buffers.
  std::vector<exla::callback_bridge::OutputBuffer> outputs;
  if (!BuildOutputs(
          rets,
          [](ffi::AnyBuffer &out, exla::callback_bridge::OutputBuffer &buf,
             std::string &error) {
            (void)error;
            buf.data = static_cast<uint8_t *>(out.untyped_data());
            buf.size = ffi::ByteWidth(out.element_type()) *
                       static_cast<size_t>(out.element_count());
            return true;
          },
          outputs, &error)) {
    return error;
  }

  // Call back into Elixir through the bridge. On success, the bridge writes
  // results directly into the provided output buffers.
  exla::callback_bridge::Result result =
      exla::callback_bridge::InvokeRuntimeCallback(
          callback_id_words, callback_id_size, callback_server_pid_words,
          callback_server_pid_size, inputs, outputs);

  if (!result.ok) {
    return ffi::Error(ffi::ErrorCode::kInternal, result.error);
  }

  return ffi::Error::Success();
}

#ifdef CUDA_ENABLED

// CUDA implementation: inputs and outputs live in device memory. We copy
// inputs to host buffers before invoking the Elixir callback bridge and copy
// the callback outputs back to device buffers afterwards. This keeps the
// Elixir-side contract identical to the host implementation (receive binaries
// and return binaries) while still allowing computations to be compiled for
// the CUDA platform.
ffi::Error exla_runtime_callback_cuda_impl(
    CUstream stream, ffi::RemainingArgs args,
    ffi::Span<const int64_t> callback_id_words, uint64_t callback_id_size,
    ffi::Span<const int64_t> callback_server_pid_words,
    uint64_t callback_server_pid_size, ffi::RemainingRets rets) {
  // Host staging buffers for inputs.
  std::vector<std::vector<uint8_t>> host_inputs;
  std::vector<exla::callback_bridge::Arg> inputs;

  ffi::Error error = ffi::Error::Success();
  if (!BuildInputs(
          args,
          [&host_inputs, stream](ffi::AnyBuffer &buf,
                                 exla::callback_bridge::Arg &tensor,
                                 std::string &error) {
            size_t size = buf.size_bytes();
            host_inputs.emplace_back(size);
            auto &storage = host_inputs.back();

            void *device_ptr = buf.untyped_data();
            cudaError_t st = cudaMemcpyAsync(storage.data(), device_ptr, size,
                                             cudaMemcpyDeviceToHost, stream);
            if (st != cudaSuccess) {
              error = std::string("cudaMemcpyAsync device-to-host failed: ") +
                      cudaGetErrorString(st);
              return false;
            }

            tensor.data = storage.data();
            tensor.size_bytes = size;
            return true;
          },
          inputs, &error)) {
    return error;
  }

  // Ensure all device-to-host transfers are visible to the Elixir callback.
  cudaError_t sync_status = cudaStreamSynchronize(stream);
  if (sync_status != cudaSuccess) {
    return ffi::Error(
        ffi::ErrorCode::kInternal,
        std::string("cudaStreamSynchronize before callback failed: ") +
            cudaGetErrorString(sync_status));
  }

  // Host staging buffers for outputs and the corresponding device pointers.
  std::vector<std::vector<uint8_t>> host_outputs;
  std::vector<void *> device_output_ptrs;
  std::vector<exla::callback_bridge::OutputBuffer> outputs;

  if (!BuildOutputs(
          rets,
          [&host_outputs, &device_output_ptrs](
              ffi::AnyBuffer &out, exla::callback_bridge::OutputBuffer &buf,
              std::string &error) {
            (void)error;

            size_t size = ffi::ByteWidth(out.element_type()) *
                          static_cast<size_t>(out.element_count());

            device_output_ptrs.push_back(out.untyped_data());
            host_outputs.emplace_back(size);
            auto &storage = host_outputs.back();

            buf.data = storage.data();
            buf.size = size;
            return true;
          },
          outputs, &error)) {
    return error;
  }

  exla::callback_bridge::Result result =
      exla::callback_bridge::InvokeRuntimeCallback(
          callback_id_words, callback_id_size, callback_server_pid_words,
          callback_server_pid_size, inputs, outputs);

  if (!result.ok) {
    return ffi::Error(ffi::ErrorCode::kInternal, result.error);
  }

  // Copy callback outputs from host staging buffers back to device memory.
  for (size_t i = 0; i < outputs.size(); ++i) {
    void *device_ptr = device_output_ptrs[i];
    auto &storage = host_outputs[i];

    if (storage.empty()) {
      continue;
    }

    cudaError_t st = cudaMemcpyAsync(device_ptr, storage.data(), storage.size(),
                                     cudaMemcpyHostToDevice, stream);
    if (st != cudaSuccess) {
      return ffi::Error(ffi::ErrorCode::kInternal,
                        std::string("cudaMemcpyAsync host-to-device failed: ") +
                            cudaGetErrorString(st));
    }
  }

  // Ensure all host-to-device transfers complete before XLA continues.
  sync_status = cudaStreamSynchronize(stream);
  if (sync_status != cudaSuccess) {
    return ffi::Error(
        ffi::ErrorCode::kInternal,
        std::string("cudaStreamSynchronize after callback failed: ") +
            cudaGetErrorString(sync_status));
  }

  return ffi::Error::Success();
}

#endif // CUDA_ENABLED

} // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    exla_runtime_callback, exla_runtime_callback_impl,
    ffi::Ffi::Bind()
        .RemainingArgs()
        .Attr<ffi::Span<const int64_t>>("callback_id")
        .Attr<uint64_t>("callback_id_size")
        .Attr<ffi::Span<const int64_t>>("callback_server_pid")
        .Attr<uint64_t>("callback_server_pid_size")
        .RemainingRets());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "exla_runtime_callback", "Host",
                         exla_runtime_callback);

#ifdef CUDA_ENABLED

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

#endif // CUDA_ENABLED
