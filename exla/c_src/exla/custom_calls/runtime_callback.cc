#include "runtime_callback_bridge.h"

#include <cstring>
#include <vector>
#include <string>

#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace ffi = xla::ffi;

namespace {

exla::callback_bridge::Arg arg_from_buffer(const ffi::AnyBuffer &buf) {
  exla::callback_bridge::Arg tensor;
  tensor.dtype = buf.element_type();

  auto dims = buf.dimensions();
  tensor.dims.assign(dims.begin(), dims.end());

  tensor.data = reinterpret_cast<const uint8_t *>(buf.untyped_data());
  tensor.size_bytes = buf.size_bytes();

  return tensor;
}

exla::callback_bridge::OutputBuffer output_buffer_from(const ffi::AnyBuffer &out) {
  exla::callback_bridge::OutputBuffer buf;
  buf.data = static_cast<uint8_t *>(out.untyped_data());
  buf.size = ffi::ByteWidth(out.element_type()) *
             static_cast<size_t>(out.element_count());
  return buf;
}

ffi::Error
exla_runtime_callback_impl(ffi::RemainingArgs args,
                           ffi::Span<const int64_t> callback_id_words,
                           uint64_t callback_id_size, ffi::RemainingRets rets) {
  if (args.size() < 2) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "host callback missing token and callback server pid operands");
  }

  std::vector<exla::callback_bridge::Arg> inputs;
  inputs.reserve(args.size() - 2);
  exla::callback_bridge::Arg callback_server_pid_arg;

  for (size_t i = 0; i < args.size(); ++i) {
    auto maybe_buf_or = args.get<ffi::AnyBuffer>(i);
    if (!maybe_buf_or) {
      return maybe_buf_or.error();
    }

    ffi::AnyBuffer buf = *maybe_buf_or;

    if (i == 0) {
      if (buf.element_type() != xla::ffi::DataType::TOKEN) {
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "host callback operand 0 must be a token");
      }
      continue;
    }

    exla::callback_bridge::Arg tensor = arg_from_buffer(buf);

    if (i == 1) {
      callback_server_pid_arg = std::move(tensor);
    } else {
      inputs.push_back(std::move(tensor));
    }
  }

  // rets[0] is the output token produced by the custom call itself.
  // Elixir only fills tensor outputs starting at rets[1].
  std::vector<exla::callback_bridge::OutputBuffer> outputs;
  outputs.reserve(rets.size() > 0 ? rets.size() - 1 : 0);

  for (size_t i = 1; i < rets.size(); ++i) {
    auto maybe_ret_or = rets.get<ffi::AnyBuffer>(i);
    if (!maybe_ret_or) {
      return maybe_ret_or.error();
    }

    ffi::Result<ffi::AnyBuffer> ret = *maybe_ret_or;
    outputs.push_back(output_buffer_from(*ret));
  }

  exla::callback_bridge::Result result =
      exla::callback_bridge::InvokeRuntimeCallback(
          callback_id_words, callback_id_size, callback_server_pid_arg, inputs,
          outputs);

  if (!result.ok) {
    return ffi::Error(ffi::ErrorCode::kInternal, result.error);
  }

  return ffi::Error::Success();
}

} // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(exla_runtime_callback, exla_runtime_callback_impl,
                              ffi::Ffi::Bind()
                                  .RemainingArgs()
                                  .Attr<ffi::Span<const int64_t>>("callback_id")
                                  .Attr<uint64_t>("callback_id_size")
                                  .RemainingRets());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "exla_runtime_callback", "Host",
                         exla_runtime_callback);
