#include "runtime_callback_bridge.h"

#include <cstring>
#include <string>
#include <vector>

#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace ffi = xla::ffi;

namespace exla::callback_bridge {

Arg::Arg(const xla::ffi::AnyBuffer &buf) {
  dtype = buf.element_type();
  auto d = buf.dimensions();
  dims.assign(d.begin(), d.end());
  data = reinterpret_cast<const uint8_t *>(buf.untyped_data());
  size_bytes = buf.size_bytes();
}

OutputBuffer::OutputBuffer(const xla::ffi::AnyBuffer &buf) {
  data = static_cast<uint8_t *>(buf.untyped_data());
  size = ffi::ByteWidth(buf.element_type()) *
         static_cast<size_t>(buf.element_count());
}

} // namespace exla::callback_bridge

namespace {

ffi::Error exla_runtime_callback_impl(
    ffi::RemainingArgs args, ffi::Span<const int64_t> callback_id_words,
    uint64_t callback_id_size, uint64_t num_aliased_data_outputs,
    ffi::RemainingRets rets) {
  // We currently only support leading aliased outputs, but this is just
  // a way to make things easier. We can extend this to support index-based
  // aliasing in the future.

  // args[0] = callback_server_pid, args[1..] = leaf operands

  if (args.size() < 1) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "host callback missing callback server pid operand");
  }

  std::vector<exla::callback_bridge::Arg> inputs;
  inputs.reserve(args.size() - 1);

  auto callback_server_pid_or = args.get<ffi::AnyBuffer>(0);
  if (!callback_server_pid_or) {
    return callback_server_pid_or.error();
  }
  exla::callback_bridge::Arg callback_server_pid_arg(*callback_server_pid_or);

  for (size_t i = 1; i < args.size(); ++i) {
    auto buf_or = args.get<ffi::AnyBuffer>(i);
    if (!buf_or) {
      return buf_or.error();
    }

    inputs.push_back(exla::callback_bridge::Arg(*buf_or));
  }

  // rets[0 .. num_aliased_data_outputs-1] are aliased to the input operands;
  // the callback is not expected to fill them.
  // Non-aliased outputs (runtime_call results) start at
  // num_aliased_data_outputs.
  std::vector<exla::callback_bridge::OutputBuffer> outputs;
  size_t first_real_ret = static_cast<size_t>(num_aliased_data_outputs);
  outputs.reserve(rets.size() > first_real_ret ? rets.size() - first_real_ret
                                               : 0);

  for (size_t i = first_real_ret; i < rets.size(); ++i) {
    auto ret_or = rets.get<ffi::AnyBuffer>(i);
    if (!ret_or) {
      return ret_or.error();
    }

    ffi::Result<ffi::AnyBuffer> ret = *ret_or;
    outputs.push_back(exla::callback_bridge::OutputBuffer(*ret));
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
                                  .Attr<uint64_t>("num_aliased_data_outputs")
                                  .RemainingRets());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "exla_runtime_callback", "Host",
                         exla_runtime_callback);
