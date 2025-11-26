#include "elixir_callback_bridge.h"

#include <cstring>
#include <vector>
#include <string>

#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace ffi = xla::ffi;

namespace {

ffi::Error exla_elixir_callback_impl(ffi::RemainingArgs args,
                                     ffi::RemainingRets rets) {
  if (args.size() == 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "exla_elixir_callback expects at least one argument");
  }

  // The first argument is a scalar S64 tensor carrying the callback id.
  auto id_buf_or = args.get<ffi::AnyBuffer>(0);
  if (!id_buf_or) {
    return id_buf_or.error();
  }

  ffi::AnyBuffer id_buf = *id_buf_or;

  if (id_buf.element_count() != 1 ||
      id_buf.element_type() != ffi::DataType::S64) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "exla_elixir_callback callback id must be scalar s64");
  }

  int64_t callback_id = id_buf.reinterpret_data<int64_t>()[0];

  // Collect all remaining input tensors (excluding callback id) into
  // lightweight payload views.
  std::vector<exla::callback_bridge::Arg> inputs;
  inputs.reserve(args.size() - 1);

  for (size_t i = 1; i < args.size(); ++i) {
    auto maybe_buf_or = args.get<ffi::AnyBuffer>(i);
    if (!maybe_buf_or) {
      return maybe_buf_or.error();
    }

    ffi::AnyBuffer buf = *maybe_buf_or;

    exla::callback_bridge::Arg tensor;
    tensor.dtype = buf.element_type();

    auto dims = buf.dimensions();
    tensor.dims.assign(dims.begin(), dims.end());

    tensor.data = reinterpret_cast<const uint8_t *>(buf.untyped_data());
    tensor.size_bytes = buf.size_bytes();

    inputs.push_back(std::move(tensor));
  }

  // Prepare output buffer descriptors so the callback bridge can write results
  // directly into the final destination buffers.
  std::vector<exla::callback_bridge::OutputBuffer> outputs;
  outputs.reserve(rets.size());

  for (size_t i = 0; i < rets.size(); ++i) {
    auto maybe_ret_or = rets.get<ffi::AnyBuffer>(i);
    if (!maybe_ret_or) {
      return maybe_ret_or.error();
    }

    ffi::Result<ffi::AnyBuffer> ret = *maybe_ret_or;
    ffi::AnyBuffer out = *ret;

    exla::callback_bridge::OutputBuffer buf;
    buf.data = static_cast<uint8_t *>(out.untyped_data());
    buf.size = ffi::ByteWidth(out.element_type()) *
               static_cast<size_t>(out.element_count());

    outputs.push_back(buf);
  }

  // Call back into Elixir through the bridge. On success, the bridge writes
  // results directly into the provided output buffers.
  exla::callback_bridge::Result result =
      exla::callback_bridge::InvokeElixirCallback(callback_id, inputs, outputs);

  if (!result.ok) {
    return ffi::Error(ffi::ErrorCode::kInternal, result.error);
  }

  return ffi::Error::Success();
}

} // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    exla_elixir_callback, exla_elixir_callback_impl,
    ffi::Ffi::Bind()
        .RemainingArgs()
        .RemainingRets());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "exla_elixir_callback", "Host",
                         exla_elixir_callback);


