#include "../elixir_callback_bridge.h"

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
  std::vector<exla::ElixirCallbackArg> inputs;
  inputs.reserve(args.size() - 1);

  for (size_t i = 1; i < args.size(); ++i) {
    auto maybe_buf_or = args.get<ffi::AnyBuffer>(i);
    if (!maybe_buf_or) {
      return maybe_buf_or.error();
    }

    ffi::AnyBuffer buf = *maybe_buf_or;

    exla::ElixirCallbackArg tensor;
    tensor.dtype = buf.element_type();

    auto dims = buf.dimensions();
    tensor.dims.assign(dims.begin(), dims.end());

    tensor.data = reinterpret_cast<const uint8_t *>(buf.untyped_data());
    tensor.size_bytes = buf.size_bytes();

    inputs.push_back(std::move(tensor));
  }

  // Call back into Elixir through the bridge.
  exla::ElixirCallbackResult result =
      exla::CallElixirCallback(callback_id, inputs);

  if (!result.ok) {
    return ffi::Error(ffi::ErrorCode::kInternal, result.error);
  }

  if (result.outputs.size() != rets.size()) {
    return ffi::Error(
        ffi::ErrorCode::kInternal,
        "mismatched number of callback outputs vs custom_call results");
  }

  // Copy returned binaries into the result buffers. We rely on the Elixir side
  // (Nx.elixir_call/3) to have already validated shapes and dtypes.
  for (size_t i = 0; i < rets.size(); ++i) {
    auto maybe_ret_or = rets.get<ffi::AnyBuffer>(i);
    if (!maybe_ret_or) {
      return maybe_ret_or.error();
    }

    ffi::Result<ffi::AnyBuffer> ret = *maybe_ret_or;
    ffi::AnyBuffer out = *ret;

    const auto &payload = result.outputs[i];

    size_t expected =
        ffi::ByteWidth(out.element_type()) * out.element_count();

    if (payload.data.size() != expected) {
      return ffi::Error(
          ffi::ErrorCode::kInternal,
          "callback returned binary of unexpected size for result buffer");
    }

    if (expected > 0) {
      std::memcpy(out.untyped_data(), payload.data.data(), expected);
    }
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


