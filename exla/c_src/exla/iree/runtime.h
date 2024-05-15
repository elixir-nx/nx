#pragma once
#include <iree/hal/api.h>
#include <iree/modules/hal/module.h>
#include <iree/modules/hal/types.h>
#include <iree/runtime/api.h>
#include <iree/vm/api.h>
#include <iree/vm/bytecode/module.h>

#include <memory>

#include "../exla_nif_util.h"

ERL_NIF_TERM run_module(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM setup_runtime(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);

namespace exla {
namespace iree {
namespace runtime {

class IREEInput {
 public:
  void* data;
  size_t size;
  std::vector<iree_hal_dim_t> dims;
  iree_hal_element_type_t type;

  // Default constructor
  IREEInput(void* data, size_t size, std::vector<int64_t> in_dims, iree_hal_element_type_t type) : size(size), type(type) {
    dims.reserve(in_dims.size());

    for (auto dim : in_dims) {
      dims.push_back(static_cast<iree_hal_dim_t>(dim));
    }

    this->data = std::malloc(size);  // Allocate memory
    std::memcpy(this->data, data, size);
  }

  // Destructor
  ~IREEInput() {
    if (data) {
      std::free(data);
      data = nullptr;
    }
  }

  // Disable copy and move semantics for simplicity
  IREEInput(const IREEInput&) = delete;
  IREEInput& operator=(const IREEInput&) = delete;
  IREEInput(IREEInput&&) = delete;
  IREEInput& operator=(IREEInput&&) = delete;

  iree_const_byte_span_t data_byte_span() const {
    return iree_make_const_byte_span(static_cast<uint8_t*>(data), size);
  }
};

}  // namespace runtime
}  // namespace iree
};  // namespace exla