#pragma once
#include <iree/hal/api.h>
#include <iree/modules/hal/types.h>
#include <iree/runtime/api.h>

#include <memory>

#include "../exla_nif_util.h"

ERL_NIF_TERM run_module(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM runtime_create_instance(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM runtime_create_session(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);

namespace exla {
namespace iree {
namespace runtime {

template <typename T, void (*ReleaseFunc)(T*)>
struct IREEDeleter {
  void operator()(T* ptr) {
    if (ptr) {
      ReleaseFunc(ptr);  // Call the specific release function
    }
  }
};

using IREEInstanceDeleter = IREEDeleter<iree_runtime_instance_t, iree_runtime_instance_release>;
using IREEDeviceDeleter = IREEDeleter<iree_hal_device_t, iree_hal_device_release>;
using IREESessionDeleter = IREEDeleter<iree_runtime_session_t, iree_runtime_session_release>;

class Instance {
 public:
  // Constructor
  explicit Instance(iree_runtime_instance_t* instance, iree_hal_device_t* device)
      : instance_(instance, IREEInstanceDeleter{}), device_(device, IREEDeviceDeleter{}) {}

  // Default destructor is fine, unique_ptr will handle the resource release
  ~Instance() = default;

  // Copy and move operations are disabled to maintain unique ownership semantics
  Instance(const Instance&) = delete;
  Instance& operator=(const Instance&) = delete;
  Instance(Instance&&) noexcept = default;
  Instance& operator=(Instance&&) noexcept = default;

  iree_runtime_instance_t* get() const {
    return instance_.get();
  }

  iree_runtime_instance_t* operator->() const {
    return instance_.get();
  }

  iree_hal_device_t* device() const {
    return device_.get();
  }

 private:
  std::unique_ptr<iree_runtime_instance_t, IREEInstanceDeleter> instance_;
  std::unique_ptr<iree_hal_device_t, IREEDeviceDeleter> device_;
};

class Session {
 public:
  // Constructor
  explicit Session(Instance* instance) : instance_(instance) {}

  iree_status_t initialize(std::vector<uint8_t> bytecode) {
    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);

    iree_runtime_session_t* session_ptr;

    iree_allocator_t host_allocator = iree_runtime_instance_host_allocator(instance_->get());
    iree_status_t status = iree_runtime_session_create_with_device(
        instance_->get(), &session_options, instance_->device(),
        host_allocator, &session_ptr);

    if (!iree_status_is_ok(status)) {
      return status;
    }

    session_.reset(session_ptr);

    iree_const_byte_span_t span{.data = bytecode.data(), .data_length = bytecode.size()};

    status = iree_runtime_session_append_bytecode_module_from_memory(session_.get(), span, host_allocator);

    if (!iree_status_is_ok(status)) {
      return status;
    }

    return status;
  }

  // Default destructor is fine, unique_ptr will handle the resource release
  ~Session() = default;

  // Copy and move operations are disabled to maintain unique ownership semantics
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  Session(Session&&) noexcept = default;
  Session& operator=(Session&&) noexcept = default;

  // Provide a way to access the underlying pointer like a raw pointer
  iree_runtime_session_t* get() const {
    return session_.get();
  }

  // Overload the arrow operator to enable direct member access to the iree_runtime_session_t
  iree_runtime_session_t* operator->() const {
    return session_.get();
  }

 private:
  Instance* instance_;
  std::unique_ptr<iree_runtime_session_t, IREESessionDeleter> session_;
};

}  // namespace runtime
}  // namespace iree
};  // namespace exla