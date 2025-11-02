#include <fine.hpp>
#include <torch/torch.h>

#if defined(USING_TORCH_V1)
#include <ATen/BatchedTensorImpl.h>
#else
#include <ATen/LegacyBatchedTensorImpl.h>
#endif

#include "torchx_nif_util.h"
#include <iostream>
#include <numeric>

namespace torchx {

// Register TorchTensor as a resource type
FINE_RESOURCE(TorchTensor);

// Macro to register both _cpu and _io variants of a function
// Following EXLA's pattern - create wrapper functions
#define REGISTER_TENSOR_NIF(NAME)                                              \
  auto NAME##_cpu = NAME;                                                      \
  auto NAME##_io = NAME;                                                       \
  FINE_NIF(NAME##_cpu, ERL_NIF_DIRTY_JOB_CPU_BOUND);                           \
  FINE_NIF(NAME##_io, ERL_NIF_DIRTY_JOB_IO_BOUND)

// Macro to register both _cpu and _io variants for a specific arity
// Creates a unified NIF handler that dispatches to the function
// Usage: REGISTER_TENSOR_NIF_ARITY(name, function_symbol)
#define REGISTER_TENSOR_NIF_ARITY(NAME, SYMBOL)                                \
  static ERL_NIF_TERM SYMBOL##_nif(ErlNifEnv *env, int argc,                   \
                                   const ERL_NIF_TERM argv[]) {                \
    return fine::nif(env, argc, argv, SYMBOL);                                 \
  }                                                                            \
  auto __nif_registration_##SYMBOL##_cpu = fine::Registration::register_nif(   \
      {#NAME "_cpu", fine::nif_arity(SYMBOL), SYMBOL##_nif,                    \
       ERL_NIF_DIRTY_JOB_CPU_BOUND});                                          \
  auto __nif_registration_##SYMBOL##_io = fine::Registration::register_nif(    \
      {#NAME "_io", fine::nif_arity(SYMBOL), SYMBOL##_nif,                     \
       ERL_NIF_DIRTY_JOB_IO_BOUND});                                           \
  static_assert(true, "require a semicolon after the macro")

// Helper to get tensor from resource, with proper error checking
torch::Tensor &get_tensor(fine::ResourcePtr<TorchTensor> tensor_res) {
  return tensor_res->tensor();
}

// Helper to create a tensor resource result
fine::Ok<fine::ResourcePtr<TorchTensor>>
tensor_ok(const torch::Tensor &tensor) {
  return fine::Ok(fine::make_resource<TorchTensor>(tensor));
}

// Helper for vector of int64 to IntArrayRef conversion
c10::IntArrayRef vec_to_array_ref(const std::vector<int64_t> &vec) {
  return c10::IntArrayRef(vec);
}

// Helper for device tuple (device_type, device_index) to torch::Device
torch::Device
tuple_to_device(const std::tuple<int64_t, int64_t> &device_tuple) {
  return torch::Device(
      static_cast<torch::DeviceType>(std::get<0>(device_tuple)),
      static_cast<torch::DeviceIndex>(std::get<1>(device_tuple)));
}

// Helper to count elements in a shape
uint64_t elem_count(const std::vector<int64_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<>{});
}

// ============================================================================
// Tensor Management Functions
// ============================================================================

fine::Atom delete_tensor(ErlNifEnv *env,
                         fine::ResourcePtr<TorchTensor> tensor) {
  if (tensor->deallocate()) {
    return fine::Atom("ok");
  } else {
    // Throw exception so backend can catch and return :already_deallocated
    throw std::invalid_argument("Tensor has been deallocated");
  }
}

REGISTER_TENSOR_NIF(delete_tensor);

fine::Ok<fine::ResourcePtr<TorchTensor>>
from_blob(ErlNifEnv *env, ErlNifBinary blob, std::vector<int64_t> shape,
          fine::Atom type_atom, std::tuple<int64_t, int64_t> device_tuple) {

  auto type = string2type(type_atom.to_string());
  auto device = tuple_to_device(device_tuple);

  // Check if binary is large enough
  if (blob.size / dtype_sizes[type_atom.to_string()] < elem_count(shape)) {
    throw std::invalid_argument(
        "Binary size is too small for the requested shape");
  }

  auto tensor = torch::from_blob(blob.data, vec_to_array_ref(shape),
                                 torch::device(torch::kCPU).dtype(type));

  if (device.type() == torch::kCPU) {
    return tensor_ok(tensor.clone());
  } else {
    return tensor_ok(tensor.to(device));
  }
}

REGISTER_TENSOR_NIF(from_blob);

// to_blob - arity 1 and 2 versions
fine::Ok<ErlNifBinary> to_blob_1(ErlNifEnv *env,
                                 fine::ResourcePtr<TorchTensor> tensor_res) {
  auto &t = get_tensor(tensor_res);
  size_t byte_size = t.nbytes();

  torch::optional<torch::Device> device = torch::device_of(t);
  torch::Tensor reshaped = t.flatten();
  void *data_ptr = reshaped.data_ptr();

  ErlNifBinary result;
  enif_alloc_binary(byte_size, &result);

  // Always copy data to avoid use-after-free when tensor is deallocated
  if (device.has_value() && device.value().type() == torch::kCPU) {
    memcpy(result.data, data_ptr, byte_size);
  } else {
    memcpy(result.data, reshaped.to(torch::kCPU).data_ptr(), byte_size);
  }

  return fine::Ok(result);
}

fine::Ok<ErlNifBinary> to_blob_2(ErlNifEnv *env,
                                 fine::ResourcePtr<TorchTensor> tensor_res,
                                 int64_t limit) {
  auto &t = get_tensor(tensor_res);
  size_t byte_size = limit * t.itemsize();

  torch::optional<torch::Device> device = torch::device_of(t);
  torch::Tensor reshaped =
      (byte_size < t.nbytes()) ? t.flatten().slice(0, 0, limit) : t.flatten();
  void *data_ptr = reshaped.data_ptr();

  ErlNifBinary result;
  enif_alloc_binary(byte_size, &result);

  // Always copy data to avoid use-after-free when tensor is deallocated
  if (device.has_value() && device.value().type() == torch::kCPU) {
    memcpy(result.data, data_ptr, byte_size);
  } else {
    memcpy(result.data, reshaped.to(torch::kCPU).data_ptr(), byte_size);
  }

  return fine::Ok(result);
}

REGISTER_TENSOR_NIF_ARITY(to_blob, to_blob_1);
REGISTER_TENSOR_NIF_ARITY(to_blob, to_blob_2);

fine::Ok<torch::Scalar> item(ErlNifEnv *env,
                             fine::ResourcePtr<TorchTensor> tensor) {
  return fine::Ok(get_tensor(tensor).item());
}

REGISTER_TENSOR_NIF(item);

fine::Ok<fine::Atom> scalar_type(ErlNifEnv *env,
                                 fine::ResourcePtr<TorchTensor> tensor) {
  const std::string *type_name = type2string(get_tensor(tensor).scalar_type());
  if (type_name != nullptr) {
    return fine::Ok(fine::Atom(*type_name));
  } else {
    throw std::runtime_error("Could not determine tensor type.");
  }
}

FINE_NIF(scalar_type, 0);

fine::Ok<fine::Term> shape(ErlNifEnv *env,
                           fine::ResourcePtr<TorchTensor> tensor) {
  auto &t = get_tensor(tensor);
  std::vector<ERL_NIF_TERM> sizes;
  for (int64_t dim = 0; dim < t.dim(); dim++) {
    sizes.push_back(fine::encode(env, t.size(dim)));
  }
  // Return as tuple (not list) since Elixir expects {} not []
  return fine::Ok(
      fine::Term(enif_make_tuple_from_array(env, sizes.data(), sizes.size())));
}

FINE_NIF(shape, 0);

bool mps_is_available(ErlNifEnv *env) {
#ifdef MAC_ARM64
  return at::hasMPS();
#else
  return false;
#endif
}

FINE_NIF(mps_is_available, 0);

bool cuda_is_available(ErlNifEnv *env) { return torch::cuda::is_available(); }

FINE_NIF(cuda_is_available, 0);

int64_t cuda_device_count(ErlNifEnv *env) {
  return static_cast<int64_t>(torch::cuda::device_count());
}

FINE_NIF(cuda_device_count, 0);

fine::Ok<int64_t> nbytes(ErlNifEnv *env,
                         fine::ResourcePtr<TorchTensor> tensor) {
  return fine::Ok(static_cast<int64_t>(get_tensor(tensor).nbytes()));
}

FINE_NIF(nbytes, 0);

// ============================================================================
// Tensor Shape Operations
// ============================================================================

fine::Ok<std::vector<fine::ResourcePtr<TorchTensor>>>
split(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> tensor,
      int64_t batch_size) {
  auto tensors = torch::split(get_tensor(tensor), batch_size);
  std::vector<fine::ResourcePtr<TorchTensor>> results;
  for (const auto &t : tensors) {
    results.push_back(fine::make_resource<TorchTensor>(t));
  }
  return fine::Ok(results);
}

REGISTER_TENSOR_NIF(split);

fine::Ok<fine::ResourcePtr<TorchTensor>>
reshape(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> tensor,
        std::vector<int64_t> shape) {
  return tensor_ok(torch::reshape(get_tensor(tensor), vec_to_array_ref(shape)));
}

REGISTER_TENSOR_NIF(reshape);

fine::Ok<fine::ResourcePtr<TorchTensor>>
to_type(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> tensor,
        fine::Atom type_atom) {
  auto type = string2type(type_atom.to_string());
  return tensor_ok(get_tensor(tensor).toType(type));
}

REGISTER_TENSOR_NIF(to_type);

fine::Ok<fine::ResourcePtr<TorchTensor>>
to_device(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> tensor,
          std::tuple<int64_t, int64_t> device_tuple) {
  auto device = tuple_to_device(device_tuple);
  return tensor_ok(get_tensor(tensor).to(device));
}

REGISTER_TENSOR_NIF(to_device);

std::variant<fine::Ok<fine::ResourcePtr<TorchTensor>>>
squeeze(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> tensor,
        std::optional<int64_t> dim) {
  if (dim.has_value()) {
    return tensor_ok(torch::squeeze(get_tensor(tensor), dim.value()));
  } else {
    return tensor_ok(torch::squeeze(get_tensor(tensor)));
  }
}

REGISTER_TENSOR_NIF(squeeze);

fine::Ok<fine::ResourcePtr<TorchTensor>>
broadcast_to(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> tensor,
             std::vector<int64_t> shape) {
  return tensor_ok(
      torch::broadcast_to(get_tensor(tensor), vec_to_array_ref(shape)).clone());
}

REGISTER_TENSOR_NIF(broadcast_to);

fine::Ok<fine::ResourcePtr<TorchTensor>>
transpose(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> tensor, int64_t dim0,
          int64_t dim1) {
  return tensor_ok(torch::transpose(get_tensor(tensor), dim0, dim1));
}

REGISTER_TENSOR_NIF(transpose);

fine::Ok<fine::ResourcePtr<TorchTensor>>
slice(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> input,
      std::vector<int64_t> starts, std::vector<int64_t> lengths,
      std::vector<int64_t> strides) {

  torch::Tensor result = get_tensor(input);
  auto shape = result.sizes();

  for (size_t dim = 0; dim < starts.size(); dim++) {
    int64_t start = starts[dim];
    int64_t stride = strides[dim];
    int64_t length = lengths[dim];
    int64_t end = std::min(start + length, shape[dim]);

    result = result.slice(dim, start, end, stride);
  }

  // Clone the result to ensure memory ownership
  return tensor_ok(result.clone());
}

REGISTER_TENSOR_NIF(slice);

fine::Ok<fine::ResourcePtr<TorchTensor>>
concatenate(ErlNifEnv *env,
            std::vector<fine::ResourcePtr<TorchTensor>> tensor_list,
            int64_t dim) {
  std::vector<torch::Tensor> tensors;
  for (const auto &t : tensor_list) {
    tensors.push_back(get_tensor(t));
  }
  return tensor_ok(torch::cat(tensors, dim));
}

REGISTER_TENSOR_NIF(concatenate);

fine::Ok<fine::ResourcePtr<TorchTensor>>
gather(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> input,
       fine::ResourcePtr<TorchTensor> index, int64_t dim) {
  return tensor_ok(torch::gather(get_tensor(input), dim, get_tensor(index)));
}

REGISTER_TENSOR_NIF(gather);

fine::Ok<fine::ResourcePtr<TorchTensor>>
index_put(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> input,
          std::vector<fine::ResourcePtr<TorchTensor>> indices,
          fine::ResourcePtr<TorchTensor> values, bool accumulate) {

  c10::List<std::optional<at::Tensor>> torch_indices;
  for (const auto &idx : indices) {
    torch_indices.push_back(get_tensor(idx));
  }

  torch::Tensor result = get_tensor(input).clone();
  result.index_put_(torch_indices, get_tensor(values), accumulate);
  return tensor_ok(result);
}

REGISTER_TENSOR_NIF(index_put);

fine::Ok<fine::ResourcePtr<TorchTensor>>
index(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> input,
      std::vector<fine::ResourcePtr<TorchTensor>> indices) {

  c10::List<std::optional<at::Tensor>> torch_indices;
  for (const auto &idx : indices) {
    torch_indices.push_back(get_tensor(idx));
  }

  return tensor_ok(get_tensor(input).index(torch_indices));
}

REGISTER_TENSOR_NIF(index);

fine::Ok<fine::ResourcePtr<TorchTensor>>
argsort(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> input, bool stable,
        int64_t dim, bool descending) {
  return tensor_ok(torch::argsort(get_tensor(input), stable, dim, descending));
}

REGISTER_TENSOR_NIF(argsort);

fine::Ok<
    std::tuple<fine::ResourcePtr<TorchTensor>, fine::ResourcePtr<TorchTensor>>>
top_k(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> input, int64_t k) {
  auto result = torch::topk(get_tensor(input), k);
  return fine::Ok(
      std::make_tuple(fine::make_resource<TorchTensor>(std::get<0>(result)),
                      fine::make_resource<TorchTensor>(std::get<1>(result))));
}

REGISTER_TENSOR_NIF(top_k);

fine::Ok<fine::ResourcePtr<TorchTensor>>
flip(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> input,
     std::vector<int64_t> dims) {
  return tensor_ok(torch::flip(get_tensor(input), vec_to_array_ref(dims)));
}

REGISTER_TENSOR_NIF(flip);

fine::Ok<fine::ResourcePtr<TorchTensor>>
unfold(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> input, int64_t dim,
       int64_t size, int64_t step) {
  return tensor_ok(get_tensor(input).unfold(dim, size, step));
}

REGISTER_TENSOR_NIF(unfold);

fine::Ok<fine::ResourcePtr<TorchTensor>>
put(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> input,
    std::vector<int64_t> indices, fine::ResourcePtr<TorchTensor> source) {

  torch::Tensor output = get_tensor(input).clone();
  torch::Tensor destination = output;

  auto source_shape = get_tensor(source).sizes();

  size_t dim = 0;
  for (dim = 0; dim < indices.size() - 1; dim++) {
    auto start = indices[dim];
    destination = destination.slice(dim, start, start + source_shape[dim]);
  }

  destination.slice(dim, indices[dim], indices[dim] + source_shape[dim])
      .copy_(get_tensor(source));

  return tensor_ok(output);
}

REGISTER_TENSOR_NIF(put);

fine::Ok<fine::ResourcePtr<TorchTensor>>
permute(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> input,
        std::vector<int64_t> permutation) {
  return tensor_ok(get_tensor(input).permute(vec_to_array_ref(permutation)));
}

REGISTER_TENSOR_NIF(permute);

// ============================================================================
// Tensor Creation Functions
// ============================================================================

fine::Ok<fine::ResourcePtr<TorchTensor>>
scalar_tensor(ErlNifEnv *env, torch::Scalar scalar, fine::Atom type_atom,
              std::tuple<int64_t, int64_t> device_tuple) {
  auto type = string2type(type_atom.to_string());
  auto device = tuple_to_device(device_tuple);
  return tensor_ok(
      torch::scalar_tensor(scalar, torch::dtype(type).device(device)));
}

REGISTER_TENSOR_NIF(scalar_tensor);

fine::Ok<fine::ResourcePtr<TorchTensor>>
randint(ErlNifEnv *env, int64_t low, int64_t high, std::vector<int64_t> shape,
        fine::Atom type_atom, std::tuple<int64_t, int64_t> device_tuple) {
  auto type = string2type(type_atom.to_string());
  auto device = tuple_to_device(device_tuple);
  return tensor_ok(torch::randint(low, high, vec_to_array_ref(shape),
                                  torch::dtype(type).device(device)));
}

REGISTER_TENSOR_NIF(randint);

fine::Ok<fine::ResourcePtr<TorchTensor>>
rand(ErlNifEnv *env, double min, double max, std::vector<int64_t> shape,
     fine::Atom type_atom, std::tuple<int64_t, int64_t> device_tuple) {
  auto type = string2type(type_atom.to_string());
  auto device = tuple_to_device(device_tuple);
  auto result =
      torch::rand(vec_to_array_ref(shape), torch::dtype(type).device(device));
  // Scale from [0, 1) to [min, max)
  result = result * (max - min) + min;
  return tensor_ok(result);
}

REGISTER_TENSOR_NIF(rand);

fine::Ok<fine::ResourcePtr<TorchTensor>>
normal(ErlNifEnv *env, double mean, double std, std::vector<int64_t> shape,
       fine::Atom type_atom, std::tuple<int64_t, int64_t> device_tuple) {
  auto type = string2type(type_atom.to_string());
  auto device = tuple_to_device(device_tuple);
  return tensor_ok(torch::normal(mean, std, vec_to_array_ref(shape),
                                 c10::nullopt,
                                 torch::dtype(type).device(device)));
}

REGISTER_TENSOR_NIF(normal);

// arange - arity 5 and 6 versions
fine::Ok<fine::ResourcePtr<TorchTensor>>
arange_5(ErlNifEnv *env, int64_t start, int64_t end, int64_t step,
         fine::Atom type_atom, std::tuple<int64_t, int64_t> device_tuple) {
  auto type = string2type(type_atom.to_string());
  auto device = tuple_to_device(device_tuple);
  return tensor_ok(torch::arange(
      static_cast<double>(start), static_cast<double>(end),
      static_cast<double>(step), torch::dtype(type).device(device)));
}

fine::Ok<fine::ResourcePtr<TorchTensor>>
arange_6(ErlNifEnv *env, int64_t start, int64_t end, int64_t step,
         fine::Atom type_atom, std::tuple<int64_t, int64_t> device_tuple,
         std::vector<int64_t> shape) {
  auto type = string2type(type_atom.to_string());
  auto device = tuple_to_device(device_tuple);
  auto result = torch::arange(
      static_cast<double>(start), static_cast<double>(end),
      static_cast<double>(step), torch::dtype(type).device(device));
  return tensor_ok(torch::reshape(result, vec_to_array_ref(shape)));
}

REGISTER_TENSOR_NIF_ARITY(arange, arange_5);
REGISTER_TENSOR_NIF_ARITY(arange, arange_6);

fine::Ok<fine::ResourcePtr<TorchTensor>>
ones(ErlNifEnv *env, std::vector<int64_t> shape, fine::Atom type_atom,
     std::tuple<int64_t, int64_t> device_tuple) {
  auto type = string2type(type_atom.to_string());
  auto device = tuple_to_device(device_tuple);
  return tensor_ok(
      torch::ones(vec_to_array_ref(shape), torch::dtype(type).device(device)));
}

REGISTER_TENSOR_NIF(ones);

fine::Ok<fine::ResourcePtr<TorchTensor>>
eye(ErlNifEnv *env, int64_t m, int64_t n, fine::Atom type_atom,
    std::tuple<int64_t, int64_t> device_tuple) {
  auto type = string2type(type_atom.to_string());
  auto device = tuple_to_device(device_tuple);
  return tensor_ok(torch::eye(m, n, torch::dtype(type).device(device)));
}

REGISTER_TENSOR_NIF(eye);

fine::Ok<fine::ResourcePtr<TorchTensor>>
full(ErlNifEnv *env, std::vector<int64_t> shape, torch::Scalar scalar,
     fine::Atom type_atom, std::tuple<int64_t, int64_t> device_tuple) {
  auto type = string2type(type_atom.to_string());
  auto device = tuple_to_device(device_tuple);
  return tensor_ok(torch::full(vec_to_array_ref(shape), scalar,
                               torch::dtype(type).device(device)));
}

REGISTER_TENSOR_NIF(full);

// ============================================================================
// Binary Operations
// ============================================================================

#define BINARY_OP(NAME, TORCH_OP)                                              \
  fine::Ok<fine::ResourcePtr<TorchTensor>> NAME(                               \
      ErlNifEnv *env, fine::ResourcePtr<TorchTensor> a,                        \
      fine::ResourcePtr<TorchTensor> b) {                                      \
    return tensor_ok(torch::TORCH_OP(get_tensor(a), get_tensor(b)));           \
  }                                                                            \
  REGISTER_TENSOR_NIF(NAME)

BINARY_OP(bitwise_and, bitwise_and);
BINARY_OP(bitwise_or, bitwise_or);
BINARY_OP(bitwise_xor, bitwise_xor);
BINARY_OP(left_shift, __lshift__);
BINARY_OP(right_shift, __rshift__);
BINARY_OP(equal, eq);
BINARY_OP(not_equal, not_equal);
BINARY_OP(greater, greater);
BINARY_OP(less, less);
BINARY_OP(greater_equal, greater_equal);
BINARY_OP(less_equal, less_equal);
BINARY_OP(logical_and, logical_and);
BINARY_OP(logical_or, logical_or);
BINARY_OP(logical_xor, logical_xor);
BINARY_OP(add, add);
BINARY_OP(subtract, subtract);
BINARY_OP(divide, divide);
BINARY_OP(remainder, remainder);
BINARY_OP(quotient, floor_divide);
BINARY_OP(multiply, multiply);
BINARY_OP(pow, pow);
BINARY_OP(atan2, atan2);
BINARY_OP(min, min);
BINARY_OP(max, max);
BINARY_OP(fmod, fmod);

#undef BINARY_OP

// ============================================================================
// Unary Operations
// ============================================================================

#define UNARY_OP(NAME, TORCH_OP)                                               \
  fine::Ok<fine::ResourcePtr<TorchTensor>> NAME(                               \
      ErlNifEnv *env, fine::ResourcePtr<TorchTensor> a) {                      \
    return tensor_ok(torch::TORCH_OP(get_tensor(a)));                          \
  }                                                                            \
  REGISTER_TENSOR_NIF(NAME)

UNARY_OP(abs, abs);
UNARY_OP(ceil, ceil);
UNARY_OP(floor, floor);
UNARY_OP(negate, neg);
UNARY_OP(round, round);
UNARY_OP(sign, sign);
UNARY_OP(exp, exp);
UNARY_OP(expm1, expm1);
UNARY_OP(sqrt, sqrt);
UNARY_OP(rsqrt, rsqrt);
UNARY_OP(log, log);
UNARY_OP(log1p, log1p);
UNARY_OP(bitwise_not, bitwise_not);
UNARY_OP(logical_not, logical_not);
UNARY_OP(sigmoid, sigmoid);
UNARY_OP(sin, sin);
UNARY_OP(asin, asin);
UNARY_OP(sinh, sinh);
UNARY_OP(asinh, asinh);
UNARY_OP(cos, cos);
UNARY_OP(acos, acos);
UNARY_OP(cosh, cosh);
UNARY_OP(acosh, acosh);
UNARY_OP(tan, tan);
UNARY_OP(atan, atan);
UNARY_OP(tanh, tanh);
UNARY_OP(atanh, atanh);
UNARY_OP(erf, erf);
UNARY_OP(erfc, erfc);
UNARY_OP(erf_inv, erfinv);
// cbrt is not in torch namespace, needs custom implementation
fine::Ok<fine::ResourcePtr<TorchTensor>>
cbrt(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> tensor) {
  auto &t = get_tensor(tensor);
  if (t.scalar_type() == torch::kDouble) {
    return tensor_ok(torch::pow(t, 1.0 / 3));
  } else {
    return tensor_ok(torch::pow(t, 1.0f / 3));
  }
}

REGISTER_TENSOR_NIF(cbrt);
UNARY_OP(is_nan, isnan);
UNARY_OP(is_infinity, isinf);
UNARY_OP(view_as_real, view_as_real);
// conjugate needs special handling - conj() returns a view, must clone
fine::Ok<fine::ResourcePtr<TorchTensor>>
conjugate(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> a) {
  at::Tensor conjugated = get_tensor(a).conj();
  return tensor_ok(conjugated.clone(conjugated.suggest_memory_format()));
}

REGISTER_TENSOR_NIF(conjugate);

#undef UNARY_OP

// ============================================================================
// Reduction Operations
// ============================================================================

fine::Ok<fine::ResourcePtr<TorchTensor>>
tensordot(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> a,
          fine::ResourcePtr<TorchTensor> b, std::vector<int64_t> axes_a,
          std::vector<int64_t> batch_a, std::vector<int64_t> axes_b,
          std::vector<int64_t> batch_b) {

  bool is_batched = batch_a.size() > 0 || batch_b.size() > 0;

  torch::Tensor result;

  if (is_batched) {
    // Handle batched tensordot using vmap/BatchDim
    std::vector<at::BatchDim> batch_dims_a, batch_dims_b;
    int64_t vmap_level = 0;

    for (auto dim : batch_a) {
      batch_dims_a.push_back(at::BatchDim(vmap_level++, dim));
    }
    torch::Tensor batched_a = at::makeBatched(
        get_tensor(a), at::BatchDims(batch_dims_a.begin(), batch_dims_a.end()));

    vmap_level = 0;
    for (auto dim : batch_b) {
      batch_dims_b.push_back(at::BatchDim(vmap_level++, dim));
    }
    torch::Tensor batched_b = at::makeBatched(
        get_tensor(b), at::BatchDims(batch_dims_b.begin(), batch_dims_b.end()));

    torch::Tensor batched_result =
        torch::tensordot(batched_a, batched_b, vec_to_array_ref(axes_a),
                         vec_to_array_ref(axes_b));

    auto impl = at::maybeGetBatchedImpl(batched_result);
    if (!impl) {
      throw std::runtime_error("unable to get tensordot result");
    }
    result = torch::clone(impl->value());
  } else {
    result =
        torch::tensordot(get_tensor(a), get_tensor(b), vec_to_array_ref(axes_a),
                         vec_to_array_ref(axes_b));
  }

  return tensor_ok(result);
}

REGISTER_TENSOR_NIF(tensordot);

fine::Ok<fine::ResourcePtr<TorchTensor>>
matmul(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> a,
       fine::ResourcePtr<TorchTensor> b) {
  return tensor_ok(torch::matmul(get_tensor(a), get_tensor(b)));
}

REGISTER_TENSOR_NIF(matmul);

fine::Ok<fine::ResourcePtr<TorchTensor>>
pad(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> tensor,
    fine::ResourcePtr<TorchTensor> constant, std::vector<int64_t> config) {
  return tensor_ok(torch::constant_pad_nd(get_tensor(tensor),
                                          vec_to_array_ref(config),
                                          get_tensor(constant).item()));
}

REGISTER_TENSOR_NIF(pad);

fine::Ok<fine::ResourcePtr<TorchTensor>>
triangular_solve(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> a,
                 fine::ResourcePtr<TorchTensor> b, bool transpose, bool upper) {
  auto ts_a = get_tensor(a);
  if (transpose) {
    auto num_dims = ts_a.dim();
    ts_a = torch::transpose(ts_a, num_dims - 2, num_dims - 1);
    upper = !upper;
  }

  torch::Tensor result =
      torch::linalg_solve_triangular(ts_a, get_tensor(b), upper, true, false);
  return tensor_ok(result);
}

REGISTER_TENSOR_NIF(triangular_solve);

fine::Ok<fine::ResourcePtr<TorchTensor>>
determinant(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> t) {
  return tensor_ok(get_tensor(t).det());
}

REGISTER_TENSOR_NIF(determinant);

fine::Ok<fine::ResourcePtr<TorchTensor>> sort(ErlNifEnv *env,
                                              fine::ResourcePtr<TorchTensor> t,
                                              bool stable, int64_t axis,
                                              bool descending) {
  std::tuple<torch::Tensor, torch::Tensor> result =
      get_tensor(t).sort(stable, axis, descending);
  return tensor_ok(std::get<0>(result));
}

REGISTER_TENSOR_NIF(sort);

fine::Ok<fine::ResourcePtr<TorchTensor>>
clip(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> t,
     fine::ResourcePtr<TorchTensor> min, fine::ResourcePtr<TorchTensor> max) {
  return tensor_ok(
      torch::clip(get_tensor(t), get_tensor(min), get_tensor(max)));
}

REGISTER_TENSOR_NIF(clip);

fine::Ok<fine::ResourcePtr<TorchTensor>>
where(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> pred,
      fine::ResourcePtr<TorchTensor> on_true,
      fine::ResourcePtr<TorchTensor> on_false) {
  return tensor_ok(torch::where(get_tensor(pred), get_tensor(on_true),
                                get_tensor(on_false)));
}

REGISTER_TENSOR_NIF(where);

fine::Ok<fine::ResourcePtr<TorchTensor>> sum(ErlNifEnv *env,
                                             fine::ResourcePtr<TorchTensor> t,
                                             std::vector<int64_t> dims,
                                             bool keep_dim) {
  return tensor_ok(torch::sum(get_tensor(t), vec_to_array_ref(dims), keep_dim));
}

REGISTER_TENSOR_NIF(sum);

// product - arity 1 and 3 versions
fine::Ok<fine::ResourcePtr<TorchTensor>>
product_1(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> t) {
  return tensor_ok(torch::prod(get_tensor(t)));
}

fine::Ok<fine::ResourcePtr<TorchTensor>>
product_3(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> t, int64_t dim,
          bool keep_dim) {
  return tensor_ok(torch::prod(get_tensor(t), dim, keep_dim));
}

REGISTER_TENSOR_NIF_ARITY(product, product_1);
REGISTER_TENSOR_NIF_ARITY(product, product_3);

fine::Ok<fine::ResourcePtr<TorchTensor>>
argmax(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> t, int64_t dim,
       bool keep_dim) {
  if (dim == -1) {
    return tensor_ok(torch::argmax(get_tensor(t)));
  } else {
    return tensor_ok(torch::argmax(get_tensor(t), dim, keep_dim));
  }
}

REGISTER_TENSOR_NIF(argmax);

fine::Ok<fine::ResourcePtr<TorchTensor>>
argmin(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> t, int64_t dim,
       bool keep_dim) {
  if (dim == -1) {
    return tensor_ok(torch::argmin(get_tensor(t)));
  } else {
    return tensor_ok(torch::argmin(get_tensor(t), dim, keep_dim));
  }
}

REGISTER_TENSOR_NIF(argmin);

fine::Ok<fine::ResourcePtr<TorchTensor>>
fft(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> tensor, int64_t length,
    int64_t axis) {
  return tensor_ok(torch::fft::fft(get_tensor(tensor), length, axis));
}

REGISTER_TENSOR_NIF(fft);

fine::Ok<fine::ResourcePtr<TorchTensor>>
ifft(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> tensor, int64_t length,
     int64_t axis) {
  return tensor_ok(torch::fft::ifft(get_tensor(tensor), length, axis));
}

REGISTER_TENSOR_NIF(ifft);

fine::Ok<fine::ResourcePtr<TorchTensor>>
fft2(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> tensor,
     std::vector<int64_t> lengths, std::vector<int64_t> axes) {
  return tensor_ok(torch::fft::fft2(
      get_tensor(tensor), vec_to_array_ref(lengths), vec_to_array_ref(axes)));
}

REGISTER_TENSOR_NIF(fft2);

fine::Ok<fine::ResourcePtr<TorchTensor>>
ifft2(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> tensor,
      std::vector<int64_t> lengths, std::vector<int64_t> axes) {
  return tensor_ok(torch::fft::ifft2(
      get_tensor(tensor), vec_to_array_ref(lengths), vec_to_array_ref(axes)));
}

REGISTER_TENSOR_NIF(ifft2);

// all - arity 1 and 3 versions
fine::Ok<fine::ResourcePtr<TorchTensor>>
all_1(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> t) {
  return tensor_ok(torch::all(get_tensor(t)));
}

fine::Ok<fine::ResourcePtr<TorchTensor>> all_3(ErlNifEnv *env,
                                               fine::ResourcePtr<TorchTensor> t,
                                               int64_t axis, bool keep_dim) {
  return tensor_ok(torch::all(get_tensor(t), axis, keep_dim));
}

REGISTER_TENSOR_NIF_ARITY(all, all_1);
REGISTER_TENSOR_NIF_ARITY(all, all_3);

// any - arity 1 and 3 versions
fine::Ok<fine::ResourcePtr<TorchTensor>>
any_1(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> t) {
  return tensor_ok(torch::any(get_tensor(t)));
}

fine::Ok<fine::ResourcePtr<TorchTensor>> any_3(ErlNifEnv *env,
                                               fine::ResourcePtr<TorchTensor> t,
                                               int64_t axis, bool keep_dim) {
  return tensor_ok(torch::any(get_tensor(t), axis, keep_dim));
}

REGISTER_TENSOR_NIF_ARITY(any, any_1);
REGISTER_TENSOR_NIF_ARITY(any, any_3);

fine::Ok<fine::ResourcePtr<TorchTensor>>
all_close(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> a,
          fine::ResourcePtr<TorchTensor> b, double rtol, double atol,
          bool equal_nan) {
  bool result =
      torch::allclose(get_tensor(a), get_tensor(b), rtol, atol, equal_nan);
  auto init_opts = torch::device(torch::kCPU).dtype(torch::kBool);
  return tensor_ok(torch::scalar_tensor(result, init_opts));
}

REGISTER_TENSOR_NIF(all_close);

fine::Ok<fine::ResourcePtr<TorchTensor>>
cumulative_sum(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> t, int64_t axis) {
  return tensor_ok(torch::cumsum(get_tensor(t), axis));
}

REGISTER_TENSOR_NIF(cumulative_sum);

fine::Ok<fine::ResourcePtr<TorchTensor>>
cumulative_product(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> t,
                   int64_t axis) {
  return tensor_ok(torch::cumprod(get_tensor(t), axis));
}

REGISTER_TENSOR_NIF(cumulative_product);

fine::Ok<fine::ResourcePtr<TorchTensor>>
cumulative_min(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> t, int64_t axis) {
  const std::tuple<torch::Tensor, torch::Tensor> &tt =
      torch::cummin(get_tensor(t), axis);
  return tensor_ok(std::get<0>(tt));
}

REGISTER_TENSOR_NIF(cumulative_min);

fine::Ok<fine::ResourcePtr<TorchTensor>>
cumulative_max(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> t, int64_t axis) {
  const std::tuple<torch::Tensor, torch::Tensor> &tt =
      torch::cummax(get_tensor(t), axis);
  return tensor_ok(std::get<0>(tt));
}

REGISTER_TENSOR_NIF(cumulative_max);

// cholesky - arity 1 and 2 versions
fine::Ok<fine::ResourcePtr<TorchTensor>>
cholesky_1(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> t) {
  return tensor_ok(torch::cholesky(get_tensor(t)));
}

fine::Ok<fine::ResourcePtr<TorchTensor>>
cholesky_2(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> t, bool upper) {
  if (upper) {
    return tensor_ok(torch::cholesky(get_tensor(t)).mH());
  }
  return tensor_ok(torch::cholesky(get_tensor(t)));
}

REGISTER_TENSOR_NIF_ARITY(cholesky, cholesky_1);
REGISTER_TENSOR_NIF_ARITY(cholesky, cholesky_2);

// qr - arity 1 and 2 versions
fine::Ok<
    std::tuple<fine::ResourcePtr<TorchTensor>, fine::ResourcePtr<TorchTensor>>>
qr_1(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> t) {
  auto result = torch::linalg_qr(get_tensor(t), "reduced");
  return fine::Ok(
      std::make_tuple(fine::make_resource<TorchTensor>(std::get<0>(result)),
                      fine::make_resource<TorchTensor>(std::get<1>(result))));
}

fine::Ok<
    std::tuple<fine::ResourcePtr<TorchTensor>, fine::ResourcePtr<TorchTensor>>>
qr_2(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> t, bool reduced) {
  auto result =
      torch::linalg_qr(get_tensor(t), reduced ? "reduced" : "complete");
  return fine::Ok(
      std::make_tuple(fine::make_resource<TorchTensor>(std::get<0>(result)),
                      fine::make_resource<TorchTensor>(std::get<1>(result))));
}

REGISTER_TENSOR_NIF_ARITY(qr, qr_1);
REGISTER_TENSOR_NIF_ARITY(qr, qr_2);

// svd - arity 1 and 2 versions
fine::Ok<
    std::tuple<fine::ResourcePtr<TorchTensor>, fine::ResourcePtr<TorchTensor>,
               fine::ResourcePtr<TorchTensor>>>
svd_1(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> t) {
  auto result = torch::linalg_svd(get_tensor(t), true);
  return fine::Ok(
      std::make_tuple(fine::make_resource<TorchTensor>(std::get<0>(result)),
                      fine::make_resource<TorchTensor>(std::get<1>(result)),
                      fine::make_resource<TorchTensor>(std::get<2>(result))));
}

fine::Ok<
    std::tuple<fine::ResourcePtr<TorchTensor>, fine::ResourcePtr<TorchTensor>,
               fine::ResourcePtr<TorchTensor>>>
svd_2(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> t, bool full_matrices) {
  auto result = torch::linalg_svd(get_tensor(t), full_matrices);
  return fine::Ok(
      std::make_tuple(fine::make_resource<TorchTensor>(std::get<0>(result)),
                      fine::make_resource<TorchTensor>(std::get<1>(result)),
                      fine::make_resource<TorchTensor>(std::get<2>(result))));
}

REGISTER_TENSOR_NIF_ARITY(svd, svd_1);
REGISTER_TENSOR_NIF_ARITY(svd, svd_2);

fine::Ok<
    std::tuple<fine::ResourcePtr<TorchTensor>, fine::ResourcePtr<TorchTensor>,
               fine::ResourcePtr<TorchTensor>>>
lu(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> t) {
  std::tuple<torch::Tensor, torch::Tensor> lu_result =
      torch::linalg_lu_factor(get_tensor(t));
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> plu =
      torch::lu_unpack(std::get<0>(lu_result), std::get<1>(lu_result));

  return fine::Ok(
      std::make_tuple(fine::make_resource<TorchTensor>(std::get<0>(plu)),
                      fine::make_resource<TorchTensor>(std::get<1>(plu)),
                      fine::make_resource<TorchTensor>(std::get<2>(plu))));
}

REGISTER_TENSOR_NIF(lu);

fine::Ok<fine::ResourcePtr<TorchTensor>>
amax(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> tensor,
     std::vector<int64_t> axes, bool keep_axes) {
  return tensor_ok(
      at::amax(get_tensor(tensor), vec_to_array_ref(axes), keep_axes));
}

REGISTER_TENSOR_NIF(amax);

fine::Ok<fine::ResourcePtr<TorchTensor>>
amin(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> tensor,
     std::vector<int64_t> axes, bool keep_axes) {
  return tensor_ok(
      at::amin(get_tensor(tensor), vec_to_array_ref(axes), keep_axes));
}

REGISTER_TENSOR_NIF(amin);

fine::Ok<
    std::tuple<fine::ResourcePtr<TorchTensor>, fine::ResourcePtr<TorchTensor>>>
eigh(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> tensor) {
  auto result = torch::linalg_eigh(get_tensor(tensor));
  return fine::Ok(
      std::make_tuple(fine::make_resource<TorchTensor>(std::get<0>(result)),
                      fine::make_resource<TorchTensor>(std::get<1>(result))));
}

REGISTER_TENSOR_NIF(eigh);

fine::Ok<fine::ResourcePtr<TorchTensor>>
solve(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> tensorA,
      fine::ResourcePtr<TorchTensor> tensorB) {
  return tensor_ok(
      torch::linalg_solve(get_tensor(tensorA), get_tensor(tensorB)));
}

REGISTER_TENSOR_NIF(solve);

fine::Ok<fine::ResourcePtr<TorchTensor>>
conv(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> tensor,
     fine::ResourcePtr<TorchTensor> kernel, std::vector<int64_t> stride,
     std::vector<int64_t> padding, std::vector<int64_t> dilation,
     bool transposed, int64_t groups) {
  c10::optional<at::Tensor> bias_tensor;
  std::vector<int64_t> output_padding;
  output_padding.push_back(0);

  return tensor_ok(at::convolution(get_tensor(tensor), get_tensor(kernel),
                                   bias_tensor, vec_to_array_ref(stride),
                                   vec_to_array_ref(padding),
                                   vec_to_array_ref(dilation), transposed,
                                   vec_to_array_ref(output_padding), groups));
}

REGISTER_TENSOR_NIF(conv);

fine::Ok<fine::ResourcePtr<TorchTensor>>
max_pool_3d(ErlNifEnv *env, fine::ResourcePtr<TorchTensor> tensor,
            std::vector<int64_t> kernel_size, std::vector<int64_t> strides,
            std::vector<int64_t> padding, std::vector<int64_t> dilation) {
  return tensor_ok(
      at::max_pool3d(get_tensor(tensor), vec_to_array_ref(kernel_size),
                     vec_to_array_ref(strides), vec_to_array_ref(padding),
                     vec_to_array_ref(dilation)));
}

REGISTER_TENSOR_NIF(max_pool_3d);

} // namespace torchx

// Initialize the NIF module
FINE_INIT("Elixir.Torchx.NIF");
