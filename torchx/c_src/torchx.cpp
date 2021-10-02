#include <torch/torch.h>
#include <iostream>

#include "nx_nif_utils.hpp"

std::map<const std::string, const torch::ScalarType> dtypes = {{"byte", torch::kByte}, {"char", torch::kChar}, {"short", torch::kShort}, {"int", torch::kInt}, {"long", torch::kLong}, {"half", torch::kHalf}, {"brain", torch::kBFloat16}, {"float", torch::kFloat}, {"double", torch::kDouble}, {"bool", torch::kBool}};
std::map<const std::string, const int> dtype_sizes = {{"byte", 1}, {"char", 1}, {"short", 2}, {"int", 4}, {"long", 8}, {"half", 2}, {"brain", 2}, {"float", 4}, {"double", 8}};

inline torch::ScalarType string2type(const std::string atom)
{
  return dtypes[atom];
}

inline std::string type2string(const torch::ScalarType type)
{
  for (std::map<const std::string, const torch::ScalarType>::iterator i = dtypes.begin(); i != dtypes.end(); ++i)
  {
    if (i->second == type)
      return i->first;
  }
  return "";
}

#define NIF(NAME) ERL_NIF_TERM NAME(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])

#define SCALAR_PARAM(ARGN, VAR) \
  torch::Scalar VAR;            \
  VAR.~Scalar();                \
  double double_##VAR;          \
  if (enif_get_double(env, argv[ARGN], &double_##VAR) == 0) { \
    long long_##VAR;                                  \
    enif_get_int64(env, argv[ARGN], &long_##VAR);     \
    new (&VAR) torch::Scalar((int64_t)long_##VAR);             \
  } else {                                            \
    new (&VAR) torch::Scalar(double_##VAR);           \
  }

#define SHAPE_PARAM(ARGN, VAR) TUPLE_PARAM(ARGN, std::vector<int64_t>, VAR)

#define TYPE_PARAM(ARGN, VAR)  \
  ATOM_PARAM(ARGN, VAR##_atom) \
  torch::ScalarType VAR = string2type(VAR##_atom)

#define DEVICE_PARAM(ARGN, VAR) TUPLE_PARAM(ARGN, std::vector<int64_t>, VAR)

#define DEVICE(DEV_VEC) torch::device(torch::Device((torch::DeviceType)DEV_VEC[0], (torch::DeviceIndex)DEV_VEC[1]))

#define OPTS(TYPE, DEV_VEC) DEVICE(DEV_VEC).dtype(TYPE)

#define TENSOR_PARAM(ARGN, VAR)                                           \
  torch::Tensor *VAR;                                                     \
  if (!enif_get_resource(env, argv[ARGN], TENSOR_TYPE, (void **)&VAR))  { \
    std::ostringstream msg;                                               \
    msg << "Unable to get " #VAR " tensor param in NIF." << __func__ << "/" << argc;  \
    return nx::nif::error(env, msg.str().c_str());                        \
  }

#define CATCH()                                              \
  catch (c10::Error &error)                                   \
  {                                                          \
    std::ostringstream msg;                                  \
    msg << error.msg() << " in NIF." << __func__ << "/" << argc; \
    return nx::nif::error(env, msg.str().c_str());           \
  }

#define SCALAR(S)                                            \
  try                                                        \
  {                                                          \
    if (c10::isFloatingType(S.type()))                       \
      return nx::nif::ok(env, nx::nif::make(env, S.toDouble())); \
    else                                                     \
      return nx::nif::ok(env, nx::nif::make(env, (long)S.toLong())); \
  }                                                          \
  CATCH()


#define TENSOR(T)                                            \
  try                                                        \
  {                                                          \
    return nx::nif::ok(env, create_tensor_resource(env, T)); \
  }                                                          \
  CATCH()

#define TENSOR_LIST(TL)                                                                        \
  try                                                                                          \
  {                                                                                            \
    std::vector<torch::Tensor> tl = TL;                                                        \
    std::vector<ERL_NIF_TERM> res_list;                                                        \
    for (torch::Tensor t : tl)                                                                 \
      res_list.push_back(create_tensor_resource(env, t));                                      \
    return nx::nif::ok(env, enif_make_list_from_array(env, res_list.data(), res_list.size())); \
  }                                                                                            \
  CATCH()

#define TENSOR_TUPLE(TT)                                                                        \
  try                                                                                           \
  {                                                                                             \
    std::tuple<torch::Tensor, torch::Tensor> tt = TT;                                           \
    std::vector<ERL_NIF_TERM> res_list;                                                         \
    for (torch::Tensor t : {std::get<0>(tt), std::get<1>(tt)})                                  \
      res_list.push_back(create_tensor_resource(env, t));                                       \
    return nx::nif::ok(env, enif_make_tuple_from_array(env, res_list.data(), res_list.size())); \
  }                                                                                             \
  CATCH()

ERL_NIF_TERM
create_tensor_resource(ErlNifEnv *env, torch::Tensor tensor)
{
  ERL_NIF_TERM ret;
  torch::Tensor *tensorPtr;

  tensorPtr = (torch::Tensor *)enif_alloc_resource(TENSOR_TYPE, sizeof(torch::Tensor));
  if (tensorPtr == NULL)
    return enif_make_badarg(env);

  new (tensorPtr) torch::Tensor(tensor.variable_data());

  ret = enif_make_resource(env, tensorPtr);
  enif_release_resource(tensorPtr);

  return ret;
}

NIF(delete_tensor)
{
  TENSOR_PARAM(0, t);

  t->~Tensor();
  enif_release_resource(t);

  return nx::nif::ok(env);
}

unsigned long elem_count(std::vector<int64_t> shape)
{
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>{});
}

NIF(from_blob)
{
  BINARY_PARAM(0, blob);
  SHAPE_PARAM(1, shape);
  TYPE_PARAM(2, type);
  DEVICE_PARAM(3, device);

  if (blob.size / dtype_sizes[type_atom] < elem_count(shape))
    return nx::nif::error(env, "Binary size is too small for the requested shape");

  // Clone here to copy data from blob, which will be GCed.
  TENSOR(torch::clone(torch::from_blob(blob.data, shape, OPTS(type, device))));
}

NIF(to_blob)
{
  ERL_NIF_TERM result;
  TENSOR_PARAM(0, t);
  size_t byte_size = t->nbytes();

  if (argc == 2)
  {
    PARAM(1, int64_t, limit);
    byte_size = limit * t->itemsize();
  }

  torch::optional<torch::Device> device = torch::device_of(*t);

  if(device.has_value() && device.value().type() == torch::kCPU) {
    return nx::nif::ok(env, enif_make_resource_binary(env, t, t->data_ptr(), byte_size));
  } else {
    void *result_data = (void *)enif_make_new_binary(env, byte_size, &result);
    memcpy(result_data, t->data_ptr(), byte_size);
    return nx::nif::ok(env, result);
  }
}

NIF(item)
{
  TENSOR_PARAM(0, t);

  SCALAR(t->item());
}

NIF(scalar_type)
{
  TENSOR_PARAM(0, t);

  std::string type_name = type2string(t->scalar_type());

  if (!type_name.empty())
    return nx::nif::ok(env, enif_make_atom(env, type_name.c_str()));
  else
    return nx::nif::error(env, "Could not determine tensor type.");
}

NIF(shape)
{
  TENSOR_PARAM(0, t);

  std::vector<ERL_NIF_TERM> sizes;
  for (int dim = 0; dim < t->dim(); dim++ )
    sizes.push_back(nx::nif::make(env, ((long)t->size(dim))));

  return nx::nif::ok(env, enif_make_tuple_from_array(env, sizes.data(), sizes.size()));
}

NIF(cuda_is_available)
{
  return nx::nif::make(env, (bool)torch::cuda::is_available());
}

NIF(cuda_device_count)
{
  return nx::nif::make(env, (int)torch::cuda::device_count());
}

NIF(nbytes)
{
  TENSOR_PARAM(0, t);

  return nx::nif::ok(env, enif_make_int64(env, t->nbytes()));
}

NIF(split)
{
  TENSOR_PARAM(0, t);
  PARAM(1, int64_t, batch_size);

  TENSOR_LIST(torch::split(*t, batch_size));
}


NIF(reshape)
{
  TENSOR_PARAM(0, t);
  SHAPE_PARAM(1, shape);

  TENSOR(torch::reshape(*t, shape));
}

NIF(to_type)
{
  TENSOR_PARAM(0, t);
  TYPE_PARAM(1, type);

  TENSOR(t->toType(type));
}

NIF(to_device)
{
  TENSOR_PARAM(0, t);
  DEVICE_PARAM(1, device);

  TENSOR(t->to(DEVICE(device)));
}

NIF(squeeze)
{
  TENSOR_PARAM(0, t);

  if (argc == 2)
  {
    PARAM(1, int64_t, dim);
    TENSOR(torch::squeeze(*t, dim));
  }
  else
    TENSOR(torch::squeeze(*t));
}

NIF(broadcast_to)
{
  TENSOR_PARAM(0, t);
  SHAPE_PARAM(1, shape);

  TENSOR(torch::broadcast_to(*t, shape).clone());
}

NIF(transpose)
{
  TENSOR_PARAM(0, t);
  PARAM(1, int64_t, dim0);
  PARAM(2, int64_t, dim1);

  TENSOR(torch::transpose(*t, dim0, dim1));
}

NIF(narrow)
{
  TENSOR_PARAM(0, t);
  PARAM(1, int64_t, dim);
  PARAM(2, int64_t, start);
  PARAM(3, int64_t, length);

  TENSOR(torch::narrow(*t, dim, start, length).clone());
}

NIF(as_strided)
{
  TENSOR_PARAM(0, t);
  SHAPE_PARAM(1, size);
  LIST_PARAM(2, std::vector<int64_t>, strides);
  PARAM(3, int64_t, offset);

  TENSOR(torch::as_strided(*t, size, strides, offset).clone());
}

NIF(concatenate)
{
  LIST_PARAM(0, std::vector<torch::Tensor>, tensors);

  PARAM(1, int64_t, axis);

  TENSOR(torch::cat(tensors, axis));
}

NIF(gather)
{
  TENSOR_PARAM(0, input);
  TENSOR_PARAM(1, indices);
  TENSOR_PARAM(2, linear_indices_offsets);
  SHAPE_PARAM(3, output_shape);

  auto linear_indices = torch::matmul(*indices, *linear_indices_offsets);

  auto gathered_tensor = torch::gather(input->reshape({input->numel()}), 0, linear_indices.reshape({linear_indices.numel()}));
  TENSOR(gathered_tensor.reshape(output_shape));
}

NIF(take_along_axis)
{
  TENSOR_PARAM(0, input);
  TENSOR_PARAM(1, indices);
  PARAM(2, int64_t, axis);

  TENSOR(torch::gather(*input, axis, *indices));
}

NIF(argsort)
{
  TENSOR_PARAM(0, input);
  PARAM(1, int64_t, axis);
  PARAM(2, bool, is_descending);

  TENSOR(torch::argsort(*input, axis, is_descending));
}

NIF(permute)
{
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int64_t>, dims);

  TENSOR(t->permute(dims).contiguous());
}


/* Creation */

NIF(scalar_tensor)
{
  SCALAR_PARAM(0, scalar);
  TYPE_PARAM(1, type);
  DEVICE_PARAM(2, device);

  TENSOR(torch::scalar_tensor(scalar, OPTS(type, device)));
}

NIF(randint)
{
  PARAM(0, int64_t, min);
  PARAM(1, int64_t, max);
  SHAPE_PARAM(2, shape);
  TYPE_PARAM(3, type);
  DEVICE_PARAM(4, device);

  TENSOR(torch::randint(min, max, shape, OPTS(type, device)));
}

NIF(rand)
{
  PARAM(0, double, min);
  PARAM(1, double, max);
  SHAPE_PARAM(2, shape);
  TYPE_PARAM(3, type);
  DEVICE_PARAM(4, device);

  TENSOR(min + torch::rand(shape, OPTS(type, device)) * (max - min));
}

NIF(normal)
{
  PARAM(0, double, mean);
  PARAM(1, double, std);
  SHAPE_PARAM(2, shape);
  TYPE_PARAM(3, type);
  DEVICE_PARAM(4, device);

  TENSOR(torch::normal(mean, std, shape, c10::nullopt, OPTS(type, device)));
}

NIF(arange)
{
  PARAM(0, int64_t, start);
  PARAM(1, int64_t, end);
  PARAM(2, int64_t, step);
  TYPE_PARAM(3, type);
  DEVICE_PARAM(4, device);

  if (argc == 6)
  {
    SHAPE_PARAM(5, shape);
    TENSOR(torch::reshape(torch::arange((double)start, (double)end, (double)step, OPTS(type, device)), shape));
  }
  else
  {
    TENSOR(torch::arange((double)start, (double)end, (double)step, OPTS(type, device)));
  }
}

NIF(ones)
{
  SHAPE_PARAM(0, shape);
  TYPE_PARAM(1, type);
  DEVICE_PARAM(2, device);

  TENSOR(torch::ones(shape, OPTS(type, device)));
}

NIF(eye)
{
  PARAM(0, int64_t, size);
  TYPE_PARAM(1, type);
  DEVICE_PARAM(2, device);

  TENSOR(torch::eye(size, OPTS(type, device)));
}

NIF(full)
{
  SHAPE_PARAM(0, shape);
  SCALAR_PARAM(1, scalar);
  TYPE_PARAM(2, type);
  DEVICE_PARAM(3, device);

  TENSOR(torch::full(shape, scalar, OPTS(type, device)));
}


/* Binary Ops */

#define BINARY_OP(OP)   BINARY_OP2(OP, OP)

#define BINARY_OP2(OP, NATIVE_OP)   \
  NIF(OP)               \
  {                     \
    TENSOR_PARAM(0, a); \
    TENSOR_PARAM(1, b); \
                        \
    TENSOR(torch::NATIVE_OP(*a, *b)); \
  }

#define BINARY_OPB(OP)   \
  NIF(OP)               \
  {                     \
    TENSOR_PARAM(0, a); \
    TENSOR_PARAM(1, b); \
                        \
    nx::nif::ok(env, nx::nif::make(env, torch::OP(*a, *b))); \
  }


#define UNARY_OP(OP) UNARY_OP2(OP, OP)

#define UNARY_OP2(OP, NATIVE)  \
  NIF(OP)                      \
  {                            \
    TENSOR_PARAM(0, a);        \
    TENSOR(torch::NATIVE(*a)); \
  }

BINARY_OP(bitwise_and)
BINARY_OP(bitwise_or)
BINARY_OP(bitwise_xor)
BINARY_OP2(left_shift, __lshift__)
BINARY_OP2(right_shift, __rshift__)

BINARY_OP2(equal, eq)
BINARY_OP(not_equal)
BINARY_OP(greater)
BINARY_OP(less)
BINARY_OP(greater_equal)
BINARY_OP(less_equal)

BINARY_OP(logical_and)
BINARY_OP(logical_or)
BINARY_OP(logical_xor)

BINARY_OP(add)
BINARY_OP(subtract)
BINARY_OP(divide)
BINARY_OP(remainder)
BINARY_OP(multiply)
BINARY_OP(matmul)
BINARY_OP2(power, pow)
BINARY_OP(atan2)
BINARY_OP(min)
BINARY_OP(max)

BINARY_OP(outer)

NIF(quotient)
{
  TENSOR_PARAM(0, a);
  TENSOR_PARAM(1, b);
  TENSOR(torch::divide(*a, *b, "trunc"));
}

NIF(tensordot)
{
  TENSOR_PARAM(0, t1);
  TENSOR_PARAM(1, t2);
  LIST_PARAM(2, std::vector<int64_t>, axes1);
  LIST_PARAM(3, std::vector<int64_t>, axes2);

  TENSOR(torch::tensordot(*t1, *t2, axes1, axes2));
}


/* Unary Ops */

UNARY_OP(abs)
UNARY_OP(ceil)
UNARY_OP(floor)
UNARY_OP2(negate, negative)
UNARY_OP(round)
UNARY_OP(sign)
UNARY_OP(exp)
UNARY_OP(expm1)
UNARY_OP(sqrt)
UNARY_OP(rsqrt)
UNARY_OP(log)
UNARY_OP(log1p)
UNARY_OP(bitwise_not)
UNARY_OP2(logistic, sigmoid)

UNARY_OP(sin)
UNARY_OP(asin)
UNARY_OP(sinh)
UNARY_OP(asinh)
UNARY_OP(cos)
UNARY_OP(acos)
UNARY_OP(cosh)
UNARY_OP(acosh)
UNARY_OP(tan)
UNARY_OP(atan)
UNARY_OP(tanh)
UNARY_OP(atanh)
UNARY_OP(erf)
UNARY_OP(erfc)
UNARY_OP2(erf_inv, erfinv)


/* Aggregates */

NIF(sum)
{
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int64_t>, dims);
  PARAM(2, bool, keep_dim);

  TENSOR(torch::sum(*t, dims, keep_dim));
}

NIF(argmax)
{
  TENSOR_PARAM(0, t);
  PARAM(1, int64_t, dim);
  PARAM(2, bool, keep_dim);

  if (dim == -1) {
    TENSOR(torch::argmax(*t));
  } else {
    TENSOR(torch::argmax(*t, dim, keep_dim));
  }
}

NIF(argmin)
{
  TENSOR_PARAM(0, t);
  PARAM(1, int64_t, dim);
  PARAM(2, bool, keep_dim);

  if (dim == -1) {
    TENSOR(torch::argmin(*t));
  } else {
    TENSOR(torch::argmin(*t, dim, keep_dim));
  }
}

NIF(all)
{
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int64_t>, axes);
  PARAM(2, bool, keep_dim);

  if (axes.size() == 0) {
    TENSOR(torch::all(*t))
  } else {
    TENSOR(torch::all(*t, axes[0], keep_dim))
  }
}

NIF(cholesky)
{
  TENSOR_PARAM(0, t);
  bool upper = false;

  if (argc == 2)
  {
    GET(1, upper);
  }

  TENSOR(torch::cholesky(*t, upper));
}


/* Transformations */

NIF(qr)
{
  TENSOR_PARAM(0, t);
  bool reduced = true;

  if (argc == 2)
  {
    GET(1, reduced);
  }

  TENSOR_TUPLE(torch::qr(*t, reduced));
}

void free_tensor(ErlNifEnv *env, void *obj)
{
  torch::Tensor* tensor = reinterpret_cast<torch::Tensor*>(obj);
  if (tensor != nullptr) {
    tensor->~Tensor();
    tensor = nullptr;
  }
}

static int
open_resource_type(ErlNifEnv *env)
{
  const char *name = "Tensor";
  ErlNifResourceFlags flags = (ErlNifResourceFlags)(ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER);

  TENSOR_TYPE = enif_open_resource_type(env, NULL, name, free_tensor, flags, NULL);
  if (TENSOR_TYPE == NULL)
    return -1;
  return 0;
}

int upgrade(ErlNifEnv *env, void **priv_data, void **old_priv_data, ERL_NIF_TERM load_info)
{
  // Silence "unused var" warnings.
  (void)(env);
  (void)(priv_data);
  (void)(old_priv_data);
  (void)(load_info);

  return 0;
}

int load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info)
{
  if (open_resource_type(env) == -1)
    return -1;

  // Silence "unused var" warnings.
  (void)(priv_data);
  (void)(load_info);

  return 0;
}

#define F(NAME, ARITY)    \
  {#NAME, ARITY, NAME, 0}

#define DF(NAME, ARITY)                                      \
  {#NAME "_cpu", ARITY, NAME, ERL_NIF_DIRTY_JOB_CPU_BOUND},  \
  {#NAME "_io", ARITY, NAME, ERL_NIF_DIRTY_JOB_IO_BOUND}

static ErlNifFunc nif_functions[] = {
    DF(randint, 5),
    DF(rand, 5),
    DF(normal, 5),
    DF(arange, 5),
    DF(arange, 6),
    DF(scalar_tensor, 3),
    DF(ones, 3),
    DF(eye, 3),
    DF(full, 4),

    DF(item, 1),
    DF(from_blob, 4),
    DF(to_blob, 1),
    DF(to_blob, 2),
    DF(delete_tensor, 1),
    DF(reshape, 2),
    DF(split, 2),
    DF(to_type, 2),
    DF(to_device, 2),
    DF(squeeze, 2),
    DF(squeeze, 1),
    DF(broadcast_to, 2),
    DF(transpose, 3),
    DF(permute, 2),
    DF(narrow, 4),
    DF(as_strided, 4),
    DF(concatenate, 2),
    DF(gather, 4),
    DF(take_along_axis, 3),
    DF(argsort, 3),

    DF(add, 2),
    DF(subtract, 2),
    DF(divide, 2),
    DF(remainder, 2),
    DF(quotient, 2),
    DF(multiply, 2),
    DF(power, 2),
    DF(atan2, 2),
    DF(min, 2),
    DF(max, 2),

    DF(bitwise_and, 2),
    DF(bitwise_or, 2),
    DF(bitwise_xor, 2),
    DF(left_shift, 2),
    DF(right_shift, 2),

    DF(equal, 2),
    DF(not_equal, 2),
    DF(greater, 2),
    DF(less, 2),
    DF(greater_equal, 2),
    DF(less_equal, 2),

    DF(logical_and, 2),
    DF(logical_or, 2),
    DF(logical_xor, 2),

    DF(outer, 2),
    DF(sum, 3),
    DF(argmax, 3),
    DF(argmin, 3),
    DF(all, 3),

    DF(abs, 1),
    DF(ceil, 1),
    DF(floor, 1),
    DF(negate, 1),
    DF(round, 1),
    DF(sign, 1),
    DF(exp, 1),
    DF(expm1, 1),
    DF(sqrt, 1),
    DF(rsqrt, 1),
    DF(log, 1),
    DF(log1p, 1),
    DF(bitwise_not, 1),
    DF(logistic, 1),
    DF(sin, 1),
    DF(asin, 1),
    DF(sinh, 1),
    DF(asinh, 1),
    DF(cos, 1),
    DF(acos, 1),
    DF(cosh, 1),
    DF(acosh, 1),
    DF(tan, 1),
    DF(atan, 1),
    DF(tanh, 1),
    DF(atanh, 1),
    DF(erf, 1),
    DF(erfc, 1),
    DF(erf_inv, 1),

    DF(tensordot, 4),
    DF(matmul, 2),

    DF(cholesky, 1),
    DF(cholesky, 2),
    DF(qr, 1),
    DF(qr, 2),

    F(cuda_is_available, 0),
    F(cuda_device_count, 0),
    F(scalar_type, 1),
    F(shape, 1),
    F(nbytes, 1)};

ERL_NIF_INIT(Elixir.Torchx.NIF, nif_functions, load, NULL, upgrade, NULL)
