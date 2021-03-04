#include <torch/torch.h>
#include <iostream>

#include "nx_nif_utils.hpp"

ErlNifResourceType *TENSOR_TYPE;

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

#define SHAPE_PARAM(ARGN, VAR) TUPLE_PARAM(ARGN, std::vector<int64_t>, VAR)

#define TYPE_PARAM(ARGN, VAR)  \
  ATOM_PARAM(ARGN, VAR##_atom) \
  torch::ScalarType VAR = string2type(VAR##_atom)

#define TENSOR_PARAM(ARGN, VAR)                                        \
  torch::Tensor *VAR;                                                     \
  if (!enif_get_resource(env, argv[ARGN], TENSOR_TYPE, (void **)&VAR)) \
    return nx::nif::error(env, "Unable to get " #VAR " tensor param.");

#define TENSOR(T)                                            \
  try                                                        \
  {                                                          \
    return nx::nif::ok(env, create_tensor_resource(env, T)); \
  }                                                          \
  catch (c10::Error error)                                   \
  {                                                          \
    return nx::nif::error(env, error.msg().c_str());         \
  }


#define TENSOR_LIST(TL)                                                                        \
  try                                                                                          \
  {                                                                                            \
    std::vector<torch::Tensor> tl = TL;                                                           \
    std::vector<ERL_NIF_TERM> res_list;                                                        \
    for (torch::Tensor t : tl)                                                                    \
      res_list.push_back(create_tensor_resource(env, t));                                      \
    return nx::nif::ok(env, enif_make_list_from_array(env, res_list.data(), res_list.size())); \
  }                                                                                            \
  catch (c10::Error error)                                                                     \
  {                                                                                            \
    return nx::nif::error(env, error.msg().c_str());                                           \
  }

#define TENSOR_TUPLE(TT)                                                                        \
  try                                                                                           \
  {                                                                                             \
    std::tuple<torch::Tensor, torch::Tensor> tt = TT;                                                 \
    std::vector<ERL_NIF_TERM> res_list;                                                         \
    for (torch::Tensor t : {std::get<0>(tt), std::get<1>(tt)})                                     \
      res_list.push_back(create_tensor_resource(env, t));                                       \
    return nx::nif::ok(env, enif_make_tuple_from_array(env, res_list.data(), res_list.size())); \
  }                                                                                             \
  catch (c10::Error error)                                                                      \
  {                                                                                             \
    return nx::nif::error(env, error.msg().c_str());                                            \
  }

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

  delete t;
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

  if (blob.size / dtype_sizes[type_atom] < elem_count(shape))
    return nx::nif::error(env, "Binary size is too small for the requested shape");

  // Clone here to copy data from blob, which will be GCed.
  TENSOR(torch::clone(torch::from_blob(blob.data, shape, type)));
}

NIF(type)
{
  TENSOR_PARAM(0, t);

  std::string type_name = type2string(t->scalar_type());

  if (!type_name.empty())
    return nx::nif::ok(env, enif_make_atom(env, type_name.c_str()));
  else
    return nx::nif::error(env, "Could not determine tensor type.");
}

NIF(device)
{
  TENSOR_PARAM(0, t);

  torch::optional<torch::Device> device = torch::device_of(*t);

  if (device.has_value())
    return nx::nif::ok(env, nx::nif::make(env, device.value().str().c_str()));
  else
    return nx::nif::error(env, "Could not determine tensor device.");
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

  void *result_data = (void *)enif_make_new_binary(env, byte_size, &result);
  memcpy(result_data, t->data_ptr(), byte_size);

  return result;
}

NIF(scalar_tensor)
{
  PARAM(0, double, scalar);
  TYPE_PARAM(1, type);

  TENSOR(torch::scalar_tensor(scalar, type));
}

NIF(randint)
{
  PARAM(0, int64_t, min);
  PARAM(1, int64_t, max);
  SHAPE_PARAM(2, shape);
  TYPE_PARAM(3, type);

  TENSOR(torch::randint(min, max, shape, type));
}

NIF(rand)
{
  PARAM(0, double, min);
  PARAM(1, double, max);
  SHAPE_PARAM(2, shape);
  TYPE_PARAM(3, type);

  TENSOR(min + torch::rand(shape, type) * (max - min));
}

NIF(normal)
{
  PARAM(0, double, mean);
  PARAM(1, double, std);
  SHAPE_PARAM(2, shape);

  TENSOR(torch::normal(mean, std, shape));
}

NIF(arange)
{
  PARAM(0, int64_t, start);
  PARAM(1, int64_t, end);
  PARAM(2, int64_t, step);
  TYPE_PARAM(3, type);

  if (argc == 5)
  {
    SHAPE_PARAM(4, shape);
    TENSOR(torch::reshape(torch::arange((double)start, (double)end, (double)step, type), shape));
  }
  else
  {
    TENSOR(torch::arange((double)start, (double)end, (double)step, type));
  }
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

NIF(permute)
{
  TENSOR_PARAM(0, t);
  LIST_PARAM(1, std::vector<int64_t>, dims);

  TENSOR(t->permute(dims).clone());
}

NIF(ones)
{
  SHAPE_PARAM(0, shape);
  TYPE_PARAM(1, type);

  TENSOR(torch::ones(shape, type));
}

NIF(eye)
{
  PARAM(0, int64_t, size);
  TYPE_PARAM(1, type);

  TENSOR(torch::eye(size, type));
}

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

#define UNARY_OP2(OP, NATIVE) \
  NIF(OP)               \
  {                     \
    TENSOR_PARAM(0, a); \
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

UNARY_OP(abs)
UNARY_OP(ceil)
UNARY_OP(floor)
UNARY_OP2(negate, negative)
UNARY_OP(round)
UNARY_OP(sign)
UNARY_OP(bitwise_not)


NIF(dot)
{
  TENSOR_PARAM(0, a);
  TENSOR_PARAM(1, b);

  TENSOR(torch::matmul(*a, *b));
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
  // std::cout << "Deleting: " << obj << std::endl;
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
  {                       \
#NAME, ARITY, NAME, 0 \
  }

#define DF(NAME, ARITY)                                  \
  {                                                      \
      #NAME, ARITY, NAME, ERL_NIF_DIRTY_JOB_CPU_BOUND},  \
  {                                                      \
#NAME "_io", ARITY, NAME, ERL_NIF_DIRTY_JOB_IO_BOUND \
  }

static ErlNifFunc nif_functions[] = {
    DF(randint, 4),
    DF(rand, 4),
    DF(normal, 3),
    DF(arange, 4),
    DF(arange, 5),
    DF(from_blob, 3),
    DF(to_blob, 1),
    DF(to_blob, 2),
    DF(scalar_tensor, 2),
    DF(delete_tensor, 1),
    DF(ones, 1),
    DF(eye, 2),
    DF(reshape, 2),
    DF(split, 2),
    DF(to_type, 2),
    DF(squeeze, 2),
    DF(squeeze, 1),
    DF(broadcast_to, 2),
    DF(transpose, 3),
    DF(permute, 2),

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

    DF(abs, 1),
    DF(ceil, 1),
    DF(floor, 1),
    DF(negate, 1),
    DF(round, 1),
    DF(sign, 1),
    DF(bitwise_not, 1),

    DF(dot, 2),

    DF(cholesky, 1),
    DF(cholesky, 2),
    DF(qr, 1),
    DF(qr, 2),

    F(type, 1),
    F(device, 1),
    F(nbytes, 1),
};

ERL_NIF_INIT(Elixir.Torchx.NIF, nif_functions, load, NULL, upgrade, NULL)
