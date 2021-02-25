#include <ATen/ATen.h>
#include <iostream>

#include "nx_nif_utils.hpp"

ErlNifResourceType *TENSOR_TYPE;

std::map<const std::string, const at::ScalarType> dtypes = {{"byte", at::kByte}, {"char", at::kChar}, {"short", at::kShort}, {"int", at::kInt}, {"long", at::kLong}, {"half", at::kHalf}, {"brain", at::kBFloat16}, {"float", at::kFloat}, {"double", at::kDouble}, {"bool", at::kBool}};
std::map<const std::string, const int> dtype_sizes = {{"byte", 1}, {"char", 1}, {"short", 2}, {"int", 4}, {"long", 8}, {"half", 2}, {"brain", 2}, {"float", 4}, {"double", 8}};

inline at::ScalarType string2type(const std::string atom)
{
  return dtypes[atom];
}

inline std::string type2string(const at::ScalarType type)
{
  for (std::map<const std::string, const at::ScalarType>::iterator i = dtypes.begin(); i != dtypes.end(); ++i)
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
  at::ScalarType VAR = string2type(VAR##_atom)

#define TENSOR_PARAM(ARGN, VAR)                                        \
  at::Tensor *VAR;                                                     \
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
    std::vector<at::Tensor> tl = TL;                                                           \
    std::vector<ERL_NIF_TERM> res_list;                                                        \
    for (at::Tensor t : tl)                                                                    \
      res_list.push_back(create_tensor_resource(env, t));                                      \
    return nx::nif::ok(env, enif_make_list_from_array(env, res_list.data(), res_list.size())); \
  }                                                                                            \
  catch (c10::Error error)                                                                     \
  {                                                                                            \
    return nx::nif::error(env, error.msg().c_str());                                           \
  }

ERL_NIF_TERM
create_tensor_resource(ErlNifEnv *env, at::Tensor tensor)
{
  ERL_NIF_TERM ret;
  at::Tensor *tensorPtr;

  tensorPtr = (at::Tensor *)enif_alloc_resource(TENSOR_TYPE, sizeof(at::Tensor));
  if (tensorPtr == NULL)
    return enif_make_badarg(env);

  new (tensorPtr) at::Tensor(tensor.variable_data());

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
  TENSOR(at::clone(at::from_blob(blob.data, shape, type)));
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

NIF(split)
{
  TENSOR_PARAM(0, t);
  PARAM(1, int64_t, batch_size);

  TENSOR_LIST(at::split(*t, batch_size));
}

NIF(to_blob)
{
  ERL_NIF_TERM result;
  TENSOR_PARAM(0, t);
  int64_t byte_size = t->nbytes();

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

  TENSOR(at::scalar_tensor(scalar, type));
}

NIF(randint)
{
  PARAM(0, int64_t, min);
  PARAM(1, int64_t, max);
  SHAPE_PARAM(2, shape);
  TYPE_PARAM(3, type);

  TENSOR(at::randint(min, max, shape, type));
}

NIF(rand)
{
  PARAM(0, double, min);
  PARAM(1, double, max);
  SHAPE_PARAM(2, shape);
  TYPE_PARAM(3, type);

  TENSOR(min + at::rand(shape, type) * (max - min));
}

NIF(normal)
{
  PARAM(0, double, mean);
  PARAM(1, double, std);
  SHAPE_PARAM(2, shape);

  TENSOR(at::normal(mean, std, shape));
}

NIF(arange)
{
  PARAM(0, int64_t, start);
  PARAM(1, int64_t, end);
  PARAM(2, int64_t, step);
  TYPE_PARAM(3, type);

  at::Tensor t = at::arange((double)start, (double)end, (double)step, type);

  if (argc == 5)
  {
    SHAPE_PARAM(4, shape);
    t = at::reshape(t, shape);
  }

  TENSOR(t);
}

NIF(reshape)
{
  TENSOR_PARAM(0, t);
  SHAPE_PARAM(1, shape);

  TENSOR(at::reshape(*t, shape));
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
    TENSOR(at::squeeze(*t, dim));
  }
  else
    TENSOR(at::squeeze(*t));
}

NIF(broadcast_to)
{
  TENSOR_PARAM(0, t);
  SHAPE_PARAM(1, shape);

  TENSOR(at::broadcast_to(*t, shape).clone());
}

NIF(transpose)
{
  TENSOR_PARAM(0, t);
  PARAM(1, int64_t, dim0);
  PARAM(2, int64_t, dim1);

  TENSOR(at::transpose(*t, dim0, dim1));
}

NIF(ones)
{
  SHAPE_PARAM(0, shape);
  TYPE_PARAM(1, type);

  TENSOR(at::ones(shape, type));
}

NIF(eye)
{
  PARAM(0, int64_t, size);
  TYPE_PARAM(1, type);

  TENSOR(at::eye(size, type));
}

NIF(add)
{
  TENSOR_PARAM(0, a);
  TENSOR_PARAM(1, b);

  if (b == NULL)
  {
    PARAM(1, double, scalar);
    TENSOR(*a + scalar);
  }
  else
    TENSOR(*a + *b);
}

NIF(dot)
{
  TENSOR_PARAM(0, a);
  TENSOR_PARAM(1, b);

  TENSOR(at::matmul(*a, *b));
}

NIF(cholesky)
{
  TENSOR_PARAM(0, t);
  bool upper = false;

  if (argc == 2)
  {
    GET(1, upper);
  }

  TENSOR(at::cholesky(*t, upper));
}

void free_tensor(ErlNifEnv *env, void *obj)
{
  std::cout << "Deleting: " << obj << std::endl;
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

static ErlNifFunc nif_functions[] = {
    F(randint, 4),
    F(rand, 4),
    F(normal, 3),
    F(arange, 4),
    F(arange, 5),
    F(from_blob, 3),
    F(to_blob, 1),
    F(to_blob, 2),
    F(type, 1),
    F(scalar_tensor, 2),
    F(delete_tensor, 1),
    F(ones, 1),
    F(eye, 2),
    F(reshape, 2),
    F(split, 2),
    F(to_type, 2),
    F(squeeze, 2),
    F(squeeze, 1),
    F(broadcast_to, 2),
    F(transpose, 3),
    F(add, 2),
    F(dot, 2),
    F(cholesky, 1),
    F(cholesky, 2)};

ERL_NIF_INIT(Elixir.Torchx.NIF, nif_functions, load, NULL, upgrade, NULL)
