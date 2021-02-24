#include <ATen/ATen.h>
#include <iostream>

#include "nx_nif_utils.hpp"

ErlNifResourceType *TENSOR_TYPE;

std::map<const std::string, const at::ScalarType> dtypes = {{"byte", at::kByte}, {"char", at::kChar}, {"short", at::kShort}, {"int", at::kInt}, {"long", at::kLong}, {"half", at::kHalf}, {"brain", at::kBFloat16}, {"float", at::kFloat}, {"double", at::kDouble}, {"bool", at::kBool}};
std::map<const std::string, const int> dtype_sizes = {{"byte", 1}, {"char", 1}, {"short", 2}, {"int", 4}, {"long", 8}, {"half", 2}, {"brain", 2}, {"float", 4}, {"double", 8}};

ERL_NIF_TERM create_tensor_resource(ErlNifEnv *env, at::Tensor tensor)
{
  ERL_NIF_TERM ret;
  at::Tensor *tensorPtr;

  tensorPtr = (at::Tensor *)enif_alloc_resource(TENSOR_TYPE, sizeof(at::Tensor));
  if (tensorPtr == NULL)
    return enif_make_badarg(env);

  new (tensorPtr) at::Tensor(tensor.variable_data());

  ret = enif_make_resource(env, tensorPtr);
  enif_release_resource(tensorPtr);

  return nx::nif::ok(env, ret);
}

at::Tensor *get_tensor(ErlNifEnv *env, ERL_NIF_TERM term)
{
  at::Tensor *tensorPtr;
  if (!enif_get_resource(env, term, TENSOR_TYPE, (void **)&tensorPtr))
  {
    return NULL; //&(at::ones({1, 1}, at::kFloat);
  }
  else
  {
    return tensorPtr;
  }
}

ERL_NIF_TERM delete_tensor(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  at::Tensor *t = get_tensor(env, argv[0]);

  delete t;
  enif_release_resource(t);

  return nx::nif::ok(env);
}

unsigned long elem_count(std::vector<int64_t> shape)
{
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>{});
}

ERL_NIF_TERM from_blob(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  ErlNifBinary blob;
  std::vector<int64_t> shape;
  std::string type;

  BINARY_PARAM(0, blob);
  TUPLE_PARAM(1, shape);
  ATOM_PARAM(2, type);

  if (blob.size / dtype_sizes[type] < elem_count(shape))
    return enif_make_badarg(env);

  // Clone here to copy data from blob, which will be GCed.
  at::Tensor t = at::clone(at::from_blob(blob.data, shape, dtypes[type]));

  return create_tensor_resource(env, t);
}

ERL_NIF_TERM to_blob(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  ERL_NIF_TERM result;
  at::Tensor *t = get_tensor(env, argv[0]);
  int64_t limit = t->nbytes();

  if (argc == 2)
    PARAM(1, limit);

  int64_t byte_size = limit * t->itemsize();

  void *result_data = (void *)enif_make_new_binary(env, byte_size, &result);
  memcpy(result_data, t->data_ptr(), byte_size);

  return result;
}

ERL_NIF_TERM scalar_tensor(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  double scalar;
  std::string type;

  PARAM(0, scalar);
  ATOM_PARAM(1, type);

  at::Tensor t = at::scalar_tensor(scalar, dtypes[type]);

  return create_tensor_resource(env, t);
}

ERL_NIF_TERM randint(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  int64_t min, max;
  std::string type;
  std::vector<int64_t> shape;

  PARAM(0, min);
  PARAM(1, max);
  TUPLE_PARAM(2, shape);
  ATOM_PARAM(3, type);

  at::Tensor t = at::randint(min, max, shape, dtypes[type]);

  return create_tensor_resource(env, t);
}

ERL_NIF_TERM rand(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  double min, max;
  std::vector<int64_t> shape;
  std::string type;

  PARAM(0, min);
  PARAM(1, max);
  TUPLE_PARAM(2, shape);
  ATOM_PARAM(3, type);

  at::Tensor t = min + at::rand(shape, dtypes[type]) * (max - min);

  return create_tensor_resource(env, t);
}

ERL_NIF_TERM normal(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  double mean, std;
  std::vector<int64_t> shape;
  std::string type;

  PARAM(0, mean);
  PARAM(1, std);
  TUPLE_PARAM(2, shape);
  ATOM_PARAM(3, type);

  at::Tensor t = at::normal(mean, std, shape);

  return create_tensor_resource(env, t);
}

ERL_NIF_TERM arange(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  int64_t start, end, step;
  std::string type;

  PARAM(0, start);
  PARAM(1, end);
  PARAM(2, step);
  ATOM_PARAM(3, type);

  at::Tensor t = at::arange((double)start, (double)end, (double)step, dtypes[type]);

  if (argc == 5)
  {
    std::vector<int64_t> shape;
    TUPLE_PARAM(4, shape);

    t = at::reshape(t, shape);
  }

  return create_tensor_resource(env, t);
}

ERL_NIF_TERM reshape(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  std::vector<int64_t> shape;
  at::Tensor *t = get_tensor(env, argv[0]);
  TUPLE_PARAM(1, shape);

  at::Tensor r = at::reshape(*t, shape);

  return create_tensor_resource(env, r);
}

ERL_NIF_TERM squeeze(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  at::Tensor *t = get_tensor(env, argv[0]);
  at::Tensor r;

  if (argc == 2)
  {
    int64_t dim;
    PARAM(1, dim);

    r = at::squeeze(*t, dim);
  }
  else
    r = at::squeeze(*t);

  return create_tensor_resource(env, r);
}

ERL_NIF_TERM broadcast_to(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  std::vector<int64_t> shape;
  at::Tensor *t = get_tensor(env, argv[0]);
  TUPLE_PARAM(1, shape);

  at::Tensor r = at::broadcast_to(*t, shape).clone();

  return create_tensor_resource(env, r);
}

ERL_NIF_TERM ones(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  std::string type;
  std::vector<int64_t> shape;

  TUPLE_PARAM(0, shape);
  ATOM_PARAM(1, type);

  at::Tensor t = at::ones(shape, dtypes[type]);

  return create_tensor_resource(env, t);
}

ERL_NIF_TERM eye(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  int64_t size;
  std::string type;

  PARAM(0, size);
  ATOM_PARAM(1, type);

  at::Tensor t = at::eye(size, dtypes[type]);

  return create_tensor_resource(env, t);
}

ERL_NIF_TERM add(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  at::Tensor *a = get_tensor(env, argv[0]);
  at::Tensor *b = get_tensor(env, argv[1]);
  double scalar = 0.0;

  try
  {
    at::Tensor c;
    if (b == NULL)
    {
      PARAM(1, scalar);
      c = *a + scalar;
    }
    else
      c = *a + *b;

    return create_tensor_resource(env, c);
  }
  catch (c10::Error error)
  {
    return enif_raise_exception(env, enif_make_string(env,
                                                      error.msg().c_str(), ERL_NIF_LATIN1));
  }
}

ERL_NIF_TERM dot(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  at::Tensor *a = get_tensor(env, argv[0]);
  at::Tensor *b = get_tensor(env, argv[1]);

  try
  {
    at::Tensor c = at::matmul(*a, *b);

    return create_tensor_resource(env, c);
  }
  catch (c10::Error error)
  {
    return enif_raise_exception(env, enif_make_string(env,
                                                      (std::string("PyTorch: ") + error.msg()).c_str(), ERL_NIF_LATIN1));
  }
}

ERL_NIF_TERM cholesky(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  at::Tensor *t = get_tensor(env, argv[0]);
  bool upper = false;

  if (argc == 2)
  {
    PARAM(1, upper);
  }

  at::Tensor r = at::cholesky(*t, upper);

  return create_tensor_resource(env, r);
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

static ErlNifFunc nif_functions[] = {
    {"randint", 4, randint, 0},
    {"rand", 4, rand, 0},
    {"normal", 4, normal, 0},
    {"arange", 4, arange, 0},
    {"arange", 5, arange, 0},
    {"from_blob", 3, from_blob, 0},
    {"to_blob", 1, to_blob, 0},
    {"to_blob", 2, to_blob, 0},
    {"scalar_tensor", 2, scalar_tensor, 0},
    {"delete_tensor", 1, delete_tensor, 0},
    {"ones", 1, ones, 0},
    {"eye", 2, eye, 0},
    {"reshape", 2, reshape, 0},
    {"squeeze", 2, squeeze, 0},
    {"squeeze", 1, squeeze, 0},
    {"broadcast_to", 2, broadcast_to, 0},
    {"add", 2, add, 0},
    {"dot", 2, dot, 0},
    {"cholesky", 1, cholesky, 0},
    {"cholesky", 2, cholesky, 0}};

ERL_NIF_INIT(Elixir.Torchx.NIF, nif_functions, load, NULL, upgrade, NULL)
