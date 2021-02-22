#include <ATen/ATen.h>
#include <iostream>

#include "nx_nif_utils.hpp"

ErlNifResourceType *TENSOR_TYPE;

std::map<const std::string, const at::ScalarType> dtypes = {{"byte", at::kByte}, {"char", at::kChar}, {"short", at::kShort}, {"int", at::kInt}, {"long", at::kLong}, {"half", at::kHalf}, {"float", at::kFloat}, {"double", at::kDouble}};
std::map<const std::string, const int> dtype_sizes = {{"byte", 1}, {"char", 1}, {"short", 2}, {"int", 4}, {"long", 8}, {"half", 2}, {"float", 4}, {"double", 8}};

ERL_NIF_TERM create_tensor_resource(ErlNifEnv *env, at::Tensor tensor)
{
  ERL_NIF_TERM ret;
  at::Tensor *tensorPtr;

  tensorPtr = (at::Tensor *)enif_alloc_resource(TENSOR_TYPE, sizeof(at::Tensor));
  if (tensorPtr == NULL)
    return enif_make_badarg(env);

  new (tensorPtr) at::Tensor(tensor.variable_data());

  ret = enif_make_resource(env, tensorPtr);
  enif_keep_resource(tensorPtr);
  enif_release_resource(tensorPtr);

  return ret;
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

unsigned long elem_count(std::vector<int64_t> shape)
{
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>{});
}

ERL_NIF_TERM from_blob(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  ErlNifBinary blob;
  std::vector<int64_t> shape;
  std::string type;

  if (!nx::nif::get_tuple(env, argv[1], shape))
    return enif_make_badarg(env);

  if (!nx::nif::get_atom(env, argv[2], type))
    return enif_make_badarg(env);

  if (!enif_inspect_binary(env, argv[0], &blob))
    return enif_make_badarg(env);

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

  void *result_data = (void *)enif_make_new_binary(env, t->nbytes(), &result);
  memcpy(result_data, t->data_ptr(), t->nbytes());

  return result;
}

ERL_NIF_TERM randint(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  int64_t min, max;
  std::string type;
  std::vector<int64_t> shape;

  nx::nif::get(env, argv[0], &min);
  nx::nif::get(env, argv[1], &max);
  nx::nif::get_tuple(env, argv[2], shape);
  nx::nif::get_atom(env, argv[3], type);

  at::Tensor t = at::randint(min, max, shape, dtypes[type]);

  std::cout << t << "\r\n";

  return create_tensor_resource(env, t);
}

ERL_NIF_TERM rand(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  double min, max;
  std::vector<int64_t> shape;
  std::string type;

  nx::nif::get(env, argv[0], &min);
  nx::nif::get(env, argv[1], &max);
  nx::nif::get_tuple(env, argv[2], shape);
  nx::nif::get_atom(env, argv[3], type);

  at::Tensor t = min + at::rand(shape, dtypes[type]) * (max - min);

  return create_tensor_resource(env, t);
}

ERL_NIF_TERM normal(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  double mean, std;
  std::vector<int64_t> shape;
  std::string type;

  nx::nif::get(env, argv[0], &mean);
  nx::nif::get(env, argv[1], &std);
  nx::nif::get_tuple(env, argv[2], shape);
  nx::nif::get_atom(env, argv[3], type);

  at::Tensor t = at::normal(mean, std, shape);

  return create_tensor_resource(env, t);
}

ERL_NIF_TERM arange(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  int64_t start, end, step;
  std::string type;

  nx::nif::get(env, argv[0], &start);
  nx::nif::get(env, argv[1], &end);
  nx::nif::get(env, argv[2], &step);
  nx::nif::get_atom(env, argv[3], type);

  at::Tensor t = at::arange((double)start, (double)end, (double)step, dtypes[type]);

  if (argc == 5)
  {
    std::vector<int64_t> shape;
    nx::nif::get_tuple(env, argv[4], shape);

    t = at::reshape(t, shape);
  }

  return create_tensor_resource(env, t);
}

ERL_NIF_TERM reshape(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  std::vector<int64_t> shape;
  at::Tensor *t = get_tensor(env, argv[0]);
  nx::nif::get_tuple(env, argv[1], shape);

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
    nx::nif::get(env, argv[1], &dim);

    r = at::squeeze(*t, dim);
  }
  else
    r = at::squeeze(*t);

  return create_tensor_resource(env, r);
}

ERL_NIF_TERM ones(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  std::string type;
  std::vector<int64_t> shape;

  nx::nif::get_tuple(env, argv[0], shape);
  nx::nif::get_atom(env, argv[1], type);

  at::Tensor t = at::ones(shape, dtypes[type]);

  return create_tensor_resource(env, t);
}

ERL_NIF_TERM add(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  at::Tensor *a = get_tensor(env, argv[0]);
  at::Tensor *b = get_tensor(env, argv[1]);
  double scalar = 0.0;

  if (b == NULL)
  {
    nx::nif::get(env, argv[1], &scalar);
  }

  try
  {
    at::Tensor c;
    if (b == NULL)
      c = *a + scalar;
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

// Used for RNG initialization.
int load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info)
{
  // Silence "unused var" warnings.
  if (open_resource_type(env) == -1)
    return -1;
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
    {"ones", 1, ones, 0},
    {"reshape", 2, reshape, 0},
    {"squeeze", 2, squeeze, 0},
    {"squeeze", 1, squeeze, 0},
    {"add", 2, add, 0},
    {"dot", 2, dot, 0}};

ERL_NIF_INIT(Elixir.Nx.Pytorch.NIF, nif_functions, load, NULL, upgrade, NULL)
