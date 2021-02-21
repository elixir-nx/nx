#include <ATen/ATen.h>
#include <iostream>

#include "nx_nif_utils.hpp"

ErlNifResourceType *TENSOR_TYPE;

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

double get_double(ErlNifEnv *env, ERL_NIF_TERM term)
{
  double var;
  enif_get_double(env, term, &var);

  return var;
}

double get_long(ErlNifEnv *env, ERL_NIF_TERM term)
{
  long var;
  enif_get_long(env, term, &var);

  return var;
}

std::vector<int64_t> shape_from_tuple(ErlNifEnv *env, ERL_NIF_TERM tuple)
{
  const ERL_NIF_TERM *shape;
  int shape_size;
  std::vector<int64_t> shape_array;
  int64_t shape_elem;

  if (!enif_get_tuple(env, tuple, &shape_size, &shape))
    // Signal error here. Raise?
    // return enif_make_badarg(env);
    ;

  for (int i = 0; i < shape_size; i++)
  {
    enif_get_int64(env, shape[i], (long *)&shape_elem);
    shape_array.push_back(shape_elem);
  }

  return shape_array;
}

unsigned long elem_count(std::vector<int64_t> shape)
{
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>{});
}

ERL_NIF_TERM
from_blob(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  ErlNifBinary blob;
  std::vector<int64_t> shape;

  nx::nif::get_tuple(env, argv[1], shape);

  if (!enif_inspect_binary(env, argv[0], &blob))
    return enif_make_badarg(env);

  if (blob.size / sizeof(float) < elem_count(shape))
    return enif_make_badarg(env);

  // float blob_array[] = {1, 2, 3.2342, 4, 5, 6, 7};
  // auto options = at::TensorOptions().dtype(at::kFloat); // .device(torch::kCUDA, 1);

  // Clone here to copy data from blob, which will be GCed.
  at::Tensor t = at::clone(at::from_blob(blob.data, c10::IntArrayRef(shape), at::kFloat));

  // at::ones(c10::IntArrayRef(shape_array, shape_size), at::kFloat);
  std::cout << t << "\r\n";

  return create_tensor_resource(env, t);
}

std::map<std::string, at::ScalarType> types = {{"byte", at::kByte}, {"short", at::kShort}, {"int", at::kInt}, {"long", at::kLong}};

ERL_NIF_TERM randint(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  long min = get_long(env, argv[0]);
  long max = get_long(env, argv[1]);
  std::string type;
  std::vector<int64_t> shape;

  nx::nif::get_tuple(env, argv[2], shape);
  nx::nif::get_atom(env, argv[3], type);

  at::Tensor t = at::randint(min, max, shape, types[type]);

  std::cout << t << "\r\n";

  return create_tensor_resource(env, t);
}

ERL_NIF_TERM ones(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  at::Tensor t = at::ones(c10::IntArrayRef(shape_from_tuple(env, argv[0])), at::kFloat);

  return create_tensor_resource(env, t);
}

ERL_NIF_TERM add(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  at::Tensor *a = get_tensor(env, argv[0]);
  at::Tensor *b = get_tensor(env, argv[1]);
  double scalar = 0.0;

  std::cerr << *a << "\r\n";
  if (b == NULL)
  {
    scalar = get_double(env, argv[1]);
    std::cout << "Adding scalar " << scalar << "\n";
  }
  else
  {

    std::cout << *b << "\r\n";
  }
  try
  {
    at::Tensor c;
    if (b == NULL)
    {
      c = *a + scalar;
    }
    else
      c = *a + *b;

    std::cout
        << c << "\r\n";

    return create_tensor_resource(env, c);
  }
  catch (c10::Error error)
  {
    return enif_raise_exception(env, enif_make_string(env,
                                                      error.msg().c_str(), ERL_NIF_LATIN1));
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
    {"from_blob", 2, from_blob, 0},
    {"ones", 1, ones, 0},
    {"add",
     2,
     add,
     0}};

ERL_NIF_INIT(Elixir.Nx.Pytorch.NIF, nif_functions, load, NULL, upgrade, NULL)
