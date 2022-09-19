#pragma once

#include "erl_nif.h"

ErlNifResourceType *TENSOR_TYPE;

#define GET(ARGN, VAR)                      \
  if (!nx::nif::get(env, argv[ARGN], &VAR)) \
    return nx::nif::error(env, "Unable to get " #VAR " param.");

#define PARAM(ARGN, TYPE, VAR) \
  TYPE VAR;                    \
  GET(ARGN, VAR)

#define ATOM_PARAM(ARGN, VAR)                   \
  std::string VAR;                              \
  if (!nx::nif::get_atom(env, argv[ARGN], VAR)) \
    return nx::nif::error(env, "Unable to get " #VAR " atom param.");

#define TUPLE_PARAM(ARGN, TYPE, VAR)                 \
  TYPE VAR;                                          \
  if (!nx::nif::get_tuple(env, argv[ARGN], VAR))  {  \
    std::ostringstream msg;                          \
    msg << "Unable to get " #VAR " tuple param in NIF." << __func__ << "/" << argc; \
    return nx::nif::error(env, msg.str().c_str());    \
  }

#define LIST_PARAM(ARGN, TYPE, VAR)             \
  TYPE VAR;                                      \
  if (!nx::nif::get_list(env, argv[ARGN], VAR)) \
    return nx::nif::error(env, "Unable to get " #VAR " list param.");

#define BINARY_PARAM(ARGN, VAR)                    \
  ErlNifBinary VAR;                                \
  if (!enif_inspect_binary(env, argv[ARGN], &VAR)) \
    return nx::nif::error(env, "Unable to get " #VAR " binary param.");

namespace nx
{
  namespace nif
  {
    // Status helpers

    // Helper for returning `{:error, msg}` from NIF.
    ERL_NIF_TERM error(ErlNifEnv *env, const char *msg)
    {
      ERL_NIF_TERM atom = enif_make_atom(env, "error");
      ERL_NIF_TERM msg_term = enif_make_string(env, msg, ERL_NIF_LATIN1);
      return enif_make_tuple2(env, atom, msg_term);
    }

    // Helper for returning `{:ok, term}` from NIF.
    ERL_NIF_TERM ok(ErlNifEnv *env)
    {
      return enif_make_atom(env, "ok");
    }

    // Helper for returning `:ok` from NIF.
    ERL_NIF_TERM ok(ErlNifEnv *env, ERL_NIF_TERM term)
    {
      return enif_make_tuple2(env, ok(env), term);
    }

    // Numeric types

    int get(ErlNifEnv *env, ERL_NIF_TERM term, int *var)
    {
      return enif_get_int(env, term,
                          reinterpret_cast<int *>(var));
    }

    int get(ErlNifEnv *env, ERL_NIF_TERM term, int64_t *var)
    {
      return enif_get_int64(env, term,
                            reinterpret_cast<ErlNifSInt64 *>(var));
    }

    int get(ErlNifEnv *env, ERL_NIF_TERM term, double *var)
    {
      return enif_get_double(env, term, var);
    }

    // Standard types

    int get(ErlNifEnv *env, ERL_NIF_TERM term, std::string &var)
    {
      unsigned len;
      int ret = enif_get_list_length(env, term, &len);

      if (!ret)
      {
        ErlNifBinary bin;
        ret = enif_inspect_binary(env, term, &bin);
        if (!ret)
        {
          return 0;
        }
        var = std::string((const char *)bin.data, bin.size);
        return ret;
      }

      var.resize(len + 1);
      ret = enif_get_string(env, term, &*(var.begin()), var.size(), ERL_NIF_LATIN1);

      if (ret > 0)
      {
        var.resize(ret - 1);
      }
      else if (ret == 0)
      {
        var.resize(0);
      }
      else
      {
      }

      return ret;
    }

    ERL_NIF_TERM make(ErlNifEnv *env, bool var)
    {
      if (var)
        return enif_make_atom(env, "true");

      return enif_make_atom(env, "false");
    }

    ERL_NIF_TERM make(ErlNifEnv *env, int64_t var)
    {
      return enif_make_int64(env, var);
    }

    ERL_NIF_TERM make(ErlNifEnv *env, int var)
    {
      return enif_make_int(env, var);
    }

    ERL_NIF_TERM make(ErlNifEnv *env, double var)
    {
      return enif_make_double(env, var);
    }

    ERL_NIF_TERM make(ErlNifEnv *env, ErlNifBinary var)
    {
      return enif_make_binary(env, &var);
    }

    ERL_NIF_TERM make(ErlNifEnv *env, std::string var)
    {
      return enif_make_string(env, var.c_str(), ERL_NIF_LATIN1);
    }

    ERL_NIF_TERM make(ErlNifEnv *env, const char *string)
    {
      return enif_make_string(env, string, ERL_NIF_LATIN1);
    }

    // Atoms

    int get_atom(ErlNifEnv *env, ERL_NIF_TERM term, std::string &var)
    {
      unsigned atom_length;
      if (!enif_get_atom_length(env, term, &atom_length, ERL_NIF_LATIN1))
      {
        return 0;
      }

      var.resize(atom_length + 1);

      if (!enif_get_atom(env, term, &(*(var.begin())), var.size(), ERL_NIF_LATIN1))
        return 0;

      var.resize(atom_length);

      return 1;
    }

    ERL_NIF_TERM atom(ErlNifEnv *env, const char *msg)
    {
      return enif_make_atom(env, msg);
    }

    // Boolean

    int get(ErlNifEnv *env, ERL_NIF_TERM term, bool *var)
    {
      std::string bool_atom;
      if (!get_atom(env, term, bool_atom))
        return 0;

      if (bool_atom == "true")
        *var = true;
      else if (bool_atom == "false")
        *var = false;
      else
        return 0; // error

      return 1;
    }

    // Containers

    template <typename T = int64_t>
    int get_tuple(ErlNifEnv *env, ERL_NIF_TERM tuple, std::vector<T> &var)
    {
      const ERL_NIF_TERM *terms;
      int length;
      if (!enif_get_tuple(env, tuple, &length, &terms))
        return 0;
      var.reserve(length);

      for (int i = 0; i < length; i++)
      {
        T data;
        if (!get(env, terms[i], &data))
          return 0;
        var.push_back(data);
      }
      return 1;
    }

    int get_list(ErlNifEnv *env,
                 ERL_NIF_TERM list,
                 std::vector<ErlNifBinary> &var)
    {
      unsigned int length;
      if (!enif_get_list_length(env, list, &length))
        return 0;
      var.reserve(length);
      ERL_NIF_TERM head, tail;

      while (enif_get_list_cell(env, list, &head, &tail))
      {
        ErlNifBinary elem;
        if (!enif_inspect_binary(env, head, &elem))
          return 0;
        var.push_back(elem);
        list = tail;
      }
      return 1;
    }

    int get_list(ErlNifEnv *env,
                 ERL_NIF_TERM list,
                 std::vector<std::string> &var)
    {
      unsigned int length;
      if (!enif_get_list_length(env, list, &length))
        return 0;
      var.reserve(length);
      ERL_NIF_TERM head, tail;

      while (enif_get_list_cell(env, list, &head, &tail))
      {
        std::string elem;
        if (!get_atom(env, head, elem))
          return 0;
        var.push_back(elem);
        list = tail;
      }
      return 1;
    }

    int get_list(ErlNifEnv *env, ERL_NIF_TERM list, std::vector<int64_t> &var)
    {
      unsigned int length;
      if (!enif_get_list_length(env, list, &length))
        return 0;
      var.reserve(length);
      ERL_NIF_TERM head, tail;

      while (enif_get_list_cell(env, list, &head, &tail))
      {
        int64_t elem;
        if (!get(env, head, &elem))
          return 0;
        var.push_back(elem);
        list = tail;
      }
      return 1;
    }

    int get_list(ErlNifEnv *env, ERL_NIF_TERM list, std::vector<torch::Tensor> &var)
    {
      unsigned int length;
      if (!enif_get_list_length(env, list, &length))
        return 0;
      var.reserve(length);
      ERL_NIF_TERM head, tail;

      while (enif_get_list_cell(env, list, &head, &tail))
      {
        torch::Tensor *elem;
        if (!enif_get_resource(env, head, TENSOR_TYPE, reinterpret_cast<void **>(&elem)))
        {
          return 0;
        }
        var.push_back(*elem);
        list = tail;
      }
      return 1;
    }
  }
}