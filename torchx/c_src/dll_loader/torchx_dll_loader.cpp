#include <stdlib.h>
#include <cstring>
#include <string>
#include <erl_nif.h>
#include <windows.h>
#include <libloaderapi.h>
#include <winbase.h>
#include <wchar.h>

#define NIF(NAME) ERL_NIF_TERM NAME(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])

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

NIF(add_dll_directory) {
  static bool path_updated = false;
  if (path_updated) return ok(env);

  if (argc != 1) {
    return enif_make_badarg(env);
  }

  ErlNifBinary path_bin;
  if (!enif_inspect_binary(env, argv[0], &path_bin) &&
      !enif_inspect_iolist_as_binary(env, argv[0], &path_bin)) {
    return error(env, "libtorch DLL directory must be a UTF-8 binary or iodata");
  }

  if (path_bin.size == 0) {
    return error(env, "libtorch DLL directory must not be empty");
  }

  int wide_len = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                     reinterpret_cast<LPCCH>(path_bin.data),
                                     static_cast<int>(path_bin.size), NULL, 0);
  if (wide_len <= 0) {
    return error(env, "libtorch DLL directory is not valid UTF-8");
  }

  std::wstring torch_dir(static_cast<size_t>(wide_len), L'\0');
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                          reinterpret_cast<LPCCH>(path_bin.data),
                          static_cast<int>(path_bin.size), &torch_dir[0],
                          wide_len) == 0) {
    return error(env, "failed to convert libtorch DLL directory to wide string");
  }

  PCWSTR directory_pcwstr = torch_dir.c_str();

  WCHAR path_buffer[65536];
  GetEnvironmentVariableW(L"PATH", path_buffer, 65536);
  WCHAR new_path[65536];
  new_path[0] = L'\0';
  wcscpy_s(new_path, _countof(new_path), (const wchar_t*)path_buffer);
  wcscat_s(new_path, _countof(new_path), (const wchar_t*)L";");
  wcscat_s(new_path, _countof(new_path), (const wchar_t*)directory_pcwstr);
  SetEnvironmentVariableW(L"PATH", new_path);

  SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
  DLL_DIRECTORY_COOKIE ret = AddDllDirectory(directory_pcwstr);
  if (ret == 0) {
    DWORD last_error = GetLastError();
    LPTSTR error_text = nullptr;
    FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, HRESULT_FROM_WIN32(last_error), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&error_text, 0, NULL);
    if (error_text != nullptr) {
      ERL_NIF_TERM ret_term = error(env, error_text);
      LocalFree(error_text);
      return ret_term;
    } else {
      ERL_NIF_TERM ret_term = error(env, "error happened when adding libtorch runtime path, but cannot get formatted error message");
      return ret_term;
    }
  }
  path_updated = true;
  return ok(env);
}

int upgrade(ErlNifEnv *env, void **priv_data, void **old_priv_data, ERL_NIF_TERM load_info) {
  // Silence "unused var" warnings.
  (void)(env);
  (void)(priv_data);
  (void)(old_priv_data);
  (void)(load_info);

  return 0;
}

int load(ErlNifEnv *,void **,ERL_NIF_TERM) {
  return 0;
}

#define F(NAME, ARITY)    \
  {                       \
#NAME, ARITY, NAME, 0 \
  }

static ErlNifFunc nif_functions[] = {
  F(add_dll_directory, 1)
};

ERL_NIF_INIT(Elixir.Torchx.NIF.DLLLoader, nif_functions, load, NULL, upgrade, NULL);
