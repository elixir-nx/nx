#include "tensorflow/compiler/xla/exla/exla_nif_util.h"

namespace exla {

  ERL_NIF_TERM error(ErlNifEnv* env, const char* msg) {
    ERL_NIF_TERM atom = enif_make_atom(env, "error");
    ERL_NIF_TERM msg_term = enif_make_string(env, msg, ERL_NIF_LATIN1);
    return enif_make_tuple2(env, atom, msg_term);
  }

  ERL_NIF_TERM ok(ErlNifEnv* env) {
    return enif_make_atom(env, "ok");
  }

  ERL_NIF_TERM ok(ErlNifEnv* env, ERL_NIF_TERM term) {
    return enif_make_tuple2(env, ok(env), term);
  }

  ERL_NIF_TERM atom(ErlNifEnv* env, const char* msg) {
    return enif_make_atom(env, msg);
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, int8* var) {
    int value;
    if (!enif_get_int(env, term, &value)) return 0;
    *var = static_cast<int8>(value);
    return 1;
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, int16* var) {
    int value;
    if (!enif_get_int(env, term, &value)) return 0;
    *var = static_cast<int16>(value);
    return 1;
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, int32* var) {
    return enif_get_int(env, term,
                        reinterpret_cast<int *>(var));
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, int64* var) {
    return enif_get_int64(env, term,
                          reinterpret_cast<long long int*>(var));
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, uint8* var) {
    unsigned int value;
    if (!enif_get_uint(env, term, &value)) return 0;
    *var = static_cast<uint8>(value);
    return 1;
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, uint16* var) {
    unsigned int value;
    if (!enif_get_uint(env, term, &value)) return 0;
    *var = static_cast<uint16>(value);
    return 1;
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, uint32* var) {
    return enif_get_uint(env, term,
                         reinterpret_cast<unsigned int*>(var));
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, uint64* var) {
    return enif_get_uint64(env, term,
                           reinterpret_cast<unsigned long long int*>(var));
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, bfloat16* var) {
    double value;
    if (!enif_get_double(env, term, &value)) return 0;
    *var = static_cast<bfloat16>(value);
    return 1;
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, float32* var) {
    double value;
    if (!enif_get_double(env, term, &value)) return 0;
    *var = static_cast<float32>(value);
    return 1;
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, float64* var) {
    return enif_get_double(env, term, var);
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, std::string &var) {
    unsigned len;
    int ret = enif_get_list_length(env, term, &len);

    if (!ret) {
      ErlNifBinary bin;
      ret = enif_inspect_binary(env, term, &bin);
      if (!ret) {
        return 0;
      }
      var = std::string((const char*)bin.data, bin.size);
      return ret;
    }

    var.resize(len+1);
    ret = enif_get_string(env, term, &*(var.begin()), var.size(), ERL_NIF_LATIN1);

    if (ret > 0) {
      var.resize(ret-1);
    } else if (ret == 0) {
      var.resize(0);
    } else {}

    return ret;
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, bool* var) {
    int value;
    if (!enif_get_int(env, term, &value)) return 0;
    *var = static_cast<bool>(value);
    return 1;
  }

  int get_binary(ErlNifEnv* env, ERL_NIF_TERM term, ErlNifBinary* var) {
    return enif_inspect_binary(env, term, var);
  }

  int get_type(ErlNifEnv* env, ERL_NIF_TERM term, xla::PrimitiveType &type) {
    std::string type_str;
    if (!get(env, term, type_str)) return 0;

    xla::StatusOr<xla::PrimitiveType> type_status =
      xla::primitive_util::StringToPrimitiveType(type_str);

    if (!type_status.ok()) {
      return 0;
    }
    type = type_status.ConsumeValueOrDie();
    return 1;
  }

  int get_atom(ErlNifEnv* env, ERL_NIF_TERM term, std::string &var) {
    unsigned atom_length;
    if (!enif_get_atom_length(env, term, &atom_length, ERL_NIF_LATIN1)) {
      return 0;
    }

    var.resize(atom_length+1);

    if (!enif_get_atom(env, term, &(*(var.begin())), var.size(), ERL_NIF_LATIN1)) return 0;

    var.resize(atom_length);

    return 1;
  }

  int get_tuple(ErlNifEnv* env, ERL_NIF_TERM tuple, std::vector<int64> &var) {
    const ERL_NIF_TERM *terms;
    int length;
    if (!enif_get_tuple(env, tuple, &length, &terms)) return 0;
    var.reserve(length);

    for (int i=0; i < length; i++) {
      int data;
      if (!get(env, terms[i], &data)) return 0;
      var.emplace_back(data);
    }
    return 1;
  }

  int get_list(ErlNifEnv* env,
               ERL_NIF_TERM list,
               std::vector<ErlNifBinary> &var) {
    uint32 length;
    if (!enif_get_list_length(env, list, &length)) return 0;
    var.reserve(length);
    ERL_NIF_TERM head, tail;

    while (enif_get_list_cell(env, list, &head, &tail)) {
      ErlNifBinary elem;
      if (!get_binary(env, head, &elem)) return 0;
      var.emplace_back(elem);
      list = tail;
    }
    return 1;
  }

  int get_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<int64> &var) {
    uint32 length;
    if (!enif_get_list_length(env, list, &length)) return 0;
    var.reserve(length);
    ERL_NIF_TERM head, tail;

    while (enif_get_list_cell(env, list, &head, &tail)) {
      int64 elem;
      if (!get(env, head, &elem)) return 0;
      var.emplace_back(elem);
      list = tail;
    }
    return 1;
  }

  int get_padding_config(ErlNifEnv* env,
                         ERL_NIF_TERM list,
                         xla::PaddingConfig* padding_config) {
    ERL_NIF_TERM head, tail;
    while (enif_get_list_cell(env, list, &head, &tail)) {
      const ERL_NIF_TERM* terms;
      int length;
      if (!enif_get_tuple(env, head, &length, &terms)) return 0;
      if (!length == 3) return 0;

      int64 pad_lo, pad_hi, interior;
      if (!get(env, terms[0], &pad_lo)) return 0;
      if (!get(env, terms[1], &pad_hi)) return 0;
      if (!get(env, terms[2], &interior)) return 0;

      auto dim = padding_config->add_dimensions();
      dim->set_edge_padding_low(pad_lo);
      dim->set_edge_padding_high(pad_hi);
      dim->set_interior_padding(interior);

      list = tail;
    }
    return 1;
  }

  int get_dot_dimension_numbers(ErlNifEnv* env,
                                ERL_NIF_TERM tuple,
                                xla::DotDimensionNumbers* dims) {
    const ERL_NIF_TERM* terms;
    int32 count;
    if (!enif_get_tuple(env, tuple, &count, &terms)) return 0;
    if (count != 2) return 0;

    ERL_NIF_TERM lhs, lhs_tail;
    ERL_NIF_TERM list = terms[0];
    while (enif_get_list_cell(env, list, &lhs, &lhs_tail)) {
      int64 dim;
      if (!get(env, lhs, &dim)) return 0;
      dims->add_lhs_contracting_dimensions(dim);

      list = lhs_tail;
    }

    ERL_NIF_TERM rhs, rhs_tail;
    list = terms[1];
    while (enif_get_list_cell(env, list, &rhs, &rhs_tail)) {
      int64 dim;
      if (!get(env, rhs, &dim)) return 0;
      dims->add_rhs_contracting_dimensions(dim);

      list = rhs_tail;
    }

    return 1;
  }

  ERL_NIF_TERM make_shape_info(ErlNifEnv* env, xla::Shape shape) {
    if (shape.IsTuple()) {
      int element_count = xla::ShapeUtil::TupleElementCount(shape);
      std::vector<ERL_NIF_TERM> terms;
      terms.reserve(element_count);
      for (int i=0; i < element_count; i++) {
        xla::Shape shape_elem = xla::ShapeUtil::GetTupleElementShape(shape, i);
        ERL_NIF_TERM shape_term = exla::make<xla::Shape>(env, shape_elem);
        terms.emplace_back(shape_term);
      }
      return enif_make_list_from_array(env, &terms[0], element_count);
    }

    xla::PrimitiveType type = shape.element_type();
    absl::Span<const int64> dims = shape.dimensions();
    int64 rank = shape.rank();

    std::string name = xla::primitive_util::LowercasePrimitiveTypeName(type);

    std::vector<ERL_NIF_TERM> dim_arr;
    dim_arr.reserve(rank);
    for (int i=0; i < rank; i++) {
      int copy;
      copy = dims.at(i);
      dim_arr.emplace_back(exla::make(env, copy));
    }

    ERL_NIF_TERM dims_term = enif_make_tuple_from_array(env, &dim_arr[0], rank);
    ERL_NIF_TERM type_term = exla::make(env, name);

    return enif_make_tuple(env, 2, dims_term, type_term);
  }

  ERL_NIF_TERM make(ErlNifEnv* env, ErlNifBinary var) {
    return enif_make_binary(env, &var);
  }

  ERL_NIF_TERM make(ErlNifEnv* env, int var) {
    return enif_make_int(env, var);
  }

  ERL_NIF_TERM make(ErlNifEnv* env, std::string var) {
    return enif_make_string(env, var.c_str(), ERL_NIF_LATIN1);
  }

  ERL_NIF_TERM make(ErlNifEnv* env, const char* string) {
    return enif_make_string(env, string, ERL_NIF_LATIN1);
  }

}  // namespace exla
