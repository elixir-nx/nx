#include "tensorflow/compiler/xla/exla/exla_nif_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"

namespace exla {
namespace nif {

  // Status helpers

  ERL_NIF_TERM error(ErlNifEnv* env, const char* msg) {
    ERL_NIF_TERM atom = enif_make_atom(env, "error");
    ERL_NIF_TERM msg_term = enif_make_string(env, msg, ERL_NIF_LATIN1);
    return enif_make_tuple2(env, atom, msg_term);
  }

  ERL_NIF_TERM ok(ErlNifEnv* env, ERL_NIF_TERM term) {
    return enif_make_tuple2(env, ok(env), term);
  }

  ERL_NIF_TERM ok(ErlNifEnv* env) {
    return enif_make_atom(env, "ok");
  }

  // Numeric types

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
                          reinterpret_cast<nif_int64_t *>(var));
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
                         reinterpret_cast<unsigned int *>(var));
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, uint64* var) {
    return enif_get_uint64(env, term,
                           reinterpret_cast<nif_uint64_t *>(var));
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, float16* var) {
    double value;
    if (!enif_get_double(env, term, &value)) return 0;
    *var = static_cast<float16>(value);
    return 1;
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

  int get(ErlNifEnv* env, ERL_NIF_TERM term, complex64* var) {
    return 0;
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, complex128* var) {
    return 0;
  }

  ERL_NIF_TERM make(ErlNifEnv* env, int32 var) {
    return enif_make_int(env, var);
  }

  // Standard types

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

  ERL_NIF_TERM make(ErlNifEnv* env, ErlNifBinary var) {
    return enif_make_binary(env, &var);
  }

  ERL_NIF_TERM make(ErlNifEnv* env, std::string var) {
    return enif_make_string(env, var.c_str(), ERL_NIF_LATIN1);
  }

  ERL_NIF_TERM make(ErlNifEnv* env, const char* string) {
    return enif_make_string(env, string, ERL_NIF_LATIN1);
  }

  // Atoms

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

  ERL_NIF_TERM atom(ErlNifEnv* env, const char* msg) {
    return enif_make_atom(env, msg);
  }

  // Containers

  int get_tuple(ErlNifEnv* env, ERL_NIF_TERM tuple, std::vector<int64> &var) {
    const ERL_NIF_TERM *terms;
    int length;
    if (!enif_get_tuple(env, tuple, &length, &terms)) return 0;
    var.reserve(length);

    for (int i = 0; i < length; i++) {
      int data;
      if (!get(env, terms[i], &data)) return 0;
      var.push_back(data);
    }
    return 1;
  }

  int get_list(ErlNifEnv* env,
               ERL_NIF_TERM list,
               std::vector<ErlNifBinary> &var) {
    unsigned int length;
    if (!enif_get_list_length(env, list, &length)) return 0;
    var.reserve(length);
    ERL_NIF_TERM head, tail;

    while (enif_get_list_cell(env, list, &head, &tail)) {
      ErlNifBinary elem;
      if (!get_binary(env, head, &elem)) return 0;
      var.push_back(elem);
      list = tail;
    }
    return 1;
  }

  int get_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<int64> &var) {
    unsigned int length;
    if (!enif_get_list_length(env, list, &length)) return 0;
    var.reserve(length);
    ERL_NIF_TERM head, tail;

    while (enif_get_list_cell(env, list, &head, &tail)) {
      int64 elem;
      if (!get(env, head, &elem)) return 0;
      var.push_back(elem);
      list = tail;
    }
    return 1;
  }

  int get_binary(ErlNifEnv* env, ERL_NIF_TERM term, ErlNifBinary* var) {
    return enif_inspect_binary(env, term, var);
  }

  ERL_NIF_TERM make_map(ErlNifEnv* env, std::map<std::string, int>& map) {
    ERL_NIF_TERM term = enif_make_new_map(env);
    std::map<std::string, int>::iterator itr;
    for (itr = map.begin(); itr != map.end(); ++itr) {
      ERL_NIF_TERM key = make(env, itr->first);
      ERL_NIF_TERM value = make(env, itr->second);
      enif_make_map_put(env, term, key, value, &term);
    }
    return term;
  }

  // Protobuf types

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
    int count;
    if (!enif_get_tuple(env, tuple, &count, &terms)) return 0;
    if (count != 4) return 0;

    ERL_NIF_TERM lhs_contract, lhs_contract_tail;
    ERL_NIF_TERM list = terms[0];
    while (enif_get_list_cell(env, list, &lhs_contract, &lhs_contract_tail)) {
      int64 dim;
      if (!get(env, lhs_contract, &dim)) return 0;
      dims->add_lhs_contracting_dimensions(dim);

      list = lhs_contract_tail;
    }

    ERL_NIF_TERM lhs_batch, lhs_batch_tail;
    list = terms[1];
    while (enif_get_list_cell(env, list, &lhs_batch, &lhs_batch_tail)) {
      int64 dim;
      if (!get(env, lhs_batch, &dim)) return 0;
      dims->add_lhs_batch_dimensions(dim);

      list = lhs_batch_tail;
    }

    ERL_NIF_TERM rhs_contract, rhs_contract_tail;
    list = terms[2];
    while (enif_get_list_cell(env, list, &rhs_contract, &rhs_contract_tail)) {
      int64 dim;
      if (!get(env, rhs_contract, &dim)) return 0;
      dims->add_rhs_contracting_dimensions(dim);

      list = rhs_contract_tail;
    }

    ERL_NIF_TERM rhs_batch, rhs_batch_tail;
    list = terms[3];
    while (enif_get_list_cell(env, list, &rhs_batch, &rhs_batch_tail)) {
      int64 dim;
      if (!get(env, rhs_batch, &dim)) return 0;
      dims->add_rhs_batch_dimensions(dim);

      list = rhs_batch_tail;
    }

    return 1;
  }

  int get_precision_config(ErlNifEnv* env,
                           ERL_NIF_TERM config_term,
                           int num_operands,
                           xla::PrecisionConfig* config) {
    int config_int;
    if (!get(env, config_term, &config_int)) return 0;

    switch (config_int) {
      case 0:
        for (int i = 0;i < num_operands;i++) {
          config->add_operand_precision(xla::PrecisionConfig::DEFAULT);
        }
        break;
      case 1:
        for (int i = 0;i < num_operands;i++) {
          config->add_operand_precision(xla::PrecisionConfig::HIGH);
        }
        break;
      case 2:
        for (int i = 0;i < num_operands;i++) {
          config->add_operand_precision(xla::PrecisionConfig::HIGHEST);
        }
        break;
      default:
        return 0;
    }

    return 1;
  }

  int get_conv_dimension_numbers(ErlNifEnv* env,
                                 ERL_NIF_TERM tuple,
                                 xla::ConvolutionDimensionNumbers* dimension_numbers) {
    const ERL_NIF_TERM* terms;
    int count;

    if (!enif_get_tuple(env, tuple, &count, &terms)) return 0;
    if (count != 3) return 0;

    const ERL_NIF_TERM* input_dims;
    int input_count;
    if (!enif_get_tuple(env, terms[0], &input_count, &input_dims)) return 0;
    if (count < 3) return 0;

    int64 input_batch_dimension;
    int64 input_feature_dimension;
    if (!get(env, input_dims[0], &input_batch_dimension)) return 0;
    if (!get(env, input_dims[1], &input_feature_dimension)) return 0;

    dimension_numbers->set_input_batch_dimension(input_batch_dimension);
    dimension_numbers->set_input_feature_dimension(input_feature_dimension);
    for (int i = 2; i < input_count; i++) {
      int64 value;
      if (!get(env, input_dims[i], &value)) return 0;
      dimension_numbers->add_input_spatial_dimensions(value);
    }

    const ERL_NIF_TERM* kernel_dims;
    int kernel_count;
    if (!enif_get_tuple(env, terms[1], &kernel_count, &kernel_dims)) return 0;
    if (kernel_count < 3) return 0;

    int64 kernel_input_feature_dimension;
    int64 kernel_output_feature_dimension;
    if (!get(env, kernel_dims[0], &kernel_input_feature_dimension)) return 0;
    if (!get(env, kernel_dims[1], &kernel_output_feature_dimension)) return 0;

    dimension_numbers->set_kernel_output_feature_dimension(kernel_output_feature_dimension);
    dimension_numbers->set_kernel_input_feature_dimension(kernel_input_feature_dimension);
    for (int i = 2; i < kernel_count; i++) {
      int64 value;
      if (!get(env, kernel_dims[i], &value)) return 0;
      dimension_numbers->add_kernel_spatial_dimensions(value);
    }

    const ERL_NIF_TERM* output_dims;
    int output_count;
    if (!enif_get_tuple(env, terms[2], &output_count, &output_dims)) return 0;
    if (output_count < 3) return 0;

    int64 output_batch_dimension;
    int64 output_feature_dimension;
    if (!get(env, output_dims[0], &output_batch_dimension)) return 0;
    if (!get(env, output_dims[1], &output_feature_dimension)) return 0;

    dimension_numbers->set_output_batch_dimension(output_batch_dimension);
    dimension_numbers->set_output_feature_dimension(output_feature_dimension);
    for (int i = 2; i < output_count; i++) {
      int64 value;
      if (!get(env, output_dims[i], &value)) return 0;
      dimension_numbers->add_output_spatial_dimensions(value);
    }

    return 1;
  }

  int get_general_padding(ErlNifEnv* env,
                          ERL_NIF_TERM padding_term,
                          std::vector<std::pair<int64, int64>>& padding) {
    unsigned int length;
    if (!enif_get_list_length(env, padding_term, &length)) return 0;

    padding.reserve(length);
    ERL_NIF_TERM head, tail;

    while (enif_get_list_cell(env, padding_term, &head, &tail)) {
      const ERL_NIF_TERM* terms;
      int count;

      if (!enif_get_tuple(env, head, &count, &terms)) return 0;
      if (count != 2) return 0;

      int64 lo, hi;
      if (!get(env, terms[0], &lo)) return 0;
      if (!get(env, terms[1], &hi)) return 0;

      padding.push_back(std::pair<int64, int64>(lo, hi));

      padding_term = tail;
    }

    return 1;
  }

  int get_primitive_type(ErlNifEnv* env, ERL_NIF_TERM term, xla::PrimitiveType* type) {
    std::string type_str;
    if (!get(env, term, type_str)) return 0;

    xla::StatusOr<xla::PrimitiveType> type_status =
      xla::primitive_util::StringToPrimitiveType(type_str);

    if (!type_status.ok()) {
      return 0;
    }
    *type = type_status.ConsumeValueOrDie();
    return 1;
  }

  ERL_NIF_TERM make_shape_info(ErlNifEnv* env, xla::Shape shape) {
    if (shape.IsTuple()) {
      int element_count = xla::ShapeUtil::TupleElementCount(shape);
      std::vector<ERL_NIF_TERM> terms;
      terms.reserve(element_count);
      for (int i = 0; i < element_count; i++) {
        xla::Shape shape_elem = xla::ShapeUtil::GetTupleElementShape(shape, i);
        ERL_NIF_TERM shape_term = make<xla::Shape>(env, shape_elem);
        terms.push_back(shape_term);
      }
      return enif_make_list_from_array(env, &terms[0], element_count);
    }

    xla::PrimitiveType type = shape.element_type();
    absl::Span<const int64> dims = shape.dimensions();
    int64 rank = shape.rank();

    std::string name = xla::primitive_util::LowercasePrimitiveTypeName(type);

    std::vector<ERL_NIF_TERM> dim_arr;
    dim_arr.reserve(rank);
    for (int i = 0; i < rank; i++) {
      int copy;
      copy = dims.at(i);
      dim_arr.push_back(make(env, copy));
    }

    ERL_NIF_TERM dims_term = enif_make_tuple_from_array(env, &dim_arr[0], rank);
    ERL_NIF_TERM type_term = make(env, name);

    return enif_make_tuple(env, 2, dims_term, type_term);
  }

}  // namespace nif
}  // namespace exla
