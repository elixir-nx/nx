#ifndef EXLA_CLIENT_H_
#define EXLA_CLIENT_H_

#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/service/platform_util.h"

namespace exla {
  xla::StatusOr<xla::LocalClient*> GetCpuClient();

  xla::StatusOr<xla::LocalClient*> GetGpuClient();
}

#endif
