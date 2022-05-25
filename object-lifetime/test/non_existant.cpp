#include "object_lifetime_test.hpp"

#include <limits>

int main(int argc, char *argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  layer_test::setup(argc, argv, CL_MAKE_VERSION(1, 1, 0), platform, device);

  cl_context context = layer_test::createContext(platform, device);

  cl_mem non_existant_buffer = reinterpret_cast<cl_mem>(std::numeric_limits<size_t>::max());
  EXPECT_DESTROYED(non_existant_buffer); // used but does not exist

  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_DESTROYED(context); // recently deleted with type: CONTEXT

  return layer_test::finalize();
}
