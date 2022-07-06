#include "param_verification_test.hpp"

// Sanity check, and also to test whether the setup does not produce any errors.

int main(int argc, char* argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  param_verification_test::setup(argc, argv, CL_MAKE_VERSION(1, 1, 0), platform, device);

  return param_verification_test::finalize();
}
