#include "param_verification_test.hpp"

namespace param_verification_test {
  layers_test::TestContext<TestOptions> TEST_CONTEXT;

  int finalize() {
    return TEST_CONTEXT.finalize();
  }

  void setup(int argc,
             char* argv[],
             cl_version required_version,
             cl_platform_id& platform,
             cl_device_id& device)
  {
    TEST_CONTEXT = layers_test::TestContext<TestOptions>::setup(argc,
                                                                argv,
                                                                required_version,
                                                                platform,
                                                                device);
  }

  bool TestOptions::parseArg(int argc, char* argv[], int& i) {
    (void) argc;
    (void) argv;
    (void) i;
    return false;
  }
}
