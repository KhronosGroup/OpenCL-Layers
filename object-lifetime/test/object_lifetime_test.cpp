#include "object_lifetime_test.hpp"

namespace object_lifetime_test {
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
    const char* arg = argv[i];
    if (std::strcmp(arg, "--ref-count-includes-implicit") == 0) {
      this->ref_count_includes_implicit = true;
      return true;
    } else if (std::strcmp(arg, "--use-released-objects") == 0) {
      this->use_released_objects = true;
      return true;
    } else if (std::strcmp(arg, "--use-inaccessible-objects") == 0) {
      this->use_inaccessible_objects = true;
      return true;
    }
    return false;
  }
}
