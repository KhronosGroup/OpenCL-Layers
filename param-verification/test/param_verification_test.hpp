#pragma once

#include "layers_test.hpp"

namespace param_verification_test {
  struct TestOptions {
    bool parseArg(int argc, char* argv[], int& i);
  };

  extern layers_test::TestContext<TestOptions> TEST_CONTEXT;

  void setup(int argc,
             char* argv[],
             cl_version required_version,
             cl_platform_id& platform,
             cl_device_id& device);

  int finalize();

  #define EXPECT_SUCCESS(status) LAYERS_TEST_EXPECT_SUCCESS(::param_verification_test::TEST_CONTEXT, status)
  #define EXPECT_ERROR(status, expected) LAYERS_TEST_EXPECT_ERROR(::param_verification_test::TEST_CONTEXT, status, expected)
}
