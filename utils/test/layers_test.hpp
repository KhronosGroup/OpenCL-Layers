#pragma once

#include <CL/cl.h>
#include <iostream>
#include <memory>
#include <cstring>
#include <cstdlib>

#include "utils.hpp"

namespace layers_test {
  template <typename TestOptions>
  struct TestContext {
      // The version to base testing off.
    cl_version version = CL_MAKE_VERSION(1, 1, 0);
    // The OpenCL platform to use.
    const char* platform = "Object Lifetime Layer Test ICD";
    // Keeps track number of failed assertions during this test.
    size_t failed_checks = 0;

    TestOptions options;

    void fail() {
      ++this->failed_checks;
    }

    static TestContext<TestOptions> setup(int argc,
                                          char* argv[],
                                          cl_version required_version,
                                          cl_platform_id& platform,
                                          cl_device_id& device);

    bool parseArgs(int argc, char* argv[]);

    int finalize() const;
  };

  inline std::ostream& log(const char* file, int line) {
    return std::cout << file << ":" << line << ": ";
  }

  #define LAYERS_TEST_LOG() ::layers_test::log(__FILE__, __LINE__)

  template <typename T>
  inline void expectSuccess(TestContext<T>& context, const char* file, int line, cl_int status) {
    if (status != CL_SUCCESS) {
      log(file, line) << "expected success, got " << status << std::endl;
      context.fail();
    }
  }

  #define LAYERS_TEST_EXPECT_SUCCESS(ctx, status) ::layers_test::expectSuccess(ctx, __FILE__, __LINE__, status)

  template <typename T>
  inline void expectError(TestContext<T>& context, const char* file, int line, cl_int status, cl_int expected) {
    if (status == CL_SUCCESS) {
      log(file, line) << "expected error " << expected << ", got success" << std::endl;
      context.fail();
    } else if (status != expected) {
      log(file, line) << "expected error " << expected << ", got " << status << std::endl;
      context.fail();
    }
  }

  #define LAYERS_TEST_EXPECT_ERROR(ctx, status, expected) ::layers_test::expectError(ctx, __FILE__, __LINE__, status, expected)

  template <typename TestOptions>
  TestContext<TestOptions> TestContext<TestOptions>::setup(int argc,
                                                           char* argv[],
                                                           cl_version required_version,
                                                           cl_platform_id& platform,
                                                           cl_device_id& device)
  {
    TestContext<TestOptions> context;
    if (!context.parseArgs(argc, argv)) {
      exit(EXIT_FAILURE);
    }

    cl_uint num_platforms = 0;
    LAYERS_TEST_EXPECT_SUCCESS(context, clGetPlatformIDs(0, nullptr, &num_platforms));

    if (num_platforms == 0) {
      LAYERS_TEST_LOG() << "No OpenCL platform detected" << std::endl;
      exit(-1);
    }

    auto platforms = std::make_unique<cl_platform_id[]>(num_platforms);
    LAYERS_TEST_EXPECT_SUCCESS(context, clGetPlatformIDs(num_platforms, platforms.get(), nullptr));

    // Select the right icd.
    bool test_platform_found = false;
    for (cl_uint i = 0; i < num_platforms; ++i) {
      size_t name_size;
      LAYERS_TEST_EXPECT_SUCCESS(context, clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, nullptr, &name_size));
      auto name = std::make_unique<char[]>(name_size);
      LAYERS_TEST_EXPECT_SUCCESS(context, clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, name_size, name.get(), nullptr));

      if (std::strcmp(name.get(), context.platform) == 0) {
        platform = platforms[i];
        test_platform_found = true;
      }
    }
    if (!test_platform_found) {
      LAYERS_TEST_LOG() << "Platform '" << context.platform << "' not found" << std::endl;
      exit(EXIT_FAILURE);
    }

    // Get a device
    cl_uint numDevices;
    LAYERS_TEST_EXPECT_SUCCESS(context, clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, &numDevices));
    if(numDevices == 0) {
      LAYERS_TEST_LOG() << "No OpenCL device found" << std::endl;
      exit(EXIT_FAILURE);
    }

    size_t version_length;
    LAYERS_TEST_EXPECT_SUCCESS(context, clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, nullptr, &version_length));
    auto version_data = std::make_unique<char[]>(version_length);
    LAYERS_TEST_EXPECT_SUCCESS(context, clGetPlatformInfo(platform, CL_PLATFORM_VERSION, version_length, version_data.get(), nullptr));

    if (!ocl_layer_utils::parse_cl_version_string(version_data.get(), &context.version)) {
      LAYERS_TEST_LOG() << "Failed to parse platform OpenCL version" << std::endl;
      exit(EXIT_FAILURE);
    }

    LAYERS_TEST_LOG() << "platform opencl version is "
                      << CL_VERSION_MAJOR(context.version) << "." << CL_VERSION_MINOR(context.version)
                      << std::endl;

    if (context.version < required_version) {
      LAYERS_TEST_LOG() << "test requires at least version " << CL_VERSION_MAJOR(required_version)
                        << "." << CL_VERSION_MINOR(required_version) << std::endl;
      exit(EXIT_FAILURE);
    }

    return context;
  }

  template <typename TestOptions>
  bool TestContext<TestOptions>::parseArgs(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
      const char* arg = argv[i];
      if (std::strcmp(arg, "--platform") == 0) {
        if (++i == argc) {
          LAYERS_TEST_LOG() << "--platform expects argument <platform>" << std::endl;
          return false;
        }
        this->platform = argv[i];
      } else if (!this->options.parseArg(argc, argv, i)) {
        LAYERS_TEST_LOG() << "invalid argument " << arg << std::endl;
        return false;
      }
    }

    return true;
  }

  template <typename TestOptions>
  int TestContext<TestOptions>::finalize() const {
    if (this->failed_checks != 0) {
      LAYERS_TEST_LOG() << "failed " << this->failed_checks << " checks" << std::endl;
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
  }
}
