#include "object_lifetime_test.hpp"
#include <memory>
#include <sstream>
#include <cstdlib>

namespace layer_test {
  TestContext TEST_CONTEXT;

  bool parseArgs(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
      const char* arg = argv[i];
      if (std::strcmp(arg, "--ref-count-includes-implicit") == 0) {
        TEST_CONTEXT.ref_count_includes_implicit = true;
      } else if (std::strcmp(arg, "--use-released-objects") == 0) {
        TEST_CONTEXT.use_released_objects = true;
      } else if (std::strcmp(arg, "--use-inaccessible-objects") == 0) {
        TEST_CONTEXT.use_inaccessible_objects = true;
      } else if (std::strcmp(arg, "--platform") == 0) {
        if (++i == argc) {
          LAYER_TEST_LOG() << "--platform expects argument <platform>" << std::endl;
          return false;
        }
        TEST_CONTEXT.platform = argv[i];
      } else {
        LAYER_TEST_LOG() << "invalid argument " << arg << std::endl;
        return false;
      }
    }

    return true;
  }

  void setup(int argc,
             char* argv[],
             cl_version required_version,
             cl_platform_id& platform,
             cl_device_id& device) {
    if (!parseArgs(argc, argv)) {
      exit(EXIT_FAILURE);
    }

    cl_uint num_platforms = 0;
    EXPECT_SUCCESS(clGetPlatformIDs(0, nullptr, &num_platforms));

    if (num_platforms == 0) {
      LAYER_TEST_LOG() << "No OpenCL platform detected" << std::endl;
      exit(-1);
    }

    auto platforms = std::make_unique<cl_platform_id[]>(num_platforms);
    EXPECT_SUCCESS(clGetPlatformIDs(num_platforms, platforms.get(), nullptr));

    // Select the right icd.
    bool test_platform_found = false;
    for (cl_uint i = 0; i < num_platforms; ++i) {
      size_t name_size;
      EXPECT_SUCCESS(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, nullptr, &name_size));
      auto name = std::make_unique<char[]>(name_size);
      EXPECT_SUCCESS(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, name_size, name.get(), nullptr));

      if (std::strcmp(name.get(), TEST_CONTEXT.platform) == 0) {
        platform = platforms[i];
        test_platform_found = true;
      }
    }
    if (!test_platform_found) {
      LAYER_TEST_LOG() << "Platform '" << TEST_CONTEXT.platform << "' not found" << std::endl;
      exit(EXIT_FAILURE);
    }

    // Get a device
    cl_uint numDevices;
    EXPECT_SUCCESS(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, &numDevices));
    if(numDevices == 0) {
      LAYER_TEST_LOG() << "No OpenCL device found" << std::endl;
      exit(EXIT_FAILURE);
    }

    size_t version_length;
    EXPECT_SUCCESS(clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, nullptr, &version_length));
    auto version_data = std::make_unique<char[]>(version_length);
    EXPECT_SUCCESS(clGetPlatformInfo(platform, CL_PLATFORM_VERSION, version_length, version_data.get(), nullptr));

    std::stringstream version;
    version.write(version_data.get(), version_length);

    std::string opencl;
    cl_uint major, minor;
    version >> opencl;
    version >> major;
    version.get();
    version >> minor;

    if (version.fail()) {
      LAYER_TEST_LOG() << "Failed to parse platform OpenCL version" << std::endl;
    }

    TEST_CONTEXT.version = CL_MAKE_VERSION(major, minor, 0);
    LAYER_TEST_LOG() << "platform opencl version is " << major << "." << minor << std::endl;

    if (TEST_CONTEXT.version < required_version) {
      LAYER_TEST_LOG() << "test requires at least version " << CL_VERSION_MAJOR(required_version)
                       << "." << CL_VERSION_MINOR(required_version) << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  int finalize() {
    if (TEST_CONTEXT.failed_checks != 0) {
      LAYER_TEST_LOG() << "failed " << TEST_CONTEXT.failed_checks << " checks" << std::endl;
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
  }
}
