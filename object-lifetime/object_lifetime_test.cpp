#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <iostream>
#include <string>
#include <memory>
#include <array>
#include <tuple>
#include <algorithm>
#include <sstream>
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <cstring>

// Returns:
// -1 if lhs <  rhs
//  0 if lhs == rhs
//  1 if lhs >  rhs
int compareVersions(cl_version lhs, cl_version rhs) {
  if (CL_VERSION_MAJOR(lhs) < CL_VERSION_MAJOR(rhs)) {
    return -1;
  } else if (CL_VERSION_MAJOR(lhs) > CL_VERSION_MAJOR(rhs)) {
    return 1;
  }

  if (CL_VERSION_MINOR(lhs) < CL_VERSION_MINOR(rhs)) {
    return -1;
  } else if (CL_VERSION_MINOR(lhs) > CL_VERSION_MINOR(rhs)) {
    return 1;
  }

  if (CL_VERSION_PATCH(lhs) < CL_VERSION_PATCH(rhs)) {
    return -1;
  } else if (CL_VERSION_PATCH(lhs) > CL_VERSION_PATCH(rhs)) {
    return 1;
  }

  return 0;
}

struct TestContext {
  // The version to base testing off.
  cl_version version = CL_MAKE_VERSION(1, 1, 0);
  // Whether CL_*_REFERENCE_COUNT reports implicit or explicit reference count.
  bool ref_count_includes_implicit = false;
  // Whether released objects can still be used.
  bool use_released_objects = false;
  // Whether inaccessible objects that are still implicitly referenced can be used.
  bool use_inaccessible_objects = false;
  // The OpenCL platform to use.
  const char* platform = "Object Lifetime Layer Test ICD";
  // Keeps track number of failed assertions during this test.
  size_t failed_checks = 0;

  void fail() {
    ++this->failed_checks;
  }

  bool isInVersionRange(cl_version minimum, cl_version maximum) const {
    return compareVersions(this->version, minimum) >= 0 && compareVersions(this->version, maximum) <= 0;
  }

  int compareVersion(cl_version version) const {
    return compareVersions(this->version, version);
  }
};

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
        std::cerr << "ERROR: --platform expects argument <platform>" << std::endl;
        return false;
      }
      TEST_CONTEXT.platform = argv[i];
    } else {
      std::cerr << "ERROR: invalid argument " << arg << std::endl;
      return false;
    }
  }

  return true;
}

namespace ocl_utils {
  cl_int getRefCount(cl_context handle, cl_uint& ref_count) {
    return clGetContextInfo(handle, CL_CONTEXT_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  cl_int getRefCount(cl_device_id handle, cl_uint &ref_count) {
    return clGetDeviceInfo(handle, CL_DEVICE_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  cl_int getRefCount(cl_command_queue handle, cl_uint &ref_count) {
    return clGetCommandQueueInfo(handle, CL_QUEUE_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  cl_int getRefCount(cl_mem handle, cl_uint &ref_count) {
    return clGetMemObjectInfo(handle, CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  cl_int getRefCount(cl_sampler handle, cl_uint &ref_count) {
    return clGetSamplerInfo(handle, CL_SAMPLER_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  cl_int getRefCount(cl_program handle, cl_uint &ref_count) {
    return clGetProgramInfo(handle, CL_PROGRAM_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  cl_int getRefCount(cl_kernel handle, cl_uint &ref_count) {
    return clGetKernelInfo(handle, CL_KERNEL_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  cl_int getRefCount(cl_event handle, cl_uint &ref_count) {
    return clGetEventInfo(handle, CL_EVENT_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  template <typename T>
  cl_int CL_INVALID;

  template <>
  constexpr const cl_int CL_INVALID<cl_context> = CL_INVALID_CONTEXT;

  template <>
  constexpr const cl_int CL_INVALID<cl_device_id> = CL_INVALID_DEVICE;

  template <>
  constexpr const cl_int CL_INVALID<cl_command_queue> = CL_INVALID_COMMAND_QUEUE;

  template <>
  constexpr const cl_int CL_INVALID<cl_mem> = CL_INVALID_MEM_OBJECT;

  template <>
  constexpr const cl_int CL_INVALID<cl_sampler> = CL_INVALID_SAMPLER;

  template <>
  constexpr const cl_int CL_INVALID<cl_program> = CL_INVALID_PROGRAM;

  template <>
  constexpr const cl_int CL_INVALID<cl_kernel> = CL_INVALID_KERNEL;

  template <>
  constexpr const cl_int CL_INVALID<cl_event> = CL_INVALID_EVENT;
}

std::ostream& reportError(const char* file, int line) {
  return std::cout << file << ":" << line << ": ";
}

#define REPORT_ERROR() reportError(__FILE__, __LINE__)

void expectSuccess(const char* file, int line, cl_int status) {
  if (status != CL_SUCCESS) {
    reportError(file, line) << "expected success, got " << status << std::endl;
    TEST_CONTEXT.fail();
  }
}

#define EXPECT_SUCCESS(status) ::expectSuccess(__FILE__, __LINE__, status)

void expectError(const char* file, int line, cl_int status, cl_int expected) {
  if (status == CL_SUCCESS) {
    reportError(file, line) << "expected error " << expected << ", got success" << std::endl;
    TEST_CONTEXT.fail();
  } else if (status != expected) {
    reportError(file, line) << "expected error " << expected << ", got " << status << std::endl;
    TEST_CONTEXT.fail();
  }
}

#define EXPECT_ERROR(status, expected) ::expectError(__FILE__, __LINE__, status, expected)

template <typename Handle>
void expectRefCount(const char* file,
                    int line,
                    Handle handle,
                    cl_uint expected_explicit_count,
                    cl_uint expected_implicit_count)
{
  cl_uint actual_ref_count;
  cl_int status = ocl_utils::getRefCount(handle, actual_ref_count);

  auto expect_not_destroyed = [&](cl_uint expected_ref_count) {
    if (status == ocl_utils::CL_INVALID<Handle>) {
      reportError(file, line) << "expected that object was not destroyed, but it was" << std::endl;
      TEST_CONTEXT.fail();
    } else if (status != CL_SUCCESS) {
      reportError(file, line) << "expected that object was not destroyed, got unexpected error " << status << std::endl;
      TEST_CONTEXT.fail();
    } else if (actual_ref_count != expected_ref_count) {
      reportError(file, line) << "expected ref count " << expected_ref_count << ", got " << actual_ref_count << std::endl;
      TEST_CONTEXT.fail();
    }
  };

  auto expect_destroyed = [&] {
    if (status == CL_SUCCESS) {
      reportError(file, line) << "expected that object was destroyed, but it is not" << std::endl;
      TEST_CONTEXT.fail();
    } else if (status != ocl_utils::CL_INVALID<Handle>) {
      reportError(file, line) << "expected that object was destroyed, got unexpect error " << status << std::endl;
      TEST_CONTEXT.fail();
    }
  };

  if (expected_explicit_count == 0) {
    // If the expected explicit ref count is zero:
    // In OpenCL 1, the object is deallocated and so we should get CL_INVALID.
    // - If use_released_objects is true, we expect a refcount of zero and CL_SUCCESS.
    // - use_inaccessible_objects does not apply.
    // In OpenCL 2:
    // - If the object still has implicit references, we expect CL_SUCCESS.
    //   - use_released_objects and use_inaccessible_objects do not apply.
    // - Otherwise:
    //   - if use_release_objects is true, we expect CL_SUCCESS and zero references.
    //   - Else, CL_INVALID.
    // - use_inaccessible_objects does not apply.
    // In OpenCL 3, the object may have implicit references, but it is inaccessible and so we should get CL_INVALID.
    // - If the implicit reference count is zero and use_release_objects is true, we expect CL_SUCCESS and zero references.
    // - Otherwise if implicit reference count is nonzero and use_inaccessible_objects is true, we expect CL_SUCCESS.
    // - Otherwise we expect CL_INVALID.

    if (TEST_CONTEXT.isInVersionRange(CL_MAKE_VERSION(1, 1, 0), CL_MAKE_VERSION(1, 2, 0))) {
      if (TEST_CONTEXT.use_released_objects) {
        expect_not_destroyed(0);
      } else {
        expect_destroyed();
      }
    } else if (TEST_CONTEXT.compareVersion(CL_MAKE_VERSION(2, 0, 0)) == 0) {
      if (expected_implicit_count != 0) {
        expect_not_destroyed(expected_implicit_count);
      } else if (TEST_CONTEXT.use_released_objects) {
        expect_not_destroyed(0);
      } else {
        expect_destroyed();
      }
    } else if (TEST_CONTEXT.compareVersion(CL_MAKE_VERSION(2, 1, 0)) >= 0) {
      if (expected_implicit_count == 0 && TEST_CONTEXT.use_released_objects) {
        expect_not_destroyed(0);
      } else if (expected_implicit_count != 0 && TEST_CONTEXT.use_inaccessible_objects) {
        expect_not_destroyed(expected_implicit_count);
      } else {
        expect_destroyed();
      }
    }
  } else {
    // If the expected explicit ref count is nonzero:
    // In OpenCL 1.1 and 1.2, we expect that the explicit ref count is the actual ref count.
    // In OpenCL 2.0 and above, the actual ref count includes the implicit count as well.
    if (TEST_CONTEXT.ref_count_includes_implicit) {
      expect_not_destroyed(expected_explicit_count + expected_implicit_count);
    } else {
      expect_not_destroyed(expected_explicit_count);
    }
  }
}

#define EXPECT_REF_COUNT(handle, explicit_count, implicit_count) \
  ::expectRefCount(__FILE__, __LINE__, handle, explicit_count, implicit_count)

#define EXPECT_DESTROYED(handle) EXPECT_REF_COUNT(handle, 0, 0)

cl_context createContext(cl_platform_id platform, cl_device_id device) {
  cl_int status;
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform), 0};
  cl_context context = clCreateContext(properties, 1, &device, nullptr, nullptr, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(context, 1, 0);

  return context;
}

void testBasicCounting(cl_platform_id platform, cl_device_id device) {
  cl_int status;
  cl_context context = createContext(platform, device);

  // Create some buffers to test with
  cl_mem buffer_a = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, nullptr, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(buffer_a, 1, 0);
  EXPECT_REF_COUNT(context, 1, 1);

  cl_mem buffer_b = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 1, nullptr, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(buffer_a, 1, 0);
  EXPECT_REF_COUNT(context, 1, 2);

  EXPECT_SUCCESS(clRetainContext(context));
  EXPECT_REF_COUNT(buffer_a, 1, 0);
  EXPECT_REF_COUNT(buffer_b, 1, 0);
  EXPECT_REF_COUNT(context, 2, 2);

  EXPECT_SUCCESS(clRetainMemObject(buffer_a));
  EXPECT_REF_COUNT(buffer_a, 2, 0);
  EXPECT_REF_COUNT(buffer_b, 1, 0);
  EXPECT_REF_COUNT(context, 2, 2);

  EXPECT_SUCCESS(clRetainMemObject(buffer_b));
  EXPECT_REF_COUNT(buffer_a, 2, 0);
  EXPECT_REF_COUNT(buffer_b, 2, 0);
  EXPECT_REF_COUNT(context, 2, 2);

  EXPECT_SUCCESS(clRetainContext(context));
  EXPECT_REF_COUNT(context, 3, 2);

  EXPECT_SUCCESS(clReleaseMemObject(buffer_b));
  EXPECT_REF_COUNT(buffer_a, 2, 0);
  EXPECT_REF_COUNT(buffer_b, 1, 0);
  EXPECT_REF_COUNT(context, 3, 2);

  EXPECT_SUCCESS(clReleaseMemObject(buffer_b));
  EXPECT_REF_COUNT(buffer_a, 2, 0);
  EXPECT_DESTROYED(buffer_b);
  EXPECT_REF_COUNT(context, 3, 1);

  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_REF_COUNT(context, 2, 1);
  EXPECT_REF_COUNT(buffer_a, 2, 0);

  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_REF_COUNT(context, 1, 1);
  EXPECT_REF_COUNT(buffer_a, 2, 0);

  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_REF_COUNT(context, 0, 1);
  EXPECT_REF_COUNT(buffer_a, 2, 0);

  EXPECT_SUCCESS(clReleaseMemObject(buffer_a));
  EXPECT_REF_COUNT(buffer_a, 1, 0);
  EXPECT_REF_COUNT(context, 0, 1);

  EXPECT_SUCCESS(clReleaseMemObject(buffer_a));
  EXPECT_DESTROYED(buffer_a);
  EXPECT_DESTROYED(context);
}

void testLifetimeEdgeCases(cl_platform_id platform, cl_device_id device) {
  cl_int status;
  cl_context context = createContext(platform, device);

  cl_image_format format = { CL_RGBA, CL_UNORM_INT8 };
  cl_mem image_2d = clCreateImage2D(context, CL_MEM_READ_ONLY, &format, 1, 1, 0, nullptr, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(image_2d, 1, 0);
  EXPECT_REF_COUNT(context, 1, 1);

  cl_mem image_3d = clCreateImage3D(context, CL_MEM_READ_ONLY, &format, 1, 1, 2, 0, 0, nullptr, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(image_3d, 1, 0);
  EXPECT_REF_COUNT(context, 1, 2);

  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_REF_COUNT(context, 0, 2);

  // Try to free the context, which is now only implicitly retained.
  EXPECT_ERROR(clReleaseContext(context), CL_INVALID_CONTEXT);
  // That should not influence the implicit ref count
  EXPECT_REF_COUNT(context, 0, 2);

  EXPECT_SUCCESS(clReleaseMemObject(image_2d));
  EXPECT_DESTROYED(image_2d);
  EXPECT_REF_COUNT(context, 0, 1);

  EXPECT_SUCCESS(clReleaseMemObject(image_3d));
  EXPECT_DESTROYED(image_3d);
  EXPECT_DESTROYED(context);

  // Double free should fail.
  EXPECT_ERROR(clReleaseMemObject(image_2d), CL_INVALID_MEM_OBJECT);
  EXPECT_ERROR(clReleaseContext(context), CL_INVALID_CONTEXT);

  // Create a new context, which might re-use a previous context allocation.
  cl_context new_context = createContext(platform, device);

  EXPECT_DESTROYED(context);

  EXPECT_SUCCESS(clRetainContext(new_context));
  EXPECT_REF_COUNT(new_context, 2, 0);
  EXPECT_DESTROYED(context);

  EXPECT_SUCCESS(clReleaseContext(new_context));
  EXPECT_REF_COUNT(new_context, 1, 0);
  EXPECT_ERROR(clReleaseContext(context), CL_INVALID_CONTEXT);
  EXPECT_REF_COUNT(new_context, 1, 0);
  EXPECT_DESTROYED(context);

  EXPECT_SUCCESS(clReleaseContext(new_context));
  EXPECT_DESTROYED(new_context);
}

void testSubBuffer(cl_platform_id platform, cl_device_id device) {
  cl_int status;
  cl_context context = createContext(platform, device);

  cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 64, nullptr, &status);
  EXPECT_SUCCESS(status);

  cl_buffer_region sub_region = {0, 32};
  cl_mem sub_buffer = clCreateSubBuffer(buffer,
                                        CL_MEM_READ_WRITE,
                                        CL_BUFFER_CREATE_TYPE_REGION,
                                        static_cast<const void*>(&sub_region),
                                        &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(buffer, 1, 1);
  EXPECT_REF_COUNT(sub_buffer, 1, 0);

  // Release buffer, the child should keep it alive
  EXPECT_SUCCESS(clReleaseMemObject(buffer));
  EXPECT_REF_COUNT(buffer, 0, 1);
  EXPECT_REF_COUNT(sub_buffer, 1, 0);

  cl_buffer_region sub_sub_region = {0, 16};
  cl_mem sub_sub_buffer = clCreateSubBuffer(buffer,
                                            CL_MEM_READ_ONLY,
                                            CL_BUFFER_CREATE_TYPE_REGION,
                                            static_cast<const void*>(&sub_sub_region),
                                            &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(sub_sub_buffer, 1, 0);
  EXPECT_REF_COUNT(sub_buffer, 1, 0);

  // Release sub_buffer, but sub_sub_buffer should keep both its parent alive.
  EXPECT_SUCCESS(clReleaseMemObject(sub_buffer));
  EXPECT_REF_COUNT(sub_buffer, 0, 1);
  EXPECT_REF_COUNT(buffer, 0, 1);

   // Release the entire chain by releasing sub_sub_buffer
  EXPECT_DESTROYED(sub_sub_buffer);
  EXPECT_DESTROYED(sub_buffer);
  EXPECT_DESTROYED(buffer);

  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_DESTROYED(context);
}

void testSubImage(cl_platform_id platform, cl_device_id device) {
  cl_int status;
  cl_context context = createContext(platform, device);

 cl_image_desc desc = {
    CL_MEM_OBJECT_IMAGE2D, // image_type
    2,                     // image_width
    2,                     // image_height
    1,                     // image_depth
    1,                     // image_array size
    0,                     // image_row_pitch
    0,                     // image_slice_pitch
    0,                     // num_mip_levels
    0,                     // num_samples
    nullptr                // mem_object
  };
  cl_image_format format = {CL_R, CL_UNORM_INT8};
  cl_mem image = clCreateImage(context,
                               CL_MEM_READ_ONLY,
                               &format,
                               &desc,
                               nullptr,
                               &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(image, 1, 0);

  cl_image_desc sub_desc = {
    CL_MEM_OBJECT_IMAGE2D, // image_type
    1,                     // image_width
    1,                     // image_height
    1,                     // image_depth
    1,                     // image_array size
    0,                     // image_row_pitch
    0,                     // image_slice_pitch
    0,                     // num_mip_levels
    0,                     // num_samples
    image                  // mem_object
  };
  cl_mem sub_image = clCreateImage(context,
                                   CL_MEM_READ_ONLY,
                                   &format,
                                   &desc,
                                   nullptr,
                                   &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(image, 1, 0);

  // Release image, the sub_image should keep the chain alive
  EXPECT_SUCCESS(clReleaseMemObject(image));
  EXPECT_REF_COUNT(sub_image, 1, 0);
  EXPECT_REF_COUNT(image, 0, 1);

  EXPECT_SUCCESS(clReleaseMemObject(sub_image));
  EXPECT_DESTROYED(sub_image);
  EXPECT_DESTROYED(image);

  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_DESTROYED(context);
}

void testSubDevice(cl_platform_id platform, cl_device_id device) {
  constexpr const cl_uint num_sub_devices = 5;
  constexpr const cl_uint min_num_sub_cus = 2;
  constexpr const cl_uint min_cus = num_sub_devices * min_num_sub_cus;

  {
    size_t size;
    EXPECT_SUCCESS(clGetDeviceInfo(device, CL_DEVICE_PARTITION_PROPERTIES, 0, nullptr, &size));
    size_t len = size / sizeof(cl_device_partition_property);
    auto properties = std::make_unique<cl_device_partition_property[]>(len);
    EXPECT_SUCCESS(clGetDeviceInfo(device, CL_DEVICE_PARTITION_PROPERTIES, size, properties.get(), nullptr));

    if (std::find(properties.get(), properties.get() + len, CL_DEVICE_PARTITION_EQUALLY) == properties.get() + len) {
      REPORT_ERROR() << "test device does not support CL_DEVICE_PARTITION_EQUALLY" << std::endl;
      TEST_CONTEXT.fail();
      return;
    }
  }

  cl_uint num_cus;
  EXPECT_SUCCESS(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num_cus, nullptr));
  if (num_cus < min_cus) {
    REPORT_ERROR() << "test device does not have enough compute units" << std::endl;
    TEST_CONTEXT.fail();
    return;
  }

  cl_uint num_sub_cus = num_cus / num_sub_devices;
  cl_device_partition_property props[] = {CL_DEVICE_PARTITION_EQUALLY, num_sub_cus, 0};
  std::array<cl_device_id, num_sub_devices> sub_devices;
  EXPECT_SUCCESS(clCreateSubDevices(device,
                                    props,
                                    static_cast<cl_uint>(sub_devices.size()),
                                    sub_devices.data(),
                                    nullptr));

  for (auto sub_device : sub_devices) {
    EXPECT_REF_COUNT(sub_device, 1, 0);
  }

  // Create one sub-sub-device for each sub-device
  std::array<cl_device_id, num_sub_devices> sub_sub_devices;
  for (size_t i = 0; i < num_sub_devices; ++i) {
    cl_device_partition_property sub_props[] = {CL_DEVICE_PARTITION_EQUALLY, num_sub_cus / num_sub_devices, 0};
    EXPECT_SUCCESS(clCreateSubDevices(sub_devices[i],
                                      sub_props,
                                      1,
                                      &sub_sub_devices[i],
                                      nullptr));
  }

  // Check ref counts of sub-sub-devices and Release sub-devices
  for (size_t i = 0; i < num_sub_devices; ++i) {
    EXPECT_REF_COUNT(sub_sub_devices[i], 1, 0);
    EXPECT_REF_COUNT(sub_devices[i], 1, 1);
    EXPECT_SUCCESS(clReleaseDevice(sub_devices[i]));
    EXPECT_REF_COUNT(sub_devices[i], 0, 1);
  }

  {
    // Root devices should always have a ref count of 1, which is unaffected by retains and releases
    EXPECT_REF_COUNT(device, 1, 0);

    EXPECT_SUCCESS(clReleaseDevice(device));
    EXPECT_REF_COUNT(device, 1, 0);

    EXPECT_SUCCESS(clRetainDevice(device));
    EXPECT_REF_COUNT(device, 1, 0);
  }

  {
    // Release the first and check if the rest is okay.
    EXPECT_SUCCESS(clReleaseDevice(sub_sub_devices[0]));
    EXPECT_DESTROYED(sub_sub_devices[0]);
    EXPECT_DESTROYED(sub_devices[0]);

    for (cl_uint i = 1; i < num_sub_devices; ++i) {
      EXPECT_REF_COUNT(sub_sub_devices[i], 1, 0);
      EXPECT_REF_COUNT(sub_devices[i], 1, 1);
    }
  }

  {
    // Check that device dependencies keep the device alive.
    cl_context context = createContext(platform, sub_sub_devices[1]);
    EXPECT_SUCCESS(clReleaseDevice(sub_sub_devices[1]));
    EXPECT_REF_COUNT(sub_sub_devices[1], 0, 1);
    EXPECT_REF_COUNT(sub_devices[1], 0, 1);

    EXPECT_SUCCESS(clReleaseContext(context));
    EXPECT_DESTROYED(context);
    EXPECT_DESTROYED(sub_sub_devices[1]);
    EXPECT_DESTROYED(sub_devices[1]);

    for (cl_uint i = 2; i < num_sub_devices; ++i) {
      EXPECT_REF_COUNT(sub_sub_devices[i], 1, 0);
      EXPECT_REF_COUNT(sub_devices[i], 1, 1);
    }
  }

  {
    constexpr const cl_uint first_sub_sub_device = 2;
    constexpr const cl_uint num_sub_sub_devices = 2;
    cl_int status;
    // Create a context with 2 devices, check that the context keeps them both alive.
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform), 0};
    cl_context context = clCreateContext(properties,
                                         num_sub_sub_devices,
                                         &sub_sub_devices[first_sub_sub_device],
                                         nullptr,
                                         nullptr,
                                         &status);
    EXPECT_SUCCESS(status);
    EXPECT_REF_COUNT(context, 1, 0);
    for (cl_uint i = 0; i < num_sub_sub_devices; ++i) {
      EXPECT_SUCCESS(clReleaseDevice(sub_sub_devices[first_sub_sub_device + i]));
      EXPECT_REF_COUNT(sub_sub_devices[first_sub_sub_device + i], 0, 1);
      EXPECT_REF_COUNT(sub_devices[first_sub_sub_device + i], 0, 1);
    }
    EXPECT_SUCCESS(clReleaseContext(context));

    for (cl_uint i = 0; i < num_sub_sub_devices; ++i) {
      EXPECT_DESTROYED(sub_sub_devices[first_sub_sub_device + i]);
      EXPECT_DESTROYED(sub_devices[first_sub_sub_device + i]);
    }
  }

  // Just release the other sub-sub-devices
  for (cl_uint i = 4; i < num_sub_devices; ++i) {
    EXPECT_SUCCESS(clReleaseDevice(sub_sub_devices[i]));
    EXPECT_DESTROYED(sub_sub_devices[i]);
    EXPECT_DESTROYED(sub_devices[i]);
  }
}

void testPipeline(cl_platform_id platform, cl_device_id device) {
  cl_int status;
  cl_context context = createContext(platform, device);

  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(context, 1, 1);
  EXPECT_REF_COUNT(queue, 1, 0);

  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_REF_COUNT(context, 0, 1);

  const char* source = "kernel void test_kernel(sampler_t sampler) {}";
  size_t length = strlen(source);
  cl_program program = clCreateProgramWithSource(context,
                                                 1,
                                                 &source,
                                                 &length,
                                                 &status);
  EXPECT_SUCCESS(status);
  EXPECT_SUCCESS(clRetainProgram(program));
  EXPECT_REF_COUNT(program, 2, 0);
  EXPECT_REF_COUNT(context, 0, 2);

  cl_sampler sampler = clCreateSampler(context,
                                       CL_TRUE,
                                       CL_ADDRESS_REPEAT,
                                       CL_FILTER_LINEAR,
                                       &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(sampler, 1, 0);
  EXPECT_REF_COUNT(context, 0, 3);

  EXPECT_SUCCESS(clBuildProgram(program,
                                1,
                                &device,
                                nullptr,
                                nullptr,
                                nullptr));

  cl_kernel kernel = clCreateKernel(program, "test_kernel", &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(kernel, 1, 0);
  EXPECT_REF_COUNT(program, 2, 1);

  cl_event top_of_pipe = clCreateUserEvent(context, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(top_of_pipe, 1, 0);

  EXPECT_SUCCESS(clRetainEvent(top_of_pipe));
  EXPECT_REF_COUNT(top_of_pipe, 2, 0);
  EXPECT_REF_COUNT(context, 0, 4);

  // Setting kernel arguments should not affect reference counts.
  EXPECT_SUCCESS(clRetainSampler(sampler));
  EXPECT_REF_COUNT(sampler, 2, 0);
  EXPECT_SUCCESS(clSetKernelArg(kernel, 0, sizeof(cl_sampler), static_cast<const void*>(&sampler)));
  EXPECT_REF_COUNT(sampler, 2, 0);
  EXPECT_SUCCESS(clReleaseSampler(sampler));
  EXPECT_REF_COUNT(sampler, 1, 0);

  size_t global_work_size = 2;
  size_t local_work_size = 1;
  cl_event bottom_of_pipe;
  EXPECT_SUCCESS(clEnqueueNDRangeKernel(queue,
                                        kernel,
                                        1,
                                        nullptr,
                                        &global_work_size,
                                        &local_work_size,
                                        1,
                                        &top_of_pipe,
                                        &bottom_of_pipe));
  EXPECT_REF_COUNT(bottom_of_pipe, 1, 0);
  EXPECT_REF_COUNT(top_of_pipe, 2, 0);
  EXPECT_SUCCESS(clSetUserEventStatus(top_of_pipe, CL_COMPLETE));
  EXPECT_SUCCESS(clWaitForEvents(1, &bottom_of_pipe));

  EXPECT_SUCCESS(clRetainKernel(kernel));
  EXPECT_REF_COUNT(kernel, 2, 0);
  EXPECT_SUCCESS(clReleaseKernel(kernel));
  EXPECT_SUCCESS(clReleaseKernel(kernel));
  EXPECT_DESTROYED(kernel);

  EXPECT_SUCCESS(clReleaseProgram(program));
  EXPECT_DESTROYED(program);
  EXPECT_SUCCESS(clReleaseSampler(sampler));
  EXPECT_DESTROYED(sampler);
  EXPECT_SUCCESS(clReleaseCommandQueue(queue));
  EXPECT_DESTROYED(queue);
  EXPECT_SUCCESS(clReleaseEvent(top_of_pipe));
  EXPECT_SUCCESS(clReleaseEvent(top_of_pipe));
  EXPECT_DESTROYED(top_of_pipe);
  EXPECT_SUCCESS(clReleaseEvent(bottom_of_pipe));
  EXPECT_DESTROYED(bottom_of_pipe);
  EXPECT_DESTROYED(context);
}

void testCL200Calls(cl_platform_id platform, cl_device_id device) {
  cl_int status;
  cl_context context = createContext(platform, device);

  cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT , 0};
  cl_command_queue queue_a = clCreateCommandQueueWithProperties(context,
                                                                device,
                                                                props,
                                                                &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(queue_a, 1, 0);

  // Creating extra queues with CL_QUEUE_ON_DEVICE_DEFAULT will return the same queue and increment its reference count.
  cl_command_queue queue_b = clCreateCommandQueueWithProperties(context,
                                                                device,
                                                                props,
                                                                &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(queue_a, 2, 0);
  EXPECT_REF_COUNT(queue_b, 2, 0);

  EXPECT_SUCCESS(clReleaseCommandQueue(queue_a));
  EXPECT_REF_COUNT(queue_a, 1, 0);
  EXPECT_REF_COUNT(queue_b, 1, 0);

  EXPECT_SUCCESS(clReleaseCommandQueue(queue_b));
  EXPECT_DESTROYED(queue_a);
  EXPECT_DESTROYED(queue_b);

  cl_sampler sampler = clCreateSamplerWithProperties(context, nullptr, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(sampler, 1, 0);
  EXPECT_SUCCESS(clReleaseSampler(sampler));
  EXPECT_DESTROYED(sampler);

  cl_mem pipe = clCreatePipe(context, CL_MEM_READ_WRITE, 8, 16, nullptr, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(pipe, 1, 0);
  EXPECT_SUCCESS(clReleaseMemObject(pipe));
  EXPECT_DESTROYED(pipe);
}

void testCloneKernel(cl_platform_id platform, cl_device_id device) {
  cl_int status;
  cl_context context = createContext(platform, device);

  const char* source = "kernel void test_kernel(sampler_t sampler) {}";
  size_t length = strlen(source);
  cl_program program = clCreateProgramWithSource(context,
                                                 1,
                                                 &source,
                                                 &length,
                                                 &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(program, 1, 0);

  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_REF_COUNT(context, 0, 1);

  EXPECT_SUCCESS(clBuildProgram(program,
                                1,
                                &device,
                                nullptr,
                                nullptr,
                                nullptr));

  cl_kernel kernel = clCreateKernel(program, "test_kernel", &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(kernel, 1, 0);
  EXPECT_SUCCESS(clReleaseProgram(program));
  EXPECT_REF_COUNT(program, 0, 1);

  EXPECT_SUCCESS(clRetainKernel(kernel));
  EXPECT_REF_COUNT(kernel, 2, 0);

  cl_kernel clone = clCloneKernel(kernel, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(kernel, 2, 0);
  EXPECT_REF_COUNT(clone, 1, 0);
  EXPECT_REF_COUNT(program, 0, 2);

  EXPECT_SUCCESS(clReleaseKernel(kernel));
  EXPECT_SUCCESS(clReleaseKernel(kernel));
  EXPECT_DESTROYED(kernel);
  EXPECT_REF_COUNT(clone, 1, 0);
  EXPECT_REF_COUNT(program, 0, 1);

  EXPECT_SUCCESS(clReleaseKernel(clone));
  EXPECT_DESTROYED(clone);
  EXPECT_DESTROYED(program);
  EXPECT_DESTROYED(context);
}

void testCL300Calls(cl_platform_id platform, cl_device_id device) {
  cl_int status;
  cl_context context = createContext(platform, device);

  cl_mem buffer = clCreateBufferWithProperties(context,
                                               nullptr,
                                               CL_MEM_READ_WRITE,
                                               16,
                                               nullptr,
                                               &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(context, 1, 1);
  EXPECT_REF_COUNT(buffer, 1, 0);

  cl_image_desc desc = {
    CL_MEM_OBJECT_IMAGE2D, // image_type
    2,                     // image_width
    2,                     // image_height
    1,                     // image_depth
    1,                     // image_array size
    0,                     // image_row_pitch
    0,                     // image_slice_pitch
    0,                     // num_mip_levels
    0,                     // num_samples
    buffer                 // mem_object
  };
  cl_image_format format = {CL_R, CL_UNORM_INT8};
  cl_mem image = clCreateImageWithProperties(context,
                                             nullptr,
                                             CL_MEM_READ_WRITE,
                                             &format,
                                             &desc,
                                             nullptr,
                                             &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(image, 1, 0);
  EXPECT_REF_COUNT(buffer, 1, 1);

  EXPECT_SUCCESS(clReleaseMemObject(buffer));
  EXPECT_REF_COUNT(buffer, 0, 1);
  EXPECT_SUCCESS(clReleaseMemObject(image));
  EXPECT_DESTROYED(image);
  EXPECT_DESTROYED(buffer);
}

void testResurrect(cl_platform_id platform, cl_device_id device) {
  cl_int status;
  cl_context context = createContext(platform, device);

  cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 32, nullptr, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(buffer, 1, 0);

  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_REF_COUNT(context, 0, 1);

  // Try to 'resurrect' the context
  EXPECT_SUCCESS(clRetainContext(context));
  EXPECT_REF_COUNT(context, 1, 1);
  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_REF_COUNT(context, 0, 0);

  EXPECT_SUCCESS(clReleaseMemObject(buffer));
  EXPECT_DESTROYED(buffer);
  EXPECT_DESTROYED(context);
}

int main(int argc, char *argv[]) {
  if (!parseArgs(argc, argv)) {
    return EXIT_FAILURE;
  }

  cl_int status = CL_SUCCESS;
  cl_uint num_platforms = 0;
  EXPECT_SUCCESS(clGetPlatformIDs(0, nullptr, &num_platforms));

  if (num_platforms == 0) {
    REPORT_ERROR() << "No OpenCL platform detected" << std::endl;
    exit(-1);
  }

  auto platforms = std::make_unique<cl_platform_id[]>(num_platforms);
  EXPECT_SUCCESS(clGetPlatformIDs(num_platforms, platforms.get(), nullptr));

  // Select the mock icd.
  cl_platform_id platform;
  bool           test_platform_found = false;
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
    REPORT_ERROR() << "Platform '" << TEST_CONTEXT.platform << "' not found" << std::endl;
    exit(-1);
  }

  // Get a device
  cl_device_id device;
  cl_uint      numDevices;
  EXPECT_SUCCESS(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, &numDevices));
  if(numDevices == 0) {
    REPORT_ERROR() << "No OpenCL device found" << std::endl;
    exit(-1);
  }

  {
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
      REPORT_ERROR() << "Failed to parse platform OpenCL version" << std::endl;
    }

    TEST_CONTEXT.version = CL_MAKE_VERSION(major, minor, 0);
    std::cout << "platform opencl version is " << major << "." << minor << std::endl;
  }

  testBasicCounting(platform, device);
  testLifetimeEdgeCases(platform, device);
  testSubBuffer(platform, device);
  testPipeline(platform, device);

  if (TEST_CONTEXT.compareVersion(CL_MAKE_VERSION(1, 2, 0)) >= 0) {
    testSubImage(platform, device);
    testSubDevice(platform, device);
  }
  if (TEST_CONTEXT.compareVersion(CL_MAKE_VERSION(2, 0, 0)) >= 0) {
    testCL200Calls(platform, device);
  }
  if (TEST_CONTEXT.compareVersion(CL_MAKE_VERSION(2, 1, 0)) >= 0) {
    testCloneKernel(platform, device);
  }
  if (TEST_CONTEXT.compareVersion(CL_MAKE_VERSION(3, 0, 0)) >= 0) {
    testCL300Calls(platform, device);
  }
  if (TEST_CONTEXT.use_inaccessible_objects) {
    testResurrect(platform, device);
  }

  if (TEST_CONTEXT.failed_checks != 0) {
    std::cout << "failed " << TEST_CONTEXT.failed_checks << " checks" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
