#include "object_lifetime_test.hpp"

#include <vector>
#include <algorithm>
#include <iterator>
#include <thread>
#include <random>
#include <utility>

typedef enum object_type_e {
  OCL_PLATFORM,
  OCL_DEVICE,
  OCL_SUB_DEVICE,
  OCL_CONTEXT,
  OCL_COMMAND_QUEUE,
  OCL_MEM,
  OCL_BUFFER,
  OCL_IMAGE,
  OCL_PIPE,
  OCL_PROGRAM,
  OCL_KERNEL,
  OCL_EVENT,
  OCL_SAMPLER,
  OBJECT_TYPE_MAX,
} object_type;

int main(int argc, char *argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  layer_test::setup(argc, argv, CL_MAKE_VERSION(1, 1, 0), platform, device);

  cl_context context = layer_test::createContext(platform, device);

  constexpr size_t increment_count = 100;
  constexpr size_t contention_factor = 1;
  const unsigned int num_sub_devices = std::thread::hardware_concurrency();

  // Create devices
  cl_device_id root_device;
  {
    cl_int err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &root_device, nullptr);
    EXPECT_SUCCESS(err);
  }
  cl_uint num_cus;
  EXPECT_SUCCESS(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num_cus, nullptr));
  cl_uint num_sub_cus = num_cus / num_sub_devices;
  cl_device_partition_property props[] = {CL_DEVICE_PARTITION_EQUALLY, static_cast<cl_device_partition_property>(num_sub_cus), 0};
  std::vector<cl_device_id> sub_devices(num_sub_devices);
  EXPECT_SUCCESS(clCreateSubDevices(root_device,
                                    props,
                                    static_cast<cl_uint>(sub_devices.size()),
                                    sub_devices.data(),
                                    nullptr));

  // Create buffers
  std::vector<cl_mem> buffers;
  std::generate_n(
    std::back_inserter(buffers),
    std::thread::hardware_concurrency() / contention_factor,
    [=]()
    {
      cl_int err;
      auto res = clCreateBuffer(context, CL_MEM_READ_ONLY, 1, nullptr, &err);
      EXPECT_SUCCESS(err);
      return res;
    }
  );
  std::vector<cl_mem> sub_buffers;
  std::transform(
    buffers.cbegin(),
    buffers.cend(),
    std::back_inserter(sub_buffers),
    [](const cl_mem& buf)
    {
      cl_int err;
      cl_buffer_region sub_region = {0, 1};
      auto res = clCreateSubBuffer(buf, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &sub_region, &err);
      EXPECT_SUCCESS(err);
      return res;
    }
  );

  // Create events
  std::vector<cl_event> events;
  std::generate_n(
    std::back_inserter(events),
    std::thread::hardware_concurrency() / contention_factor,
    [=]()
    {
      cl_int err;
      auto res = clCreateUserEvent(context, &err);
      EXPECT_SUCCESS(err);
      return res;
    }
  );

  // Create programs
  std::vector<cl_program> programs;
  std::generate_n(
    std::back_inserter(programs),
    std::thread::hardware_concurrency() / contention_factor,
    [&, source = std::string{"kernel void copy(global int* a, global int* b){ a[0] = b[0]; }"}]()
    {
      cl_int err;
      const char* sources = source.c_str();
      size_t lengths = std::strlen(sources);
      auto res = clCreateProgramWithSource(context, 1, &sources, &lengths, &err);
      EXPECT_SUCCESS(err);
      err = clBuildProgram(res, 1, &root_device, "", nullptr, nullptr);
      EXPECT_SUCCESS(err);
      return res;
    }
  );

  // Create kernels
  std::vector<cl_kernel> kernels;
  std::transform(
    programs.cbegin(),
    programs.cend(),
    std::back_inserter(kernels),
    [](const cl_program& prog)
    {
      cl_int err;
      auto res = clCreateKernel(prog, "copy", &err);
      EXPECT_SUCCESS(err);
      return res;
    }
  );

  // Create samplers
  std::vector<cl_sampler> samplers;
  std::generate_n(
    std::back_inserter(samplers),
    std::thread::hardware_concurrency() / contention_factor,
    [=]()
    {
      cl_int err;
      auto res = clCreateSampler(context, false, CL_ADDRESS_REPEAT, CL_FILTER_NEAREST, &err);
      EXPECT_SUCCESS(err);
      return res;
    }
  );

  // Create type-erased increments
  std::vector<std::pair<object_type_e, void*>> increments;
  for (auto sub_device : sub_devices)
    std::fill_n(std::back_inserter(increments), increment_count, std::make_pair<object_type_e, void*>(OCL_SUB_DEVICE, reinterpret_cast<void*>(sub_device)));
  for (auto buffer : buffers)
    std::fill_n(std::back_inserter(increments), increment_count, std::make_pair<object_type_e, void*>(OCL_BUFFER, reinterpret_cast<void*>(buffer)));
  for (auto sub_buffer : sub_buffers)
    std::fill_n(std::back_inserter(increments), increment_count, std::make_pair<object_type_e, void*>(OCL_BUFFER, reinterpret_cast<void*>(sub_buffer)));
  for (auto event : events)
    std::fill_n(std::back_inserter(increments), increment_count, std::make_pair<object_type_e, void*>(OCL_EVENT, reinterpret_cast<void*>(event)));
  for (auto program : programs)
    std::fill_n(std::back_inserter(increments), increment_count, std::make_pair<object_type_e, void*>(OCL_PROGRAM, reinterpret_cast<void*>(program)));
  for (auto kernel : kernels)
    std::fill_n(std::back_inserter(increments), increment_count, std::make_pair<object_type_e, void*>(OCL_KERNEL, reinterpret_cast<void*>(kernel)));
  for (auto sampler : samplers)
    std::fill_n(std::back_inserter(increments), increment_count, std::make_pair<object_type_e, void*>(OCL_SAMPLER, reinterpret_cast<void*>(sampler)));
  std::shuffle(increments.begin(), increments.end(), std::default_random_engine{});

  layer_test::parallel_for(increments.begin(), increments.end(), [](const std::pair<object_type_e, void*>& object)
  {
    cl_int err = CL_SUCCESS;
    switch(object.first)
    {
      case OCL_SUB_DEVICE:
        err = clRetainDevice(reinterpret_cast<cl_device_id>(object.second));
        break;
      case OCL_BUFFER:
        err = clRetainMemObject(reinterpret_cast<cl_mem>(object.second));
        break;
      case OCL_EVENT:
        err = clRetainEvent(reinterpret_cast<cl_event>(object.second));
        break;
      case OCL_PROGRAM:
        err = clRetainProgram(reinterpret_cast<cl_program>(object.second));
        break;
      case OCL_KERNEL:
        err = clRetainKernel(reinterpret_cast<cl_kernel>(object.second));
        break;
      case OCL_SAMPLER:
        err = clRetainSampler(reinterpret_cast<cl_sampler>(object.second));
        break;
      default:
        std::cerr << "Unkown object type" << std::endl;
        std::exit(-1);
    }
    EXPECT_SUCCESS(err);
  });

  for (auto sub_device : sub_devices)
    EXPECT_REF_COUNT(sub_device, increment_count + 1, 0);
  for (auto buffer : buffers)
    EXPECT_REF_COUNT(buffer, increment_count + 1, 1);
  for (auto sub_buffer : sub_buffers)
    EXPECT_REF_COUNT(sub_buffer, increment_count + 1, 0);
  for (auto event : events)
    EXPECT_REF_COUNT(event, increment_count + 1, 0);
  for (auto program : programs)
    EXPECT_REF_COUNT(program, increment_count + 1, 1);
  for (auto kernel : kernels)
    EXPECT_REF_COUNT(kernel, increment_count + 1, 0);
  for (auto sampler : samplers)
    EXPECT_REF_COUNT(sampler, increment_count + 1, 0);

  clReleaseContext(context);
  layer_test::parallel_for(increments.begin(), increments.end(), [](const std::pair<object_type_e, void*>& object)
  {
    cl_int err = CL_SUCCESS;
    switch(object.first)
    {
      case OCL_SUB_DEVICE:
        err = clReleaseDevice(reinterpret_cast<cl_device_id>(object.second));
        break;
      case OCL_BUFFER:
        err = clReleaseMemObject(reinterpret_cast<cl_mem>(object.second));
        break;
      case OCL_EVENT:
        err = clReleaseEvent(reinterpret_cast<cl_event>(object.second));
        break;
      case OCL_PROGRAM:
        err = clReleaseProgram(reinterpret_cast<cl_program>(object.second));
        break;
      case OCL_KERNEL:
        err = clReleaseKernel(reinterpret_cast<cl_kernel>(object.second));
        break;
      case OCL_SAMPLER:
        err = clReleaseSampler(reinterpret_cast<cl_sampler>(object.second));
        break;
      default:
        std::cerr << "Unkown object type" << std::endl;
        std::exit(-1);
    }
    EXPECT_SUCCESS(err);
  });

  // Undo the initial retain
  for (auto sub_device : sub_devices)
    clReleaseDevice(sub_device);
  for (auto buffer : buffers)
    clReleaseMemObject(buffer);
  for (auto sub_buffer : sub_buffers)
    clReleaseMemObject(sub_buffer);
  for (auto event : events)
    clReleaseEvent(event);
  for (auto program : programs)
    clReleaseProgram(program);
  for (auto kernel : kernels)
    clReleaseKernel(kernel);
  for (auto sampler : samplers)
    clReleaseSampler(sampler);

  for (auto sub_device : sub_devices)
    EXPECT_DESTROYED(sub_device);
  for (auto buffer : buffers)
    EXPECT_DESTROYED(buffer);
  for (auto sub_buffer : sub_buffers)
    EXPECT_DESTROYED(sub_buffer);
  for (auto event : events)
    EXPECT_DESTROYED(event);
  for (auto program : programs)
    EXPECT_DESTROYED(program);
  for (auto kernel : kernels)
    EXPECT_DESTROYED(kernel);
  for (auto sampler : samplers)
    EXPECT_DESTROYED(sampler);

  EXPECT_DESTROYED(context);

  return layer_test::finalize();
}
