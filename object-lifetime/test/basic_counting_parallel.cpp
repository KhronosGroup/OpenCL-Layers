#include "object_lifetime_test.hpp"

#include <vector>
#include <algorithm>
#include <iterator>
#include <thread>
#include <random>

int main(int argc, char *argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  layer_test::setup(argc, argv, CL_MAKE_VERSION(1, 1, 0), platform, device);

  cl_context context = layer_test::createContext(platform, device);

  constexpr size_t increment_count = 100;
  constexpr size_t contention_factor = 1;

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

  std::vector<cl_event> increments;
  for (auto event : events)
    std::fill_n(std::back_inserter(increments), increment_count, event);
  std::shuffle(increments.begin(), increments.end(), std::default_random_engine{});

  layer_test::parallel_for(increments.begin(), increments.end(), [](const cl_event& event)
  {
    cl_int err = clRetainEvent(event);
    EXPECT_SUCCESS(err);
  });

  for (auto event : events)
    EXPECT_REF_COUNT(event, increment_count + 1, 0);
  EXPECT_REF_COUNT(context, 1, static_cast<cl_uint>(events.size()));

  clReleaseContext(context);
  layer_test::parallel_for(events.begin(), events.end(), [=](const cl_event& event)
  {
    for (size_t i = 0 ; i < increment_count + 1; ++i)
    {
      cl_int err = clReleaseEvent(event);
      EXPECT_SUCCESS(err);
    }
    EXPECT_DESTROYED(event);
  });
  EXPECT_DESTROYED(context);

  return layer_test::finalize();
}
