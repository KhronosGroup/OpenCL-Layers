#include <ocl_program_cache/program_cache.hpp>

#include <CL/opencl.hpp>
#include <gtest/gtest-printers.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>

using ocl::program_cache::program_cache;

#ifdef _WIN32
namespace testing::internal {
// Passing std::filesystem::path containing non-ASCII characters to ostream
// crashes on Windows
template <> void PrintTo(const std::filesystem::path&, std::ostream*) {}
}
#endif

class ProgramCacheTest : public testing::TestWithParam<std::filesystem::path> {
protected:
    const cl::Context& context() const { return *context_; }
    const program_cache& cache() const { return *cache_; }

    void SetUp() override
    {
        const auto cache_path = std::filesystem::current_path() / GetParam();
        if (std::filesystem::exists(cache_path))
        {
            std::filesystem::remove_all(cache_path);
        }
        context_ = std::make_shared<cl::Context>(cl::Context::getDefault());
        cache_ = std::make_unique<program_cache>(context_, cache_path);
    }

    std::string get_program_source(int i = 100)
    {
        return "kernel void foo(global int* i) { *i = " + std::to_string(i)
            + "; }";
    }

    std::vector<char> get_program_il()
    {
        // $ echo "kernel void foo(global int* i) { *i = 100; }" > kernel.cl
        // $ clang -c --target=spirv64 -o kernel.spv64 kernel.cl
        return { 3,   2,   35,  7,   0,  0,  1,  0,   14,  0,   6,    0,   14,
                 0,   0,   0,   0,   0,  0,  0,  17,  0,   2,   0,    4,   0,
                 0,   0,   17,  0,   2,  0,  5,  0,   0,   0,   17,   0,   2,
                 0,   6,   0,   0,   0,  11, 0,  5,   0,   1,   0,    0,   0,
                 79,  112, 101, 110, 67, 76, 46, 115, 116, 100, 0,    0,   14,
                 0,   3,   0,   2,   0,  0,  0,  2,   0,   0,   0,    15,  0,
                 4,   0,   6,   0,   0,  0,  10, 0,   0,   0,   102,  111, 111,
                 0,   3,   0,   3,   0,  3,  0,  0,   0,   112, -114, 1,   0,
                 5,   0,   3,   0,   6,  0,  0,  0,   102, 111, 111,  0,   71,
                 0,   5,   0,   6,   0,  0,  0,  41,  0,   0,   0,    102, 111,
                 111, 0,   0,   0,   0,  0,  71, 0,   4,   0,   7,    0,   0,
                 0,   38,  0,   0,   0,  5,  0,  0,   0,   71,  0,    4,   0,
                 7,   0,   0,   0,   44, 0,  0,  0,   4,   0,   0,    0,   71,
                 0,   4,   0,   11,  0,  0,  0,  38,  0,   0,   0,    5,   0,
                 0,   0,   71,  0,   4,  0,  11, 0,   0,   0,   44,   0,   0,
                 0,   4,   0,   0,   0,  21, 0,  4,   0,   3,   0,    0,   0,
                 32,  0,   0,   0,   0,  0,  0,  0,   43,  0,   4,    0,   3,
                 0,   0,   0,   9,   0,  0,  0,  100, 0,   0,   0,    19,  0,
                 2,   0,   2,   0,   0,  0,  32, 0,   4,   0,   4,    0,   0,
                 0,   5,   0,   0,   0,  3,  0,  0,   0,   33,  0,    4,   0,
                 5,   0,   0,   0,   2,  0,  0,  0,   4,   0,   0,    0,   54,
                 0,   5,   0,   2,   0,  0,  0,  6,   0,   0,   0,    0,   0,
                 0,   0,   5,   0,   0,  0,  55, 0,   3,   0,   4,    0,   0,
                 0,   7,   0,   0,   0,  -8, 0,  2,   0,   8,   0,    0,   0,
                 62,  0,   5,   0,   7,  0,  0,  0,   9,   0,   0,    0,   2,
                 0,   0,   0,   4,   0,  0,  0,  -3,  0,   1,   0,    56,  0,
                 1,   0,   54,  0,   5,  0,  2,  0,   0,   0,   10,   0,   0,
                 0,   0,   0,   0,   0,  5,  0,  0,   0,   55,  0,    3,   0,
                 4,   0,   0,   0,   11, 0,  0,  0,   -8,  0,   2,    0,   12,
                 0,   0,   0,   57,  0,  5,  0,  2,   0,   0,   0,    13,  0,
                 0,   0,   6,   0,   0,  0,  11, 0,   0,   0,   -3,   0,   1,
                 0,   56,  0,   1,   0 };
    }

    void check_program(const cl::Program& program, int i = 100)
    {
        cl::Buffer output(context(), CL_MEM_WRITE_ONLY, sizeof(int));
        cl::KernelFunctor<cl::Buffer> kernel_functor(program, "foo");
        kernel_functor(cl::EnqueueArgs(cl::NDRange(1)), output);
        int h_out{};
        cl::enqueueReadBuffer(output, true, 0, sizeof(h_out), &h_out);
        ASSERT_EQ(i, h_out);
    }

private:
    std::shared_ptr<cl::Context> context_;
    std::unique_ptr<program_cache> cache_;
};

INSTANTIATE_TEST_SUITE_P(
    ProgramCacheTest,
    ProgramCacheTest,
    testing::Values(std::filesystem::path("test-cache"),
                    // "Each person must strive for the completion and
                    // perfection of one's [UTF-8] character"
                    std::filesystem::u8path("人格 完成に努める こと")));

TEST(ProgramCacheBasicTest, InstantiateDefaultCache) { program_cache cache; }

TEST_P(ProgramCacheTest, UnbuiltProgramThrows)
{
    cl::Program program(context(), get_program_source());
    const std::string key = "abcdef";
    ASSERT_THROW(cache().store(program, key),
                 ocl::program_cache::unbuilt_program_error);
}

TEST_P(ProgramCacheTest, StoreAndFetch)
{
    cl::Program program(context(), get_program_source());
    ASSERT_EQ(CL_SUCCESS, program.build());

    const std::string key = "abcdef";
    cache().store(program, key);

    program = *cache().fetch(key);
    check_program(program);
}

TEST_P(ProgramCacheTest, CacheSimpleTextProgram)
{
    const auto source_100 = get_program_source(100);
    const auto source_5 = get_program_source(5);

    auto program_100 = cache().fetch_or_build_source(source_100);
    auto program_5 = cache().fetch_or_build_source(source_5);
    check_program(program_5, 5);
    check_program(program_100, 100);

    // The subsequent lookups should be cache hits
    program_5 = cache().fetch_or_build_source(source_5);
    program_100 = cache().fetch_or_build_source(source_100);
    check_program(program_5, 5);
    check_program(program_100, 100);
}

TEST_P(ProgramCacheTest, CacheSimpleILProgram)
{
    const auto devices = context().getInfo<CL_CONTEXT_DEVICES>();
    const bool no_il_version = std::any_of(
        devices.begin(), devices.end(), [](const cl::Device& device) {
            const auto il_version = device.getInfo<CL_DEVICE_IL_VERSION>();
            return il_version.empty();
        });
    if (no_il_version)
    {
        GTEST_SKIP();
    }

    const auto program_100 = cache().fetch_or_build_il(get_program_il());
    check_program(program_100, 100);
}

TEST_P(ProgramCacheTest, ParallelAccessToCache)
{
    using namespace std::chrono_literals;

    constexpr std::size_t num_futures = 16;
    constexpr std::size_t num_programs_per_thread = 20;

    std::vector<std::future<void>> futures;
    for (std::size_t i = 0; i < num_futures; ++i)
    {
        futures.push_back(std::async(std::launch::async, [&] {
            std::vector<int> program_indices(num_programs_per_thread);
            std::iota(program_indices.begin(), program_indices.end(), 1);
            std::default_random_engine prng(std::random_device{}());
            std::shuffle(program_indices.begin(), program_indices.end(), prng);
            for (auto idx : program_indices)
            {
                const auto program =
                    cache().fetch_or_build_source(get_program_source(idx));
                check_program(program, idx);
            }
        }));
    }

    for (auto& future : futures)
    {
        future.get();
    }
}
