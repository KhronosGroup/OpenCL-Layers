#include <ocl_program_cache/program_cache.hpp>

#include <CL/opencl.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string_view>
#include <string>
#include <vector>

#define CHECK_CL_BUILD_ERROR(expression)                                       \
    {                                                                          \
        if (const cl_int error = (expression); error != CL_SUCCESS)            \
            throw ::ocl::program_cache::opencl_build_error(error);             \
    }

namespace ocl::program_cache {
namespace {

#ifdef _WIN32

std::filesystem::path get_default_cache_root()
{
    std::size_t buffer_length{};
    std::wstring appdata_local;
    constexpr std::size_t max_size = 512;
    appdata_local.resize(max_size);
    if (_wgetenv_s(&buffer_length, appdata_local.data(), max_size,
                   L"LOCALAPPDATA"))
    {
        throw cache_access_error("Could not get default cache root directory");
    }
    else
    {
        appdata_local.resize(buffer_length);
        return std::filesystem::path(appdata_local) / "Khronos" / "OpenCL"
            / "cache";
    }
}

#elif defined(__APPLE__)

std::filesystem::path get_default_cache_root()
{
    if (char* home_path = std::getenv("HOME"); home_path == nullptr)
    {
        throw cache_access_error("Could not get default cache root directory");
    }
    else
    {
        return std::filesystem::path(home_path) / "Library" / "Caches"
            / "Khronos" / "OpenCL";
    }
}

#else

std::filesystem::path get_default_cache_root()
{
    const auto cache_home = []() -> std::filesystem::path {
        if (char* cache_home = std::getenv("XDG_CACHE_HOME");
            cache_home == nullptr)
        {
            if (char* home_path = std::getenv("HOME"); home_path == nullptr)
            {
                throw cache_access_error(
                    "Could not get default cache root directory");
            }
            else
            {
                return std::filesystem::path(home_path) / ".cache";
            }
        }
        else
        {
            return { cache_home };
        }
    }();
    return cache_home / "Khronos" / "OpenCL" / "cache";
}

#endif

std::string get_device_identifier(const cl::Device& device)
{
    const auto device_name = device.getInfo<CL_DEVICE_NAME>();
    const auto platform = cl::Platform(device.getInfo<CL_DEVICE_PLATFORM>());
    const auto platform_version = platform.getInfo<CL_PLATFORM_VERSION>();
    return platform_version + "/" + device_name;
}

std::vector<unsigned char> build_program_to_binary(const cl::Context& context,
                                                   const cl::Device& device,
                                                   std::string_view source,
                                                   std::string_view options)
{
    cl::Program program(context, std::string(source));
    CHECK_CL_BUILD_ERROR(program.build(device, options.data()));
    return program.getInfo<CL_PROGRAM_BINARIES>().front();
}

std::vector<unsigned char> build_program_to_binary(const cl::Context& context,
                                                   const cl::Device& device,
                                                   const std::vector<char>& il,
                                                   std::string_view options)
{
    cl::Program program(context, il);
    CHECK_CL_BUILD_ERROR(program.build(device, options.data()));
    return program.getInfo<CL_PROGRAM_BINARIES>().front();
}

void save_binary_to_cache(const std::filesystem::path& cache_path,
                          const std::vector<unsigned char>& binary_data)
{
    std::filesystem::create_directory(cache_path.parent_path());
    std::default_random_engine prng(std::random_device{}());
    std::stringstream sstream;
    sstream << std::setw(8) << std::setfill('0') << std::hex
            << std::uniform_int_distribution<unsigned int>{}(prng);
    const auto tmp_file_path =
        cache_path.parent_path() / (sstream.str() + ".tmp");
    {
        std::ofstream ofs(tmp_file_path, std::ios::binary);
        std::copy(binary_data.begin(), binary_data.end(),
                  std::ostreambuf_iterator(ofs));
    }
    try
    {
        std::filesystem::rename(tmp_file_path, cache_path);
    } catch (const std::filesystem::filesystem_error&)
    {
        // If the rename fails due to e.g. file being locked by another process
        // on Windows, silently return
    }
}

std::string hash_str(std::string_view data, std::string_view options = "")
{
    std::stringstream sstream;
    sstream << std::setfill('0') << std::setw(16) << std::hex
            << std::hash<std::string_view>{}(data)
            + std::hash<std::string_view>{}(options);
    return sstream.str();
}

std::string hash_str(const std::vector<char>& data,
                     std::string_view options = "")
{
    return hash_str(std::string(data.begin(), data.end()), options);
}

} // namespace

program_cache::program_cache(
    std::shared_ptr<const cl::Context> context,
    const std::optional<std::filesystem::path>& cache_root)
    : context_(context),
      cache_root_(cache_root.value_or(get_default_cache_root()))
{
    std::filesystem::create_directories(cache_root_);
}

std::optional<cl::Program> program_cache::fetch(std::string_view key) const
{
    return fetch(key,
                 (context_ == nullptr ? cl::Context::getDefault() : *context_)
                     .getInfo<CL_CONTEXT_DEVICES>());
}

std::optional<cl::Program>
program_cache::fetch(std::string_view key,
                     const std::vector<cl::Device>& devices) const
{
    std::vector<std::vector<unsigned char>> device_binaries;
    for (const auto& device : devices)
    {
        const auto cache_path =
            get_path_for_device_binary(device, hash_str(key));
        std::ifstream ifs(cache_path, std::ios::binary);
        if (!ifs.good())
        {
            return std::nullopt;
        }
        device_binaries.emplace_back(std::istreambuf_iterator<char>(ifs),
                                     std::istreambuf_iterator<char>());
    }
    cl::Program program(context_ == nullptr ? cl::Context::getDefault()
                                            : *context_,
                        devices, device_binaries);
    CHECK_CL_BUILD_ERROR(program.build());
    return program;
}

void program_cache::store(const cl::Program& program,
                          std::string_view key) const
{
    const auto build_statuses = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>();
    const bool unsuccessful_build =
        std::any_of(build_statuses.begin(), build_statuses.end(),
                    [](const auto& device_status) {
                        return device_status.second != CL_BUILD_SUCCESS;
                    });
    if (unsuccessful_build)
    {
        throw unbuilt_program_error();
    }
    const auto devices = program.getInfo<CL_PROGRAM_DEVICES>();
    const auto binaries = program.getInfo<CL_PROGRAM_BINARIES>();
    assert(devices.size() == binaries.size());
    auto device_it = devices.begin();
    for (auto binary_it = binaries.begin(), end = binaries.end();
         binary_it != end; ++binary_it, ++device_it)
    {
        const auto cache_path =
            get_path_for_device_binary(*device_it, hash_str(key));
        save_binary_to_cache(cache_path, *binary_it);
    }
}

cl::Program program_cache::fetch_or_build_source(std::string_view source,
                                                 std::string_view options) const
{
    return fetch_or_build_source(
        source, context_ == nullptr ? cl::Context::getDefault() : *context_,
        options);
}

cl::Program program_cache::fetch_or_build_source(std::string_view source,
                                                 const cl::Context& context,
                                                 std::string_view options) const
{
    return fetch_or_build_source(
        source, context, context.getInfo<CL_CONTEXT_DEVICES>(), options);
}

cl::Program
program_cache::fetch_or_build_source(std::string_view source,
                                     const cl::Context& context,
                                     const std::vector<cl::Device>& devices,
                                     std::string_view options) const
{
    // ToDo preprocess source
    return fetch_or_build_impl(source, context, devices, options);
}

cl::Program program_cache::fetch_or_build_il(const std::vector<char>& il,
                                             std::string_view options) const
{
    return fetch_or_build_il(
        il, context_ == nullptr ? cl::Context::getDefault() : *context_,
        options);
}

cl::Program program_cache::fetch_or_build_il(const std::vector<char>& il,
                                             const cl::Context& context,
                                             std::string_view options) const
{
    return fetch_or_build_il(il, context, context.getInfo<CL_CONTEXT_DEVICES>(),
                             options);
}

cl::Program
program_cache::fetch_or_build_il(const std::vector<char>& il,
                                 const cl::Context& context,
                                 const std::vector<cl::Device>& devices,
                                 std::string_view options) const
{
    return fetch_or_build_impl(il, context, devices, options);
}

template <class T>
cl::Program
program_cache::fetch_or_build_impl(const T& input,
                                   const cl::Context& context,
                                   const std::vector<cl::Device>& devices,
                                   std::string_view options) const
{
    const auto key_hash = hash_str(input, options);
    std::vector<std::vector<unsigned char>> program_binaries;
    std::transform(
        devices.begin(), devices.end(), std::back_inserter(program_binaries),
        [&](const auto& device) {
            auto cache_path = get_path_for_device_binary(device, key_hash);

            if (std::ifstream ifs(cache_path, std::ios::binary); ifs.good())
            {
                return std::vector<unsigned char>(
                    std::istreambuf_iterator<char>(ifs), {});
            }
            const auto program_binary =
                build_program_to_binary(context, device, input, options);
            save_binary_to_cache(cache_path, program_binary);
            return program_binary;
        });

    const auto program = cl::Program(context, devices, program_binaries);
    CHECK_CL_BUILD_ERROR(program.build());
    return program;
}

std::filesystem::path
program_cache::get_path_for_device_binary(const cl::Device& device,
                                          std::string_view key_hash) const
{
    const auto device_hash = hash_str(get_device_identifier(device));
    assert(key_hash.size() == 16);
    auto path =
        cache_root_ / std::string(key_hash.begin(), key_hash.begin() + 2);
    path /=
        std::string(key_hash.begin() + 2, key_hash.end()) + "_" + device_hash;
    return path;
}

} // namespace ocl::program_cache
