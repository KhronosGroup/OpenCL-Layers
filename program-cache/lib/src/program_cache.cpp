/*
 * Copyright (c) 2023 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * OpenCL is a trademark of Apple Inc. used under license by Khronos.
 */

#include <ocl_program_cache/program_cache.hpp>

#include <ocl_program_cache/common.hpp>

#include "preprocessor.hpp"
#include "utils.hpp"

#include <CL/opencl.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string_view>
#include <string>
#include <type_traits>
#include <vector>

namespace {
namespace pc = ocl::program_cache;

#ifdef _WIN32

std::filesystem::path get_default_cache_root()
{
    std::size_t buffer_length{};
    std::wstring appdata_local;
    constexpr std::size_t max_size = 512;
    appdata_local.resize(max_size);
    if (_wgetenv_s(&buffer_length, appdata_local.data(), max_size, L"LOCALAPPDATA"))
    {
        throw pc::cache_access_error("Could not get default cache root directory");
    }
    else
    {
        // Exclude the terminating \0
        appdata_local.resize(buffer_length - 1);
        return std::filesystem::path(appdata_local) / "Khronos" / "OpenCL" / "cache";
    }
}

#elif defined(__APPLE__)

std::filesystem::path get_default_cache_root()
{
    if (char* home_path = std::getenv("HOME"); home_path == nullptr)
    {
        throw pc::cache_access_error("Could not get default cache root directory");
    }
    else
    {
        return std::filesystem::path(home_path) / "Library" / "Caches" / "Khronos" / "OpenCL";
    }
}

#else

std::filesystem::path get_default_cache_root()
{
    const auto cache_home = []() -> std::filesystem::path {
        if (char* cache_home = std::getenv("XDG_CACHE_HOME"); cache_home == nullptr)
        {
            if (char* home_path = std::getenv("HOME"); home_path == nullptr)
            {
                throw ocl::program_cache::cache_access_error(
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

void save_binary_to_cache(const std::filesystem::path& cache_path,
                          const std::vector<unsigned char>& binary_data)
{
    std::filesystem::create_directory(cache_path.parent_path());
    std::default_random_engine prng(std::random_device{}());
    std::stringstream sstream;
    sstream << pc::utils::hex_format(std::uniform_int_distribution<unsigned int>{}(prng));
    const auto tmp_file_path = cache_path.parent_path() / (sstream.str() + ".tmp");
    {
        std::ofstream ofs(tmp_file_path, std::ios::binary);
        std::copy(binary_data.begin(), binary_data.end(), std::ostreambuf_iterator(ofs));
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
    const auto hash_value =
        std::hash<std::string_view>{}(data) + std::hash<std::string_view>{}(options);
    std::stringstream sstream;
    sstream << pc::utils::hex_format(hash_value);
    return sstream.str();
}

std::string hash_str(const std::vector<char>& data, std::string_view options = "")
{
    return hash_str(std::string(data.begin(), data.end()), options);
}

} // namespace

pc::program_cache::program_cache(const program_cache_dispatch& dispatch,
                                 cl_context context,
                                 const std::optional<std::filesystem::path>& cache_root)
    : dispatch_(dispatch), context_(context),
      cache_root_(cache_root.value_or(get_default_cache_root()))
{
    std::filesystem::create_directories(cache_root_);
}

cl_program pc::program_cache::fetch(std::string_view key) const
{
    return fetch(key, get_devices(context_ ? context_ : get_default_context()));
}

cl_program pc::program_cache::fetch(std::string_view key,
                                    const std::vector<cl_device_id>& devices) const
{
    std::vector<std::vector<unsigned char>> device_binaries;
    std::vector<std::size_t> binary_lengths;
    std::vector<const unsigned char*> binary_ptrs;
    for (const auto& device : devices)
    {
        const auto cache_path = get_path_for_device_binary(device, hash_str(key));
        std::ifstream ifs(cache_path, std::ios::binary);
        if (!ifs.good())
        {
            return nullptr;
        }
        auto& binary_data = device_binaries.emplace_back(std::istreambuf_iterator<char>(ifs),
                                                         std::istreambuf_iterator<char>());
        binary_lengths.push_back(binary_data.size());
        binary_ptrs.push_back(binary_data.data());
    }
    cl_int error = CL_SUCCESS;
    const cl_program program = dispatch_.clCreateProgramWithBinary(
        context_ ? context_ : get_default_context(), static_cast<cl_uint>(devices.size()),
        devices.data(), binary_lengths.data(), binary_ptrs.data(), nullptr, &error);
    CHECK_CL_ERROR(error);
    error = dispatch_.clBuildProgram(program, static_cast<cl_uint>(devices.size()), devices.data(),
                                     nullptr, nullptr, nullptr);
    if (error != CL_SUCCESS)
    {
        dispatch_.clReleaseProgram(program);
        throw opencl_build_error(error);
    }
    return program;
}

void pc::program_cache::store(cl_program program, std::string_view key) const
{
    cl_uint num_devices{};
    CHECK_CL_ERROR(dispatch_.clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(num_devices),
                                              &num_devices, nullptr));
    std::vector<cl_device_id> program_devices(num_devices);
    CHECK_CL_ERROR(dispatch_.clGetProgramInfo(program, CL_PROGRAM_DEVICES,
                                              num_devices * sizeof(cl_device_id),
                                              program_devices.data(), nullptr));
    for (auto device_id : program_devices)
    {
        cl_build_status build_status{};
        CHECK_CL_ERROR(dispatch_.clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS,
                                                       sizeof(build_status), &build_status,
                                                       nullptr));
        if (build_status != CL_BUILD_SUCCESS)
        {
            throw unbuilt_program_error();
        }
    }
    std::vector<std::size_t> binary_sizes(program_devices.size());
    CHECK_CL_ERROR(dispatch_.clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                                              binary_sizes.size() * sizeof(std::size_t),
                                              binary_sizes.data(), nullptr));
    std::vector<std::vector<unsigned char>> program_binaries;
    std::vector<unsigned char*> binary_ptrs;
    for (auto binary_size : binary_sizes)
    {
        auto& binary_data = program_binaries.emplace_back(binary_size);
        binary_ptrs.push_back(binary_data.data());
    }
    CHECK_CL_ERROR(dispatch_.clGetProgramInfo(program, CL_PROGRAM_BINARIES,
                                              binary_ptrs.size() * sizeof(unsigned char*),
                                              binary_ptrs.data(), nullptr));
    auto device_it = program_devices.begin();
    for (auto binary_it = program_binaries.begin(), end = program_binaries.end(); binary_it != end;
         ++binary_it, ++device_it)
    {
        const auto cache_path = get_path_for_device_binary(*device_it, hash_str(key));
        save_binary_to_cache(cache_path, *binary_it);
    }
}

cl_program pc::program_cache::fetch_or_build_source(std::string_view source,
                                                    std::string_view options) const
{
    return fetch_or_build_source(source, context_ ? context_ : get_default_context(), options);
}

cl_program pc::program_cache::fetch_or_build_source(std::string_view source,
                                                    cl_context context,
                                                    std::string_view options) const
{
    return fetch_or_build_source(source, context, get_devices(context), options);
}

cl_program pc::program_cache::fetch_or_build_source(std::string_view source,
                                                    cl_context context,
                                                    const std::vector<cl_device_id>& devices,
                                                    std::string_view options) const
{
    return fetch_or_build_impl(source, context, devices.empty() ? get_devices(context) : devices,
                               options);
}

cl_program pc::program_cache::fetch_or_build_il(const std::vector<char>& il,
                                                std::string_view options) const
{
    return fetch_or_build_il(il, context_ ? context_ : get_default_context(), options);
}

cl_program pc::program_cache::fetch_or_build_il(const std::vector<char>& il,
                                                cl_context context,
                                                std::string_view options) const
{
    return fetch_or_build_il(il, context, get_devices(context), options);
}

cl_program pc::program_cache::fetch_or_build_il(const std::vector<char>& il,
                                                cl_context context,
                                                const std::vector<cl_device_id>& devices,
                                                std::string_view options) const
{
    return fetch_or_build_impl(il, context, devices.empty() ? get_devices(context) : devices,
                               options);
}

template <class T>
cl_program pc::program_cache::fetch_or_build_impl(const T& input,
                                                  cl_context context,
                                                  const std::vector<cl_device_id>& devices,
                                                  std::string_view options) const
{
    std::vector<std::vector<unsigned char>> program_binaries;
    std::transform(devices.begin(), devices.end(), std::back_inserter(program_binaries),
                   [&](const auto& device) {
                       // If input is string_view, preprocessed_input is a new value
                       // Else (input is std::vector<char>), preprocessed_input is a
                       // const& to input
                       decltype(auto) preprocessed_input = [&] {
                           if constexpr (std::is_same_v<T, std::string_view>)
                           {
                               return preprocess(input, device, options, this->dispatch_);
                           }
                           else
                           {
                               return input;
                           }
                       }();
                       const auto key_hash = hash_str(preprocessed_input, options);
                       auto cache_path = get_path_for_device_binary(device, key_hash);

                       if (std::ifstream ifs(cache_path, std::ios::binary); ifs.good())
                       {
                           return std::vector<unsigned char>(std::istreambuf_iterator<char>(ifs),
                                                             {});
                       }
                       const auto program_binary =
                           build_program_to_binary(context, device, preprocessed_input, options);
                       save_binary_to_cache(cache_path, program_binary);
                       return program_binary;
                   });
    std::vector<std::size_t> binary_lengths;
    std::vector<const unsigned char*> binary_ptrs;
    for (const auto& binary_data : program_binaries)
    {
        binary_lengths.push_back(binary_data.size());
        binary_ptrs.push_back(binary_data.data());
    }

    cl_int error = CL_SUCCESS;
    const auto program = dispatch_.clCreateProgramWithBinary(
        context, static_cast<cl_uint>(devices.size()), devices.data(), binary_lengths.data(),
        binary_ptrs.data(), nullptr, &error);
    CHECK_CL_ERROR(error);
    error = dispatch_.clBuildProgram(program, static_cast<cl_uint>(devices.size()), devices.data(),
                                     nullptr, nullptr, nullptr);
    if (error != CL_SUCCESS)
    {
        dispatch_.clReleaseProgram(program);
        throw opencl_build_error(error);
    }
    return program;
}

std::filesystem::path pc::program_cache::get_path_for_device_binary(cl_device_id device,
                                                                    std::string_view key_hash) const
{
    const auto device_name = utils::get_info_str(device, dispatch_.clGetDeviceInfo, CL_DEVICE_NAME);
    cl_platform_id platform{};
    CHECK_CL_ERROR(dispatch_.clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform),
                                             &platform, nullptr));
    const auto platform_version =
        utils::get_info_str(platform, dispatch_.clGetPlatformInfo, CL_PLATFORM_VERSION);
    const auto device_hash = hash_str(platform_version + "/" + device_name);
    assert(key_hash.size() == 16);
    auto path = cache_root_ / std::string(key_hash.begin(), key_hash.begin() + 2);
    path /= std::string(key_hash.begin() + 2, key_hash.end()) + "_" + device_hash;
    return path;
}

template <class T>
std::vector<unsigned char> pc::program_cache::build_program_to_binary(
    cl_context context, cl_device_id device, const T& source, std::string_view options) const
{
    cl_int error = CL_SUCCESS;
    const auto program = [&]() -> cl_program {
        const char* source_data = source.data();
        const std::size_t source_size = source.size();
        if constexpr (std::is_same_v<T, std::string>)
        {
            return dispatch_.clCreateProgramWithSource(context, 1, &source_data, &source_size,
                                                       &error);
        }
        else if constexpr (std::is_same_v<T, std::vector<char>>)
        {
            return dispatch_.clCreateProgramWithIL(context, source_data, source_size, &error);
        }
        else
        {
            static_assert(false, "T is expected to be std::string or std::vector<char>");
        }
    }();
    if (error != CL_SUCCESS)
    {
        dispatch_.clReleaseProgram(program);
        throw opencl_error(error);
    }
    const std::string options_str(options.begin(), options.end());
    error = dispatch_.clBuildProgram(program, 1, &device, options_str.c_str(), nullptr, nullptr);
    if (error != CL_SUCCESS)
    {
        dispatch_.clReleaseProgram(program);
        throw opencl_error(error);
    }
    std::size_t binary_size{};
    error = dispatch_.clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(binary_size),
                                       &binary_size, nullptr);
    if (error != CL_SUCCESS)
    {
        dispatch_.clReleaseProgram(program);
        throw opencl_error(error);
    }
    std::vector<unsigned char> binaries(binary_size);
    unsigned char* binary_data = binaries.data();
    error = dispatch_.clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(binary_data),
                                       &binary_data, nullptr);
    dispatch_.clReleaseProgram(program);
    if (error != CL_SUCCESS)
    {
        throw opencl_error(error);
    }
    return binaries;
}

cl_context pc::program_cache::get_default_context() const
{
    static cl_context default_context = [&] {
        cl_uint num_platforms{};
        CHECK_CL_ERROR(dispatch_.clGetPlatformIDs(0, nullptr, &num_platforms));
        std::vector<cl_platform_id> platform_ids(num_platforms);
        CHECK_CL_ERROR(dispatch_.clGetPlatformIDs(num_platforms, platform_ids.data(), nullptr));
        std::array<cl_context_properties, 3> props{
            CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform_ids.front()), 0
        };
        cl_int error = CL_SUCCESS;
        auto context = dispatch_.clCreateContextFromType(props.data(), CL_DEVICE_TYPE_ALL, nullptr,
                                                         nullptr, &error);
        CHECK_CL_ERROR(error);
        return context;
    }();
    // TODO release on exit
    return default_context;
}

std::vector<cl_device_id> pc::program_cache::get_devices(cl_context context) const
{
    std::size_t num_devices{};
    CHECK_CL_ERROR(dispatch_.clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(num_devices),
                                              &num_devices, nullptr));
    std::vector<cl_device_id> device_ids(num_devices);
    CHECK_CL_ERROR(dispatch_.clGetContextInfo(context, CL_CONTEXT_DEVICES,
                                              num_devices * sizeof(cl_device_id), device_ids.data(),
                                              nullptr));
    return device_ids;
}
