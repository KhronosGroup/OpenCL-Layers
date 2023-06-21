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

#ifndef OCL_PROGRAM_CACHE_LIB_INC_OCL_PROGRAM_CACHE_PROGRAM_CACHE_HPP_
#define OCL_PROGRAM_CACHE_LIB_INC_OCL_PROGRAM_CACHE_PROGRAM_CACHE_HPP_

#include <CL/opencl.hpp>

#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <string>
#include <vector>

#include <cstddef>

namespace ocl::program_cache {

/// @brief Error thrown when the path for the cache cannot be accessed.
struct cache_access_error : public std::runtime_error
{
    cache_access_error(const std::string& what_arg)
        : std::runtime_error(what_arg)
    {}
};

/// @brief Error thrown when the passed OpenCL program could not be built.
struct opencl_build_error : public std::runtime_error
{
    opencl_build_error(cl_int error)
        : std::runtime_error("An OpenCL kernel build error occured: "
                             + std::to_string(error))
    {}
};

/// @brief Error thrown when a cl::Program is passed which should have been
/// built previously.
struct unbuilt_program_error : public std::runtime_error
{
    unbuilt_program_error()
        : std::runtime_error("The passed program has not been built")
    {}
};

/// @brief Store and fetch OpenCL program binaries to/from the filesystem.
class program_cache {
public:
    /// @brief Creates a new instance of the \c program_cache.
    /// @param context OpenCL context to build the programs for. If \c nullptr
    /// is passed, then the default context is used.
    /// @param cache_root Path to the program cache root on the filesystem. If
    /// \c nullopt is passed, the platform dependent default location is used.
    program_cache(
        std::shared_ptr<const cl::Context> context = nullptr,
        const std::optional<std::filesystem::path>& cache_root = std::nullopt);

    /// @brief Loads cached binaries for all devices associated with the \c
    /// cl::Context passed in the constructor and builds a \c cl::Program.
    /// @param key Key to the cache entries. The key must be equal to the key
    /// passed to a previous \c store call to retrieve the same binaries.
    /// @return The built \c cl::Program if a cache entry was found for all
    /// devices, \c std::nullopt otherwise.
    std::optional<cl::Program> fetch(std::string_view key) const;

    /// @brief Loads cached binaries for all devices passed and returns a built
    /// \c cl::Program.
    /// @param key Key to the cache entries. The key must be equal to the key
    /// passed to a previous \c store call to retrieve the same binaries.
    /// @param devices The devices to load the programs for.
    /// @return The built \c cl::Program if a cache entry was found for all
    /// devices, \c std::nullopt otherwise.
    std::optional<cl::Program>
    fetch(std::string_view key, const std::vector<cl::Device>& devices) const;

    /// @brief Stores the binary representation of a \c cl::Program in the
    /// cache.
    /// @param program The program to store. It must be built previously,
    /// otherwise \c unbuilt_program_error is thrown.
    /// @param key The key to the cache entry, which can be used to retrieve the
    /// cache entry later via \c fetch.
    void store(const cl::Program& program, std::string_view key) const;

    /// @brief Builds OpenCL source code to a \c cl::Program and stores it in
    /// the cache. If the program existed in the cache previously, loads it back
    /// from the cache without building.
    /// @param source Source code of the program.
    /// @param options Build options that are passed to the OpenCL compiler.
    /// @return The built OpenCL program.
    /// @note The cache lookup considers the passed source code, the passed
    /// options, the platform's version and the devices to which the code is
    /// compiled. In this overload, the code is compiled to all devices
    /// associated with the \c cl::Context passed in the constructor.
    cl::Program fetch_or_build_source(std::string_view source,
                                      std::string_view options = {}) const;

    /// @brief Builds OpenCL source code to a \c cl::Program and stores it in
    /// the cache. If the program existed in the cache previously, loads it back
    /// from the cache without building.
    /// @param source Source code of the program.
    /// @param context The OpenCL context to compile for.
    /// @param options Build options that are passed to the OpenCL compiler.
    /// @return The built OpenCL program.
    /// @note The cache lookup considers the passed source code, the passed
    /// options, the platform's version and the devices to which the code is
    /// compiled. In this overload, the code is compiled to all devices
    /// associated with the \c cl::Context passed.
    cl::Program fetch_or_build_source(std::string_view source,
                                      const cl::Context& context,
                                      std::string_view options = {}) const;

    /// @brief Builds OpenCL source code to a \c cl::Program and stores it in
    /// the cache. If the program existed in the cache previously, loads it back
    /// from the cache without building.
    /// @param source Source code of the program.
    /// @param context The OpenCL context to compile for.
    /// @param devices The OpenCL devices associated with the \c context to
    /// compile for.
    /// @param options Build options that are passed to the OpenCL compiler.
    /// @return The built OpenCL program.
    /// @note The cache lookup considers the passed source code, the passed
    /// options, the platform's version and the devices to which the code is
    /// compiled.
    cl::Program fetch_or_build_source(std::string_view source,
                                      const cl::Context& context,
                                      const std::vector<cl::Device>& devices,
                                      std::string_view options = {}) const;

    /// @brief Builds OpenCL IL code to a \c cl::Program and stores it in the
    /// cache. If the program existed in the cache previously, loads it back
    /// from the cache without building.
    /// @param il IL code of the program.
    /// @param options Build options that are passed to the OpenCL compiler.
    /// @return The built OpenCL program.
    /// @note The cache lookup considers the passed IL code, the passed options,
    /// the platform's version and the devices to which the code is compiled.
    /// In this overload, the code is compiled to all devices associated with
    /// the \c cl::Context passed in the constructor.
    cl::Program fetch_or_build_il(const std::vector<char>& il,
                                  std::string_view options = {}) const;

    /// @brief Builds OpenCL IL code to a \c cl::Program and stores it in the
    /// cache. If the program existed in the cache previously, loads it back
    /// from the cache without building.
    /// @param il IL code of the program.
    /// @param context The OpenCL context to compile for.
    /// @param options Build options that are passed to the OpenCL compiler.
    /// @return The built OpenCL program.
    /// @note The cache lookup considers the passed IL code, the passed options,
    /// the platform's version and the devices to which the code is compiled.
    /// In this overload, the code is compiled to all devices associated with
    /// the \c cl::Context passed.
    cl::Program fetch_or_build_il(const std::vector<char>& il,
                                  const cl::Context& context,
                                  std::string_view options = {}) const;

    /// @brief Builds OpenCL IL code to a \c cl::Program and stores it in the
    /// cache. If the program existed in the cache previously, loads it back
    /// from the cache without building.
    /// @param il IL code of the program.
    /// @param context The OpenCL context to compile for.
    /// @param devices The OpenCL devices associated with the \c context to
    /// compile for.
    /// @param options Build options that are passed to the OpenCL compiler.
    /// @return The built OpenCL program.
    /// @note The cache lookup considers the passed IL code, the passed options,
    /// the platform's version and the devices to which the code is compiled.
    cl::Program fetch_or_build_il(const std::vector<char>& il,
                                  const cl::Context& context,
                                  const std::vector<cl::Device>& devices,
                                  std::string_view options = {}) const;

private:
    template <class T>
    cl::Program fetch_or_build_impl(const T& input,
                                    const cl::Context& context,
                                    const std::vector<cl::Device>& devices,
                                    std::string_view options) const;

    std::filesystem::path
    get_path_for_device_binary(const cl::Device& device,
                               std::string_view key_hash) const;

    std::shared_ptr<const cl::Context> context_;
    std::filesystem::path cache_root_;
};

}

#endif // OCL_PROGRAM_CACHE_LIB_INC_OCL_PROGRAM_CACHE_PROGRAM_CACHE_HPP_
