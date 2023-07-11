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

/// @file program_cache.hpp
/// @brief Definition of class \c program_cache.

#ifndef OCL_PROGRAM_CACHE_LIB_INC_OCL_PROGRAM_CACHE_PROGRAM_CACHE_HPP_
#define OCL_PROGRAM_CACHE_LIB_INC_OCL_PROGRAM_CACHE_PROGRAM_CACHE_HPP_

#include "common.hpp"

#include <CL/opencl.h>

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string_view>
#include <string>
#include <vector>

namespace ocl::program_cache {

/// @brief Store and fetch OpenCL program binaries to/from the filesystem.
class program_cache {
public:
    /// @brief Creates a new instance of the \c program_cache.
    /// @param dispatch The function dispatch table that contains the pointers for the OpenCL
    /// runtime functions that are used by the \c program_cache.
    /// @param context OpenCL context to build the programs for. If \c nullptr is passed, then the
    /// default context is used.
    /// @param cache_root Path to the program cache root on the filesystem. If \c nullopt is passed,
    /// the platform dependent default location is used.
    program_cache(const program_cache_dispatch& dispatch,
                  cl_context context = nullptr,
                  const std::optional<std::filesystem::path>& cache_root = std::nullopt);

    /// @brief Loads cached binaries for all devices associated with the \c cl_context passed in the
    /// constructor and builds a \c cl_program.
    /// @param key Key to the cache entries. The key must be equal to the key passed to a previous
    /// \c store call to retrieve the same binaries.
    /// @return The built \c cl_program if a cache entry was found for all devices, \c NULL
    /// otherwise.
    [[nodiscard]] cl_program fetch(std::string_view key) const;

    /// @brief Loads cached binaries for all devices passed and returns a built \c cl_program.
    /// @param key Key to the cache entries. The key must be equal to the key passed to a previous
    /// \c store call to retrieve the same binaries.
    /// @param devices The devices to load the programs for.
    /// @return The built \c cl_program if a cache entry was found for all devices, \c NULL
    /// otherwise.
    [[nodiscard]] cl_program fetch(std::string_view key,
                                   const std::vector<cl_device_id>& devices) const;

    /// @brief Stores the binary representation of a \c cl_program in the cache.
    /// @param program The program to store. It must be built previously, otherwise \c
    /// unbuilt_program_error is thrown.
    /// @param key The key to the cache entry, which can be used to retrieve the cache entry later
    /// via \c fetch.
    void store(cl_program program, std::string_view key) const;

    /// @brief Builds OpenCL source code to a \c cl_program and stores it in the cache. If the
    /// program existed in the cache previously, loads it back from the cache without building.
    /// @param source Source code of the program.
    /// @param options Build options that are passed to the OpenCL compiler.
    /// @return The built OpenCL program.
    /// @note The cache lookup considers the passed source code, the passed options, the platform's
    /// version and the devices to which the code is compiled. In this overload, the code is
    /// compiled to all devices associated with the \c cl_context passed in the constructor.
    [[nodiscard]] cl_program fetch_or_build_source(std::string_view source,
                                                   std::string_view options = {}) const;

    /// @brief Builds OpenCL source code to a \c cl_program and stores it in the cache. If the
    /// program existed in the cache previously, loads it back from the cache without building.
    /// @param source Source code of the program.
    /// @param context The OpenCL context to compile for.
    /// @param options Build options that are passed to the OpenCL compiler.
    /// @return The built OpenCL program.
    /// @note The cache lookup considers the passed source code, the passed options, the platform's
    /// version and the devices to which the code is compiled. In this overload, the code is
    /// compiled to all devices associated with the \c cl_context passed.
    [[nodiscard]] cl_program fetch_or_build_source(std::string_view source,
                                                   cl_context context,
                                                   std::string_view options = {}) const;

    /// @brief Builds OpenCL source code to a \c cl_program and stores it in the cache. If the
    /// program existed in the cache previously, loads it back from the cache without building.
    /// @param source Source code of the program.
    /// @param context The OpenCL context to compile for.
    /// @param devices The OpenCL devices associated with the \c context to compile for.
    /// @param options Build options that are passed to the OpenCL compiler.
    /// @return The built OpenCL program.
    /// @note The cache lookup considers the passed source code, the passed options, the platform's
    /// version and the devices to which the code is compiled.
    [[nodiscard]] cl_program fetch_or_build_source(std::string_view source,
                                                   cl_context context,
                                                   const std::vector<cl_device_id>& devices,
                                                   std::string_view options = {}) const;

    /// @brief Builds OpenCL IL code to a \c cl_program and stores it in the cache. If the program
    /// existed in the cache previously, loads it back from the cache without building.
    /// @param il IL code of the program.
    /// @param options Build options that are passed to the OpenCL compiler.
    /// @return The built OpenCL program.
    /// @note The cache lookup considers the passed IL code, the passed options, the platform's
    /// version and the devices to which the code is compiled. In this overload, the code is
    /// compiled to all devices associated with the \c cl_context passed in the constructor.
    [[nodiscard]] cl_program fetch_or_build_il(const std::vector<char>& il,
                                               std::string_view options = {}) const;

    /// @brief Builds OpenCL IL code to a \c cl_program and stores it in the cache. If the program
    /// existed in the cache previously, loads it back from the cache without building.
    /// @param il IL code of the program.
    /// @param context The OpenCL context to compile for.
    /// @param options Build options that are passed to the OpenCL compiler.
    /// @return The built OpenCL program.
    /// @note The cache lookup considers the passed IL code, the passed options, the platform's
    /// version and the devices to which the code is compiled. In this overload, the code is
    /// compiled to all devices associated with the \c cl_context passed.
    [[nodiscard]] cl_program fetch_or_build_il(const std::vector<char>& il,
                                               cl_context context,
                                               std::string_view options = {}) const;

    /// @brief Builds OpenCL IL code to a \c cl_program and stores it in the cache. If the program
    /// existed in the cache previously, loads it back from the cache without building.
    /// @param il IL code of the program.
    /// @param context The OpenCL context to compile for.
    /// @param devices The OpenCL devices associated with the \c context to compile for.
    /// @param options Build options that are passed to the OpenCL compiler.
    /// @return The built OpenCL program.
    /// @note The cache lookup considers the passed IL code, the passed options, the platform's
    /// version and the devices to which the code is compiled.
    [[nodiscard]] cl_program fetch_or_build_il(const std::vector<char>& il,
                                               cl_context context,
                                               const std::vector<cl_device_id>& devices,
                                               std::string_view options = {}) const;

private:
    template <class T>
    cl_program fetch_or_build_impl(const T& input,
                                   cl_context context,
                                   const std::vector<cl_device_id>& devices,
                                   std::string_view options) const;

    std::filesystem::path get_path_for_device_binary(cl_device_id device,
                                                     std::string_view key_hash) const;

    template <class T>
    std::vector<unsigned char> build_program_to_binary(cl_context context,
                                                       cl_device_id device,
                                                       const T& source,
                                                       std::string_view options) const;

    [[nodiscard]] cl_context get_default_context() const;

    std::vector<cl_device_id> get_devices(cl_context context) const;

    program_cache_dispatch dispatch_;
    cl_context context_;
    std::filesystem::path cache_root_;
};

}

#endif // OCL_PROGRAM_CACHE_LIB_INC_OCL_PROGRAM_CACHE_PROGRAM_CACHE_HPP_
