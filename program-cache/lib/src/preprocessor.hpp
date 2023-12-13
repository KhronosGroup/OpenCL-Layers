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

/// @file preprocessor.hpp
/// @brief C preprocessor utilities.
///
/// Utilities that implement a C preprocessor and parsing of OpenCL compilation options.
/// Some internally used functions are exposed to this header for testability.

#ifndef OCL_PROGRAM_CACHE_LIB_SRC_PREPROCESSOR_HPP_
#define OCL_PROGRAM_CACHE_LIB_SRC_PREPROCESSOR_HPP_

#include <common.hpp>

#include <CL/opencl.h>

#include <string_view>
#include <string>
#include <variant>
#include <vector>

namespace ocl::program_cache {

// Boost::Wave crashes on "#pragma\n": https://github.com/boostorg/wave/issues/188
// Function declaration is added to this header for testing purpose
std::string_view remove_empty_pragmas(std::string_view kernel_source, std::string& allocated_str);

struct DefinitionOpt
{
    std::string_view definition_;

    bool operator==(const DefinitionOpt& other) const { return definition_ == other.definition_; }

    bool operator!=(const DefinitionOpt& other) const { return !(*this == other); }
};

struct IncludeOpt
{
    std::string_view path_;

    bool operator==(const IncludeOpt& other) const { return path_ == other.path_; }

    bool operator!=(const IncludeOpt& other) const { return !(*this == other); }
};

struct FastRelaxedMathOpt
{
    bool operator==(const FastRelaxedMathOpt& /*other*/) const { return true; }

    bool operator!=(const FastRelaxedMathOpt& other) const { return !(*this == other); }
};

class LanguageVersion {
public:
    LanguageVersion(int id, bool is_cpp = false): id_(id), is_cpp_(is_cpp) {}

    [[nodiscard]] int id() const { return id_; }
    [[nodiscard]] bool is_cpp() const { return is_cpp_; }

    bool operator==(const LanguageVersion& other) const
    {
        return id_ == other.id_ && is_cpp_ == other.is_cpp_;
    }

    bool operator!=(const LanguageVersion& other) const { return !(*this == other); }

private:
    int id_;
    bool is_cpp_;
};

class LanguageVersionOpt {
public:
    explicit LanguageVersionOpt(std::string_view version_str);

    bool operator==(const LanguageVersionOpt& other) const { return language_ == other.language_; }

    bool operator!=(const LanguageVersionOpt& other) const { return !(*this == other); }

    [[nodiscard]] LanguageVersion get_language() const { return language_; }

private:
    LanguageVersion language_;
};

using Option = std::variant<DefinitionOpt, IncludeOpt, FastRelaxedMathOpt, LanguageVersionOpt>;

std::vector<Option> parse_options(std::string_view options);

std::string preprocess(std::string_view kernel_source,
                       cl_device_id device,
                       std::string_view options,
                       const program_cache_dispatch& dispatch);

} // namespace ocl::program_cache

#endif // OCL_PROGRAM_CACHE_LIB_SRC_PREPROCESSOR_HPP_
