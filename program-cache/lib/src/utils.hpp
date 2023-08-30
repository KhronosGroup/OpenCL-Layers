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

/// @file utils.hpp
/// @brief Simple utility functions used in the module.
///
/// These utilities are placed in this header file for testing purposes.

#ifndef OCL_PROGRAM_CACHE_LIB_SRC_UTILS_HPP_
#define OCL_PROGRAM_CACHE_LIB_SRC_UTILS_HPP_

#include <common.hpp>

#include <CL/opencl.h>

#include <charconv>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>


namespace ocl::program_cache::utils {

/// @brief Throws an exception, if \c errorcode is not \c CL_SUCCESS
inline void check_cl_error(cl_int errorcode)
{
    if (errorcode != CL_SUCCESS) throw ::ocl::program_cache::opencl_error(errorcode);
}

/// @brief Splits a string at \c delimiter. For supported cases, see the corresponding test.
inline std::vector<std::string_view> split(std::string_view input, char delimiter = ' ')
{
    std::vector<std::string_view> ret;
    std::size_t pos = input.find_first_not_of(delimiter);
    while (pos != std::string_view::npos)
    {
        const auto last_pos = pos;
        pos = input.find(delimiter, pos);
        ret.push_back(input.substr(last_pos, pos - last_pos));
        pos = input.find_first_not_of(delimiter, pos);
    }
    return ret;
}

/// @brief Returns whether \c str starts with \c start
/// @note Remove once on C++20.
inline bool starts_with(std::string_view str, std::string_view start)
{
    return str.find(start) == 0;
}

/// @brief Generic function to wrap \c clGet*Info functions that return a string.
template <class T, class Fun, class Param>
std::string get_info_str(T obj, Fun fun, Param param_name)
{
    std::size_t param_value_size{};
    check_cl_error(fun(obj, param_name, param_value_size, nullptr, &param_value_size));
    std::vector<char> char_data(param_value_size);
    check_cl_error(fun(obj, param_name, param_value_size, char_data.data(), &param_value_size));
    return { char_data.data() };
}

/// @brief Parses \c str to \c int. Throws \c bad_opencl_version_format if unsuccessful.
inline int parse_int(std::string_view str)
{
    int val{};
    if (auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), val);
        ec != std::errc{})
    {
        throw bad_opencl_version_format();
    }
    return val;
}

/// @brief Parses OpenCL version string to major and minor version. Throws \c
/// bad_opencl_version_format if the \c version_string is non-conformant.
inline std::pair<int, int> parse_platform_opencl_version(std::string_view version_string)
{
    // Version format must be
    // OpenCL<space><major_version.minor_version><space><platform-specific-information>
    if (version_string.size() < 10) throw bad_opencl_version_format();
    return { parse_int(version_string.substr(7, 1)), parse_int(version_string.substr(9, 1)) };
}

/// @brief Parses OpenCL C version string to major and minor version. Throws \c
/// bad_opencl_version_format if the \c version_string is non-conformant.
inline std::pair<int, int> parse_device_opencl_c_version(std::string_view version_string)
{
    // Version format must be
    // OpenCL<space>C<space><major_version.minor_version><space><vendor-specific information>
    if (version_string.size() < 12) throw bad_opencl_version_format();
    return { parse_int(version_string.substr(9, 1)), parse_int(version_string.substr(11, 1)) };
}

/// @brief Can be used to wrap a number of functors (e.g. lambdas) to an overload set.
template <class... Args> struct overloads : Args...
{
    using Args::operator()...;
};

template <class... Ts> overloads(Ts...) -> overloads<Ts...>;

template <class T> class hex_format;
template <class T> std::ostream& operator<<(std::ostream& os, const hex_format<T>& h);

/// @brief Formats an unsigned value to a hexadecimal string.
template <class T> class hex_format {
    static_assert(std::is_unsigned_v<T>, "T must be unsigned");

public:
    explicit hex_format(T value): value_(value) {}

private:
    friend std::ostream& operator<< <T>(std::ostream& os, const hex_format& h);
    T value_;
};

template <class T> std::ostream& operator<<(std::ostream& os, const hex_format<T>& h)
{
    // Don't mess up the state of the passed stream
    std::stringstream sstream;
    sstream << std::setfill('0') << std::setw(sizeof(h.value_) * 2) << std::hex << h.value_;
    os << sstream.str();
    return os;
}

} // namespace ocl::program_cache::utils

#endif // OCL_PROGRAM_CACHE_LIB_SRC_UTILS_HPP_