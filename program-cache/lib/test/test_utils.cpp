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

/// @file test_utils.cpp
/// @brief Testing utils in utils.hpp

#include "../src/utils.hpp"

#include <common.hpp>

#include <gtest/gtest.h>

#include <sstream>
#include <vector>

using namespace ocl::program_cache::utils;
using ocl::program_cache::bad_opencl_version_format;

TEST(UtilsTest, Split)
{
    EXPECT_EQ(std::vector<std::string_view>{}, split(""));
    EXPECT_EQ(std::vector<std::string_view>{}, split("   "));
    EXPECT_EQ(std::vector<std::string_view>{ "abc" }, split("abc"));
    EXPECT_EQ((std::vector<std::string_view>{ "a", "bb", "ccc", "d" }), split("a  bb ccc d"));
    EXPECT_EQ((std::vector<std::string_view>{ "a", "bb", "ccc", "d" }), split("   a  bb ccc d  "));
}

TEST(UtilsTest, StartWith)
{
    EXPECT_TRUE(starts_with("", ""));
    EXPECT_TRUE(starts_with("test", "test"));
    EXPECT_TRUE(starts_with("tester", "test"));
    EXPECT_FALSE(starts_with("test", "tester"));
    EXPECT_FALSE(starts_with("test   ", "tester"));
    EXPECT_FALSE(starts_with("", "tester"));
}

TEST(UtilsTest, ParsePlatformOpenCLVersion)
{
    EXPECT_EQ(std::make_pair(1, 2), parse_platform_opencl_version("OpenCL 1.2 Platform"));
    EXPECT_EQ(std::make_pair(3, 0), parse_platform_opencl_version("OpenCL 3.0"));
    EXPECT_THROW([] { parse_platform_opencl_version(""); }(), bad_opencl_version_format);
    EXPECT_THROW([] { parse_platform_opencl_version("2.1"); }(), bad_opencl_version_format);
    EXPECT_THROW([] { parse_platform_opencl_version("OpenCL 2.X"); }(), bad_opencl_version_format);
}

TEST(UtilsTest, ParseDeviceOpenCLCVersion)
{
    EXPECT_EQ(std::make_pair(1, 2), parse_device_opencl_c_version("OpenCL C 1.2 Platform"));
    EXPECT_EQ(std::make_pair(3, 0), parse_device_opencl_c_version("OpenCL C 3.0"));
    EXPECT_THROW([] { parse_device_opencl_c_version(""); }(), bad_opencl_version_format);
    EXPECT_THROW([] { parse_device_opencl_c_version("2.1"); }(), bad_opencl_version_format);
    EXPECT_THROW([] { parse_device_opencl_c_version("OpenCL C 2.X"); }(),
                 bad_opencl_version_format);
}

TEST(UtilsTest, HexFormat)
{
    const auto check = [](auto x, std::string_view expected) {
        std::stringstream sstream;
        sstream << hex_format(x);
        EXPECT_EQ(expected, sstream.str());
    };
    check(0u, "00000000");
    check(1555556u, "0017bc64");
    check(0ull, "0000000000000000");
    check(884121569ull, "0000000034b29fe1");
}
