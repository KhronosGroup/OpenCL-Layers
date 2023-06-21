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

#include "../src/utils.hpp"

#include <gtest/gtest.h>

using namespace ocl::program_cache::utils;

TEST(UtilsTest, Split)
{
    EXPECT_EQ(std::vector<std::string_view>{}, split(""));
    EXPECT_EQ(std::vector<std::string_view>{}, split("   "));
    EXPECT_EQ(std::vector<std::string_view>{ "abc" }, split("abc"));
    EXPECT_EQ((std::vector<std::string_view>{ "a", "bb", "ccc", "d" }),
              split("a  bb ccc d"));
    EXPECT_EQ((std::vector<std::string_view>{ "a", "bb", "ccc", "d" }),
              split("   a  bb ccc d  "));
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
    EXPECT_EQ(std::make_pair(1, 2),
              parse_platform_opencl_version("OpenCL 1.2 Platform"));
    EXPECT_EQ(std::make_pair(3, 0),
              parse_platform_opencl_version("OpenCL 3.0"));
    EXPECT_THROW([] { parse_platform_opencl_version(""); }(),
                 bad_opencl_version_format);
    EXPECT_THROW([] { parse_platform_opencl_version("2.1"); }(),
                 bad_opencl_version_format);
    EXPECT_THROW([] { parse_platform_opencl_version("OpenCL 2.X"); }(),
                 bad_opencl_version_format);
}

TEST(UtilsTest, ParseDeviceOpenCLCVersion)
{
    EXPECT_EQ(std::make_pair(1, 2),
              parse_device_opencl_c_version("OpenCL C 1.2 Platform"));
    EXPECT_EQ(std::make_pair(3, 0),
              parse_device_opencl_c_version("OpenCL C 3.0"));
    EXPECT_THROW([] { parse_device_opencl_c_version(""); }(),
                 bad_opencl_version_format);
    EXPECT_THROW([] { parse_device_opencl_c_version("2.1"); }(),
                 bad_opencl_version_format);
    EXPECT_THROW([] { parse_device_opencl_c_version("OpenCL C 2.X"); }(),
                 bad_opencl_version_format);
}
