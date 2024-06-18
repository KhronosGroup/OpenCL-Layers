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

/// @file test_preprocessor.cpp
/// @brief Testing utilities in preprocessor.hpp

#include "utils_test.hpp"

#include "../src/preprocessor.hpp"
#include "../src/utils.hpp"

#include <program_cache.hpp>

#include <CL/opencl.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace ocl::program_cache;

TEST(PreprocessorTest, RemoveEmptyPragmas)
{
    std::string alloc_str;
    EXPECT_EQ("kernel void foo(){}\n", remove_empty_pragmas("kernel void foo(){}\n", alloc_str));
    EXPECT_EQ("kernel void foo(){}\n",
              remove_empty_pragmas("kernel void foo(){}\n#pragma\n", alloc_str));
    EXPECT_EQ("kernel void foo(){}\n",
              remove_empty_pragmas("kernel void foo(){}\n#pragma", alloc_str));
    EXPECT_EQ("kernel void foo(){}\n  ",
              remove_empty_pragmas("kernel void foo(){}\n  #pragma\n", alloc_str));
    EXPECT_EQ("kernel void foo(){}\n  ",
              remove_empty_pragmas("kernel void foo(){}\n  #pragma", alloc_str));
    EXPECT_EQ("kernel void foo(){\n#pragma bar\n}",
              remove_empty_pragmas("kernel void foo(){\n#pragma bar\n}", alloc_str));
    EXPECT_EQ(
        "kernel void foo(){\n#pragma bar\n}\n",
        remove_empty_pragmas("kernel void foo(){\n#pragma\n#pragma bar\n#pragma\n}\n", alloc_str));
    EXPECT_EQ("kernel void foo(){\n#pragma bar\n}\n",
              remove_empty_pragmas("kernel void foo(){\n#pragma\n#pragma bar\n#pragma\n}\n#pragma",
                                   alloc_str));
}

TEST(PreprocessorTest, ParseOptions)
{
    ASSERT_EQ(std::vector<Option>{}, parse_options(""));
    ASSERT_EQ(std::vector<Option>{}, parse_options(" -cl-single-precision-constant "));
    ASSERT_EQ((std::vector<Option>{ LanguageVersionOpt("CL2.0"), DefinitionOpt{ "TEST=100" } }),
              parse_options(" -cl-single-precision-constant -cl-std=CL2.0  -DTEST=100 "));
    ASSERT_EQ((std::vector<Option>{ DefinitionOpt{ "TEST2" }, FastRelaxedMathOpt{},
                                    IncludeOpt{ "/usr/include" }, DefinitionOpt{ "TEST=100" } }),
              parse_options(" -D TEST2 -whatever -cl-fast-relaxed-math  -I /usr/include "
                            "-cl-single-precision-constant -I/usr/bin  -DTEST=100 "));
    ASSERT_THROW([] { parse_options("-cl-single-precision-constant -D "); }(),
                 preprocess_exception);
    ASSERT_THROW([] { parse_options("-cl-single-precision-constant -I"); }(), preprocess_exception);
}

TEST(PreprocessorTest, PreprocessProgram)
{
    const std::string kernel_source =
        R"(// Comment
#pragma OPENCL EXTENSION all : behavior
#include <header.clh>
#if defined(FOO)
kernel void foo(global int* i)
{
    *i = 100;
}
#else
kernel void foo(global int* i)
{
    *i = 101;
}
#endif
)";

    std::string random_name;
    std::uniform_int_distribution dist(int('a'), int('z') + 1);
    std::default_random_engine engine{ std::random_device{}() };
    std::generate_n(std::back_inserter(random_name), 8, [&] { return char(dist(engine)); });
    const auto include_dir = std::filesystem::temp_directory_path() / random_name;
    std::filesystem::create_directories(include_dir);
    const auto include_file = include_dir / "header.clh";
    const std::string include_source = R"(#pragma once
void bar() {};
)";
    {
        std::ofstream ofs(include_file);
        ASSERT_TRUE(ofs.good());
        ofs << include_source;
    }

    const auto preprocessed_source = ocl::program_cache::preprocess(
        kernel_source, cl::Device::getDefault()(), "-D FOO -I " + include_dir.string(),
        get_default_program_cache_dispatch());
    ASSERT_NE(std::string::npos, preprocessed_source.find("kernel void foo(global int* i)"));
    ASSERT_NE(std::string::npos, preprocessed_source.find("*i = 100;"));
    ASSERT_EQ(std::string::npos, preprocessed_source.find("*i = 101;"));
    ASSERT_EQ(std::string::npos, preprocessed_source.find("Comment"));
    ASSERT_EQ(std::string::npos, preprocessed_source.find("#include"));
    ASSERT_NE(std::string::npos, preprocessed_source.find("void bar() {};"));
    ASSERT_NE(std::string::npos,
              preprocessed_source.find("#pragma OPENCL EXTENSION all : behavior"));
}
