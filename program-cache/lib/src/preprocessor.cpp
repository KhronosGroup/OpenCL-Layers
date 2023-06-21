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

#include "preprocessor.hpp"

#include "utils.hpp"

// See https://github.com/boostorg/wave/issues/159
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfree-nonheap-object"
#endif
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4706)
#endif

#include <boost/config/warning_disable.hpp>
#include <boost/wave/cpplexer/cpp_lex_iterator.hpp>
#include <boost/wave.hpp>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <array>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace ocl::program_cache {
namespace {
using lex_iterator_t =
    boost::wave::cpplexer::lex_iterator<boost::wave::cpplexer::lex_token<>>;
using context_t =
    boost::wave::context<std::string_view::iterator, lex_iterator_t>;


void undefine_default_macros(context_t& context)
{
    using string_type = context_t::string_type;
    static const std::array<string_type, 8> not_removable_macros{
        "__LINE__", "__FILE__", "__BASE_FILE__",     "__DATE__",
        "__TIME__", "__STDC__", "__INCLUDE_LEVEL__", "__cplusplus",
    };
    std::vector<string_type> macro_names;
    std::copy(context.macro_names_begin(), context.macro_names_end(),
              std::back_inserter(macro_names));
    for (const auto& macro : macro_names)
    {
        if (std::find(not_removable_macros.begin(), not_removable_macros.end(),
                      macro)
            == not_removable_macros.end())
        {
            context.remove_macro_definition(macro, true);
        }
    }
}

int get_device_opencl_c_id(const cl::Device& device)
{
    const auto device_opencl_c_version =
        device.getInfo<CL_DEVICE_OPENCL_C_VERSION>();
    const auto [device_c_major, device_c_minor] =
        utils::parse_device_opencl_c_version(device_opencl_c_version);
    return 100 * device_c_major + 10 * device_c_minor;
}

void add_opencl_macro_defs(const cl::Device& device,
                           context_t& context,
                           LanguageVersion language)
{
    const cl::Platform platform(device.getInfo<CL_DEVICE_PLATFORM>());
    const auto platform_opencl_version =
        platform.getInfo<CL_PLATFORM_VERSION>();
    const auto [platform_major, platform_minor] =
        utils::parse_platform_opencl_version(platform_opencl_version);
    const int platform_id = 100 * platform_major + 10 * platform_minor;
    const int device_c_id = get_device_opencl_c_id(device);

    context.add_macro_definition("__OPENCL_VERSION__="
                                 + std::to_string(platform_id));
    if (device_c_id >= 110)
    {
        context.add_macro_definition("CL_VERSION_1_0=100");
        context.add_macro_definition("CL_VERSION_1_1=110");
    }
    if (device_c_id >= 120)
    {
        context.add_macro_definition("CL_VERSION_1_2=120");
        if (!language.is_cpp())
        {
            context.add_macro_definition("__OPENCL_C_VERSION__="
                                         + std::to_string(language.id()));
        }
    }
    if (device_c_id >= 200)
    {
        context.add_macro_definition("CL_VERSION_2_0=200");
    }
    if (device_c_id >= 300)
    {
        context.add_macro_definition("CL_VERSION_3_0=300");
    }
    if (device.getInfo<CL_DEVICE_ENDIAN_LITTLE>())
    {
        context.add_macro_definition("__ENDIAN_LITTLE__=1");
    }
    if (device.getInfo<CL_DEVICE_IMAGE_SUPPORT>())
    {
        context.add_macro_definition("__IMAGE_SUPPORT__=1");
    }
    if (language.is_cpp())
    {
        context.add_macro_definition("__OPENCL_CPP_VERSION__="
                                     + std::to_string(language.id()));
        context.add_macro_definition("__CL_CPP_VERSION_1_0__=100");
        context.add_macro_definition("__CL_CPP_VERSION_2021__=202100");
    }
}

void process_option(const Option& option,
                    context_t& context,
                    LanguageVersion& language)
{
    std::visit(utils::overloads{
                   [&](const DefinitionOpt& opt) {
                       context.add_macro_definition(
                           std::string(opt.definition_));
                   },
                   [&](const IncludeOpt& opt) {
                       context.add_include_path(std::string(opt.path_).c_str());
                   },
                   [&](const LanguageVersionOpt& opt) {
                       language = opt.get_language();
                   },
                   [&](const FastRelaxedMathOpt&) {
                       context.add_macro_definition("__FAST_RELAXED_MATH__=1");
                   } },
               option);
}

LanguageVersion get_default_language(const cl::Device& device)
{
    return LanguageVersion(std::min(120, get_device_opencl_c_id(device)));
}

} // namespace

LanguageVersionOpt::LanguageVersionOpt(std::string_view version_str)
    : language_(100)
{
    std::string version_str_upper;
    std::transform(version_str.begin(), version_str.end(),
                   std::back_inserter(version_str_upper),
                   [](const auto c) { return std::toupper(c); });
    if (version_str_upper == "CL1.1")
        language_ = LanguageVersion(110, false);
    else if (version_str_upper == "CL1.2")
        language_ = LanguageVersion(120, false);
    else if (version_str_upper == "CL2.0")
        language_ = LanguageVersion(200, false);
    else if (version_str_upper == "CL3.0")
        language_ = LanguageVersion(300, false);
    else if (version_str_upper == "CLC++1.0")
        language_ = LanguageVersion(100, true);
    else if (version_str_upper == "CLC++2021")
        language_ = LanguageVersion(2021, true);
    else
        throw preprocess_exception("Invalid -cl-std specification");
}

std::vector<Option> parse_options(std::string_view options)
{
    std::vector<Option> ret;
    const auto words = utils::split(options);
    for (auto it = words.begin(), end = words.end(); it != end;)
    {
        if (*it == "-D")
        {
            if (it == std::prev(words.end()))
                throw preprocess_exception("Missing option after -D");
            ++it;
            ret.push_back(DefinitionOpt{ *it });
            ++it;
        }
        else if (utils::starts_with(*it, "-D"))
        {
            ret.push_back(DefinitionOpt{ it->substr(2) });
            ++it;
        }
        else if (*it == "-I")
        {
            if (it == std::prev(words.end()))
                throw preprocess_exception("Missing option after -I");
            ++it;
            ret.push_back(IncludeOpt{ *it });
            ++it;
        }
        else if (*it == "-cl-fast-relaxed-math")
        {
            ret.push_back(FastRelaxedMathOpt{});
            ++it;
        }
        else if (utils::starts_with(*it, "-cl-std="))
        {
            ret.push_back(LanguageVersionOpt(it->substr(8)));
            ++it;
        }
        else
        {
            ++it;
        }
    }
    return ret;
}

std::string preprocess(std::string_view kernel_source,
                       const cl::Device& device,
                       std::string_view options)
{
    try
    {
        const auto parsed_options = parse_options(options);
        context_t context(kernel_source.begin(), kernel_source.end());
        undefine_default_macros(context);
        context.set_sysinclude_delimiter();
        LanguageVersion language = get_default_language(device);
        for (const auto& option : parsed_options)
        {
            process_option(option, context, language);
        }
        add_opencl_macro_defs(device, context, language);

        std::stringstream preprocessed;
        for (auto it = context.begin(), end = context.end(); it != end;)
        {
            try
            {
                preprocessed << (it++)->get_value();
            } catch (const boost::wave::preprocess_exception& ex)
            {
                if (!ex.is_recoverable())
                {
                    throw;
                }
            }
        }
        return preprocessed.str();
    } catch (boost::wave::preprocess_exception& ex)
    {
        throw preprocess_exception(
            boost::wave::preprocess_exception::error_text(ex.get_errorcode()));
    }
}

} // namespace ocl::program_cache
