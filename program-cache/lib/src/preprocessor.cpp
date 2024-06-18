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

/// @file preprocessor.cpp
/// @brief Implementation of C preprocessor utilities.
///
/// Boost::Wave is used to implement a C preprocessor which is conformant with the OpenCL standard.

#include "preprocessor.hpp"

#include "utils.hpp"

// See https://github.com/boostorg/wave/issues/159
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfree-nonheap-object"
#endif
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100) // \W4 - unreferenced formal parameter
#pragma warning(disable : 4706) // \W4 - assignment within conditional expression
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
#include <string_view>
#include <string>
#include <variant>
#include <vector>

namespace {

namespace pc = ocl::program_cache;
using lex_iterator_t = boost::wave::cpplexer::lex_iterator<boost::wave::cpplexer::lex_token<>>;
using context_t = boost::wave::context<std::string_view::iterator, lex_iterator_t>;

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
        if (std::find(not_removable_macros.begin(), not_removable_macros.end(), macro)
            == not_removable_macros.end())
        {
            context.remove_macro_definition(macro, true);
        }
    }
}

int get_device_opencl_c_id(cl_device_id device, const pc::program_cache_dispatch& dispatch)
{
    const auto device_opencl_c_version =
        pc::utils::get_info_str(device, dispatch.clGetDeviceInfo, CL_DEVICE_OPENCL_C_VERSION);
    const auto [device_c_major, device_c_minor] =
        pc::utils::parse_device_opencl_c_version(device_opencl_c_version);
    return 100 * device_c_major + 10 * device_c_minor;
}

void add_opencl_macro_defs(cl_device_id device,
                           context_t& context,
                           pc::LanguageVersion language,
                           const pc::program_cache_dispatch& dispatch)
{
    cl_platform_id platform{};
    pc::utils::check_cl_error(
        dispatch.clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr));
    const auto platform_opencl_version =
        pc::utils::get_info_str(platform, dispatch.clGetPlatformInfo, CL_PLATFORM_VERSION);
    const auto [platform_major, platform_minor] =
        pc::utils::parse_platform_opencl_version(platform_opencl_version);
    const int platform_id = 100 * platform_major + 10 * platform_minor;
    const int device_c_id = get_device_opencl_c_id(device, dispatch);

    context.add_macro_definition("__kernel_exec(X, typen)=__kernel "
                                 "__attribute__((work_group_size_hint(X, 1, "
                                 "1))) __attribute__((vec_type_hint(typen)))");
    context.add_macro_definition(
        "kernel_exec(X, typen)=__kernel __attribute__((work_group_size_hint(X, "
        "1, 1))) __attribute__((vec_type_hint(typen)))");
    context.add_macro_definition("__OPENCL_VERSION__=" + std::to_string(device_c_id));
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
            context.add_macro_definition("__OPENCL_C_VERSION__=" + std::to_string(language.id()));
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
    cl_bool endian_little{};
    pc::utils::check_cl_error(dispatch.clGetDeviceInfo(
        device, CL_DEVICE_ENDIAN_LITTLE, sizeof(endian_little), &endian_little, nullptr));
    if (endian_little)
    {
        context.add_macro_definition("__ENDIAN_LITTLE__=1");
    }
    cl_bool image_support{};
    pc::utils::check_cl_error(dispatch.clGetDeviceInfo(
        device, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, nullptr));
    if (image_support)
    {
        context.add_macro_definition("__IMAGE_SUPPORT__=1");
    }
    if (language.is_cpp())
    {
        context.add_macro_definition("__OPENCL_CPP_VERSION__=" + std::to_string(language.id()));
        context.add_macro_definition("__CL_CPP_VERSION_1_0__=100");
        context.add_macro_definition("__CL_CPP_VERSION_2021__=202100");
    }
    const auto extension_list =
        pc::utils::get_info_str(device, dispatch.clGetDeviceInfo, CL_DEVICE_EXTENSIONS);
    for (const auto& extension_name : pc::utils::split(extension_list))
    {
        context.add_macro_definition(std::string(extension_name.begin(), extension_name.end())
                                     + "=1");
    }
    if (platform_id >= 300)
    {
        std::size_t feature_bytes{};
        pc::utils::check_cl_error(dispatch.clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_FEATURES, 0,
                                                           nullptr, &feature_bytes));
        std::vector<cl_name_version> features(feature_bytes / sizeof(cl_name_version));
        pc::utils::check_cl_error(dispatch.clGetDeviceInfo(
            device, CL_DEVICE_OPENCL_C_FEATURES, feature_bytes, features.data(), nullptr));
        for (const auto& feature : features)
        {
            context.add_macro_definition(std::string(&feature.name[0]) + "=1");
        }
    }
}

void process_option(const pc::Option& option, context_t& context, pc::LanguageVersion& language)
{
    using namespace pc;
    std::visit(
        utils::overloads{ [&](const DefinitionOpt& opt) {
                             context.add_macro_definition(std::string(opt.definition_));
                         },
                          [&](const IncludeOpt& opt) {
                              const std::string path(opt.path_);
                              context.add_include_path(path.c_str());
                              context.add_sysinclude_path(path.c_str());
                          },
                          [&](const LanguageVersionOpt& opt) { language = opt.get_language(); },
                          [&](const FastRelaxedMathOpt&) {
                              context.add_macro_definition("__FAST_RELAXED_MATH__=1");
                          } },
        option);
}

pc::LanguageVersion get_default_language(cl_device_id device,
                                         const pc::program_cache_dispatch& dispatch)
{
    return { std::min(120, get_device_opencl_c_id(device, dispatch)) };
}

} // namespace

std::string_view pc::remove_empty_pragmas(std::string_view kernel_source,
                                          std::string& allocated_str)
{
    auto pragma_newline_idx = kernel_source.find("#pragma\n");
    const auto last_pragma_idx = kernel_source.rfind("#pragma");
    const bool ends_with_pragma =
        last_pragma_idx == (kernel_source.size() - std::string_view("#pragma").size());
    if (pragma_newline_idx == std::string_view::npos)
    {
        if (ends_with_pragma)
        {
            return kernel_source.substr(0, last_pragma_idx);
        }
        else
        {
            return kernel_source;
        }
    }
    // At this point, #pragma\n is somewhere in the middle of the string
    // To remove them, we must make a copy of the entire source.
    if (ends_with_pragma)
    {
        allocated_str = std::string(kernel_source.begin(), kernel_source.begin() + last_pragma_idx);
    }
    else
    {
        allocated_str = std::string(kernel_source.begin(), kernel_source.end());
    }
    while (pragma_newline_idx != std::string::npos)
    {
        allocated_str.erase(pragma_newline_idx, std::string_view("#pragma\n").size());
        pragma_newline_idx = allocated_str.find("#pragma\n");
    }
    return allocated_str;
}

pc::LanguageVersionOpt::LanguageVersionOpt(std::string_view version_str): language_(100)
{
    std::string version_str_upper;
    std::transform(version_str.begin(), version_str.end(), std::back_inserter(version_str_upper),
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

std::vector<pc::Option> pc::parse_options(std::string_view options)
{
    std::vector<Option> ret;
    const auto words = utils::split(options);
    for (auto it = words.begin(), end = words.end(); it != end;)
    {
        if (*it == "-D")
        {
            if (it == std::prev(words.end())) throw preprocess_exception("Missing option after -D");
            ++it;
            ret.emplace_back(DefinitionOpt{ *it });
            ++it;
        }
        else if (utils::starts_with(*it, "-D"))
        {
            ret.emplace_back(DefinitionOpt{ it->substr(2) });
            ++it;
        }
        else if (*it == "-I")
        {
            if (it == std::prev(words.end())) throw preprocess_exception("Missing option after -I");
            ++it;
            ret.emplace_back(IncludeOpt{ *it });
            ++it;
        }
        else if (*it == "-cl-fast-relaxed-math")
        {
            ret.emplace_back(FastRelaxedMathOpt{});
            ++it;
        }
        else if (const std::string_view std_opt("-cl-std="); utils::starts_with(*it, std_opt))
        {
            ret.emplace_back(LanguageVersionOpt(it->substr(std_opt.size())));
            ++it;
        }
        else
        {
            ++it;
        }
    }
    return ret;
}

std::string pc::preprocess(std::string_view kernel_source,
                           cl_device_id device,
                           std::string_view options,
                           const program_cache_dispatch& dispatch)
{
    try
    {
        const auto parsed_options = parse_options(options);
        std::string alloc_str;
        kernel_source = remove_empty_pragmas(kernel_source, alloc_str);
        context_t context(kernel_source.begin(), kernel_source.end());
        undefine_default_macros(context);
        LanguageVersion language = get_default_language(device, dispatch);
        for (const auto& option : parsed_options)
        {
            process_option(option, context, language);
        }
        add_opencl_macro_defs(device, context, language, dispatch);

        std::stringstream preprocessed;
        for (auto it = context.begin(), end = context.end(); it != end;)
        {
            try
            {
                preprocessed << (it++)->get_value();
            } catch (const boost::wave::preprocess_exception& ex)
            {
                if (!ex.is_recoverable()
                    || ex.get_errorcode() == boost::wave::preprocess_exception::error_directive)
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
