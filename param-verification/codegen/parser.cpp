#include <string.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <regex>
#include <map>
#include <array>
#include "rapidxml.hpp"

using namespace rapidxml;

bool generate_get_version;
std::map<std::string, std::string> func_params;

// prototypes
std::string parse_expression(xml_node<> const* const node);
std::array<std::string, 2> parse_2expressions(xml_node<> const* const node);
std::vector<std::string> parse_list_expressions(xml_node<> const* const node);
std::string parse_violation(xml_node<> const* const violation);
std::vector<std::string> parse_list(xml_node<> const* const node);

// realizations

// Render a call to 'from' given a version string like `1.2`.
std::string render_from(const char* const version_str, bool call_get_version)
{
    std::stringstream ss;
    ss << version_str;
    std::string major;
    std::string minor;
    std::getline(ss, major, '.');
    std::getline(ss, minor);
    return std::string(call_get_version ? "(get_version()" : "(version") + " >= CL_MAKE_VERSION("
        + major + ", " + minor + ", 0))";
}

std::string parse_expression(xml_node<> const* const node)
{
    std::string res;
    char const* const name = node->name();

    if (strcmp(name, "name") == 0)
    {
        res = node->value();
    }
    else if (strcmp(name, "literal") == 0)
    {
        res = node->value();
    }
    else if (strcmp(name, "literal_list") == 0)
    {
        res = node->value();
        res = "literal_list(get_version(), \"" + func_params[res] + "\", " + res + ")";
        generate_get_version = true;
    }
    else if (strcmp(name, "sizeof") == 0)
    {
        res = "sizeof(" + std::string(node->value()) + ")";
    }
    else if (strcmp(name, "mod") == 0)
    {
        auto list = parse_2expressions(node);

        res = "(" + list[0] + " % " + list[1] + ")";
    }
    else if (strcmp(name, "mult") == 0)
    {
        std::vector<std::string> list(parse_list_expressions(node));
        int n = 0;
        for (auto a : list)
        {
            res += (n == 0) ? "(" : " * ";
            res += a;
            ++n;
        }
        res += ")";
    }
    else if (strcmp(name, "add") == 0)
    {
        std::vector<std::string> list(parse_list_expressions(node));

        int n = 0;
        for (auto a : list)
        {
            res += (n == 0) ? "(" : " + ";
            res += a;
            ++n;
        }
        res += ")";
    }
    else if (strcmp(name, "max") == 0)
    {
        auto list = parse_2expressions(node);

        res = "std::max(" + list[0] + ", " + list[1] + ")";
    }
    else if (strcmp(name, "query") == 0)
    {
        res = "query<" + std::string(node->first_attribute("property")->value()) + ">("
            + node->first_attribute("object")->value() + ")";
    }

    return res;
}

std::array<std::string, 2> parse_2expressions(xml_node<> const* const node)
{
    std::array<std::string, 2> res;

    res[0] = parse_expression(node->first_node());
    res[1] = parse_expression(node->first_node()->next_sibling());

    return res;
}

std::vector<std::string> parse_list_expressions(xml_node<> const* const node)
{
    std::vector<std::string> res;

    for (xml_node<> const* list_node = node->first_node(); list_node != nullptr;
         list_node = list_node->next_sibling())
    {
        res.push_back(parse_expression(list_node));
    }

    return res;
}

std::vector<std::string> parse_list(xml_node<> const* const node)
{
    std::vector<std::string> res;

    for (xml_node<> const* list_node = node->first_node(); list_node != nullptr;
         list_node = list_node->next_sibling())
    {
        res.push_back(parse_violation(list_node));
        // res.push_back(std::string(list_node->value()));
    }

    return res;
}

std::string parse_violation(xml_node<> const* const violation)
{
    std::string test;

    if (violation != nullptr)
    {
        char const* const name = violation->name();

        if (strcmp(name, "or") == 0)
        {
            // printf("or:");
            std::vector<std::string> list(parse_list(violation));

            int n = 0;
            for (auto a : list)
            {
                test += (n == 0) ? "(" : " ||\n    ";
                test += a;
                ++n;
                // printf("%s ", a.c_str());
            }
            test += ")";
            // printf("\n");
        }
        else if (strcmp(name, "and") == 0)
        {
            // printf("and:");
            std::vector<std::string> list(parse_list(violation));

            int n = 0;
            for (auto a : list)
            {
                test += (n == 0) ? "(" : " &&\n    ";
                test += a;
                ++n;
                // printf("%s ", a.c_str());
            }
            test += ")";
            // printf("\n");
        }
        else if (strcmp(name, "mutex_violation") == 0)
        {
            std::vector<std::string> list(parse_list(violation));

            int n = 0;
            for (auto a : list)
            {
                test += (n == 0) ? "((bool)" : " +\n    (bool)";
                test += a;
                ++n;
            }
            test += " > 1)";
        }
        else if (strcmp(name, "not") == 0)
        {
            test = "(!(" + parse_violation(violation->first_node()) + "))";
        }
        else if (strcmp(name, "eq") == 0)
        {
            auto list = parse_2expressions(violation);

            test = "(" + list[0] + " == " + list[1] + ")";
        }
        else if (strcmp(name, "neq") == 0)
        {
            auto list = parse_2expressions(violation);

            test = "(" + list[0] + " != " + list[1] + ")";
        }
        else if (strcmp(name, "ls") == 0)
        {
            auto list = parse_2expressions(violation);

            test = "(" + list[0] + " < " + list[1] + ")";
        }
        else if (strcmp(name, "gt") == 0)
        {
            auto list = parse_2expressions(violation);

            test = "(" + list[0] + " > " + list[1] + ")";
        }
        else if (strcmp(name, "bit_and") == 0)
        {
            auto list = parse_2expressions(violation);

            test = "(" + list[0] + " & " + list[1] + ")";
        }
        else if (strcmp(name, "array_len_ls") == 0)
        {
            auto list = parse_2expressions(violation);

            test = "(array_len_ls(" + list[0] + ", " + list[1] + "))";
        }
        else if (strcmp(name, "enum_violation") == 0)
        {
            std::string tmp = violation->first_attribute("name")->value();

            test = "(enum_violation(get_version(), \"" + func_params[tmp] + "\", " + tmp + "))";
            generate_get_version = true;
        }
        else if (strcmp(name, "bitfield_violation") == 0)
        {
            std::string tmp = violation->first_attribute("name")->value();

            test = "(bitfield_violation(get_version(), \"" + func_params[tmp] + "\", " + tmp + "))";
            generate_get_version = true;
        }
        else if (strcmp(name, "list_violation") == 0)
        {
            std::string tmp = violation->first_attribute("name")->value();

            if (violation->first_attribute("param"))
                test = "(list_violation(get_version(), \"" + func_params[tmp] + "\", " + tmp + ", "
                    + violation->first_attribute("param")->value() + "))";
            else
                test = "(list_violation(get_version(), \"" + func_params[tmp] + "\", " + tmp + "))";
            generate_get_version = true;
        }
        else if (strcmp(name, "struct_violation") == 0)
        {
            std::string tmp = violation->first_attribute("name")->value();

            if (violation->first_attribute("param"))
                test = "(struct_violation(get_version(), " + tmp + ", "
                    + violation->first_attribute("param")->value() + "))";
            else
                test = "(struct_violation(get_version(), " + tmp + "))";
            generate_get_version = true;
        }
        else if (strcmp(name, "not_aligned") == 0)
        {
            test = "(not_aligned(" + std::string(violation->first_attribute("pointer")->value())
                + ", " + violation->first_attribute("align")->value() + "))";
        }
        else if (strcmp(name, "object_is_invalid") == 0)
        {
            std::string tmp = violation->first_attribute("name")->value();

            if (violation->first_attribute("type"))
                test = "(!object_is_valid(" + tmp + ", "
                    + violation->first_attribute("type")->value() + "))";
            else
                test = "(!object_is_valid(" + tmp + "))";
        }
        else if (strcmp(name, "any_zero") == 0)
        {
            test = "(any_zero(" + std::string(violation->first_attribute("array")->value()) + ", "
                + std::string(violation->first_attribute("elements")->value()) + "))";
        }
        else if (strcmp(name, "any_nullptr") == 0)
        {
            test = "(any_nullptr(" + std::string(violation->first_attribute("array")->value())
                + ", " + std::string(violation->first_attribute("elements")->value()) + "))";
        }
        else if (strcmp(name, "any_invalid") == 0)
        {
            test = "(any_invalid(" + std::string(violation->first_attribute("array")->value())
                + ", " + std::string(violation->first_attribute("elements")->value()) + "))";
        }
        else if (strcmp(name, "any_non_null_invalid") == 0)
        {
            test = "(any_non_null_invalid("
                + std::string(violation->first_attribute("array")->value()) + ", "
                + std::string(violation->first_attribute("elements")->value()) + "))";
        }
        else if (strcmp(name, "any_not_available") == 0)
        {
            test = "(any_not_available(" + std::string(violation->first_attribute("array")->value())
                + ", " + std::string(violation->first_attribute("elements")->value()) + "))";
        }
        else if (strcmp(name, "object_not_in") == 0)
        {
            test = "(object_not_in(" + std::string(violation->first_attribute("object")->value())
                + ", " + std::string(violation->first_attribute("in")->value()) + "))";
        }
        else if (strcmp(name, "any_object_not_in") == 0)
        {
            test = "(any_object_not_in(" + std::string(violation->first_attribute("array")->value())
                + ", " + std::string(violation->first_attribute("elements")->value()) + ", "
                + std::string(violation->first_attribute("in")->value()) + "))";
        }
        else if ((strcmp(name, "for_all") == 0) || (strcmp(name, "for_any") == 0))
        {
            if (violation->first_attribute("in") != nullptr)
                test = std::string("(") + name + "<" + violation->first_attribute("query")->value()
                    + ">(" + violation->first_attribute("in")->value() + ",\n    [=](return_type<"
                    + violation->first_attribute("query")->value()
                    + "> query){\n      return static_cast<bool>("
                    + parse_violation(violation->first_node()) + "); }))";
            else
                test = std::string("(") + name + "<" + violation->first_attribute("query")->value()
                    + ">(" + violation->first_attribute("array")->value() + ", "
                    + violation->first_attribute("elements")->value() + ",\n    [=](return_type<"
                    + violation->first_attribute("query")->value()
                    + "> query){\n      return static_cast<bool>("
                    + parse_violation(violation->first_node()) + "); }))";
        }
        else if (strcmp(name, "check_copy_overlap") == 0)
        {
            test = "(check_copy_overlap("
                + std::string(violation->first_attribute("param")->value()) + "))";
        }
        else if (strcmp(name, "from") == 0)
        {
            test = render_from(violation->first_attribute("version")->value(), true);
            generate_get_version = true;
        }
        /*        else if (strcmp(name, "name") == 0)
                {
                    test = std::string(node->value());
                }
                else if (strcmp(name, "literal") == 0)
                {
                    test = std::string(node->value());
                }*/
    }

    return test;
}

void parse_enums(std::stringstream& code, xml_node<>*& root_node)
{
    ///////////////////////////////////////////////////////////////////////
    // enums
    ///////////////////////////////////////////////////////////////////////

    code << "template<typename T>\n"
         << "bool enum_violation(cl_version version, const char * name, T param)\n"
         << "{\n";

    // Iterate over the versions
    for (xml_node<>* version_node = root_node->first_node("feature"); version_node != nullptr;
         version_node = version_node->next_sibling("feature"))
    {
        code << "  if " << render_from(version_node->first_attribute("number")->value(), false)
             << " {\n";

        std::vector<std::string> enums_list = { "cl_platform_info",
                                                "cl_device_info",
                                                "cl_context_info",
                                                "cl_command_queue_info",
                                                "cl_buffer_create_type",
                                                "cl_image_info",
                                                "cl_mem_info",
                                                "cl_addressing_mode",
                                                "cl_filter_mode",
                                                "cl_sampler_info",
                                                "cl_program_info",
                                                "cl_program_build_info",
                                                "cl_kernel_exec_info",
                                                "cl_kernel_info",
                                                "cl_kernel_work_group_info",
                                                "cl_kernel_sub_group_info",
                                                "cl_kernel_arg_info",
                                                "cl_event_info",
                                                "cl_profiling_info",
                                                "cl_channel_order",
                                                "cl_channel_type" };

        for (auto a : enums_list)
        {
            for (xml_node<>* enum_node = version_node->first_node("require"); enum_node != nullptr;
                 enum_node = enum_node->next_sibling("require"))
            {
                if (enum_node->first_attribute("comment") != nullptr)
                {
                    // printf("I have visited %s.\n",
                    //     enum_node->first_attribute("comment")->value());
                    if (strstr(enum_node->first_attribute("comment")->value(), a.c_str())
                        != nullptr)
                    {
                        code << "    if (strcmp(name, \"" << a << "\") == 0)\n"
                             << "      switch (param) {\n";

                        for (xml_node<>* enum_val = enum_node->first_node("enum"); enum_val;
                             enum_val = enum_val->next_sibling("enum"))
                        {
                            code << "        case " << enum_val->first_attribute("name")->value()
                                 << ":\n";
                        }

                        code << "          return false;\n"
                             << "      }\n\n";
                    }
                }
            }
        }

        code << "  }\n\n";
        // printf("I have visited %s.\n",
        //     version_node->first_attribute("number")->value());
        //     version_node->value());
    }

    code << "  return true;\n"
         << "}\n\n";
    // printf("\n");
}

void parse_bitfields(std::stringstream& code, xml_node<>*& root_node)
{
    ///////////////////////////////////////////////////////////////////////
    // bitfields
    ///////////////////////////////////////////////////////////////////////

    code << "// function checks if there are set bits in the bitfield outside of defined\n"
         << "// 0 is then always valid param\n"
         << "template<typename T>\n"
         << "bool bitfield_violation(cl_version version, const char * name, T param)\n"
         << "{\n"
         << "  T mask = 0;\n\n";

    // Iterate over the versions
    for (xml_node<>* version_node = root_node->first_node("feature"); version_node != nullptr;
         version_node = version_node->next_sibling("feature"))
    {
        code << "  if " << render_from(version_node->first_attribute("number")->value(), false)
             << " {\n";

        std::vector<std::string> bitfields_list = { "cl_device_type",
                                                    "cl_command_queue_properties",
                                                    "cl_mem_flags",
                                                    "cl_map_flags",
                                                    "cl_mem_migration_flags",
                                                    "cl_svm_mem_flags",
                                                    "cl_device_affinity_domain",
                                                    "cl_command_queue_properties",
                                                    "cl_device_fp_config",
                                                    "cl_device_exec_capabilities" };

        for (auto i : bitfields_list)
        {
            for (xml_node<>* bitfield_node = version_node->first_node("require");
                 bitfield_node != nullptr; bitfield_node = bitfield_node->next_sibling("require"))
            {
                if (bitfield_node->first_attribute("comment") != nullptr)
                {
                    // printf("I have visited %s.\n",
                    //     bitfield_node->first_attribute("comment")->value());
                    if (strstr(bitfield_node->first_attribute("comment")->value(), i.c_str())
                        != nullptr)
                    {
                        code << "    if (strcmp(name, \"" << i << "\") == 0) {\n";

                        for (xml_node<>* bitfield_val = bitfield_node->first_node("enum");
                             bitfield_val != nullptr;
                             bitfield_val = bitfield_val->next_sibling("enum"))
                        {
                            code << "        mask |= "
                                 << bitfield_val->first_attribute("name")->value() << ";\n";
                        }

                        code << "      }\n\n";
                    }
                }
            }
        }

        code << "  }\n\n";
        // printf("I have visited %s.\n",
        //     version_node->first_attribute("number")->value());
        //     version_node->value());
    }

    code << "  return (param & ~mask);\n"
         << "}\n\n";
    // printf("\n");
}

void parse_literal_lists(std::stringstream& code, xml_node<>*& root_node)
{
    ///////////////////////////////////////////////////////////////////////
    // literal lists
    ///////////////////////////////////////////////////////////////////////

    code << "template<typename T>\n"
         << "size_t literal_list(cl_version version, const char * name, T param)\n"
         << "{\n";

    // Iterate over the versions
    for (xml_node<>* version_node = root_node->first_node("feature"); version_node != nullptr;
         version_node = version_node->next_sibling("feature"))
    {
        code << "  if " << render_from(version_node->first_attribute("number")->value(), false)
             << " {\n";

        std::vector<std::string> enums_list = { "cl_platform_info",
                                                "cl_device_info",
                                                "cl_context_info",
                                                "cl_command_queue_info",
                                                "cl_buffer_create_type",
                                                "cl_image_info",
                                                "cl_pipe_info",
                                                "cl_mem_info",
                                                "cl_sampler_info",
                                                "cl_program_info",
                                                "cl_program_build_info",
                                                "cl_kernel_exec_info",
                                                "cl_kernel_info",
                                                "cl_kernel_work_group_info",
                                                "cl_kernel_sub_group_info",
                                                "cl_kernel_arg_info",
                                                "cl_event_info",
                                                "cl_profiling_info" };

        for (auto i : enums_list)
        {
            for (xml_node<>* enum_node = version_node->first_node("require"); enum_node != nullptr;
                 enum_node = enum_node->next_sibling("require"))
            {
                if (enum_node->first_attribute("comment") != nullptr)
                {
                    // printf("I have visited %s.\n",
                    //     enum_node->first_attribute("comment")->value());
                    if (strstr(enum_node->first_attribute("comment")->value(), i.c_str())
                        != nullptr)
                    {
                        code << "    if (strcmp(name, \"" << i << "\") == 0)\n"
                             << "      switch (param) {\n";

                        for (xml_node<>* enum_val = enum_node->first_node("enum");
                             enum_val != nullptr; enum_val = enum_val->next_sibling("enum"))
                        {
                            // remove [] from the type - we demand at least 1 element for any
                            // returned array
                            std::string type = enum_val->first_attribute("return_type")->value();
                            type = std::regex_replace(type, std::regex("\\[\\]$"), "[1]");
                            // type = std::regex_replace(type, std::regex("\\[\\]$"), "");

                            code << "        case " << enum_val->first_attribute("name")->value()
                                 << ":\n"
                                 << "          return sizeof(" << type << ");\n";
                        }

                        code << "      }\n\n";
                    }
                }
            }
        }

        code << "  }\n\n";
        // printf("I have visited %s.\n",
        //     version_node->first_attribute("number")->value());
        //     version_node->value());
    }

    code << "  *layer::log_stream << \"Unknown return type passed to literal_list(). This is a bug "
            "in the param_verification layer.\" << std::endl;\n"
         << "  return 0;\n"
         << "}\n\n";
    // printf("\n");

    code << "// special case of cl_image_format *\n"
         << "template<>\n"
         << "size_t literal_list(cl_version version, const char * name, cl_image_format * const "
            "param)\n"
         << "{\n"
         << "  (void)version;\n"
         << "  (void)name;\n"
         << "  return pixel_size(param);\n"
         << "}\n"
         << "template<>\n"
         << "size_t literal_list(cl_version version, const char * name, const cl_image_format * "
            "const param)\n"
         << "{\n"
         << "  (void)version;\n"
         << "  (void)name;\n"
         << "  return pixel_size(param);\n"
         << "}\n\n";
}

void parse_queries(std::stringstream& code, xml_node<>*& root_node)
{
    ///////////////////////////////////////////////////////////////////////
    // queries
    ///////////////////////////////////////////////////////////////////////

    code << "template<cl_uint property>\n"
         << "using return_type =\n";

    std::string end = ";\n";

    std::vector<std::string> enums_list = { "cl_platform_info",
                                            "cl_device_info",
                                            "cl_context_info",
                                            "cl_command_queue_info",
                                            "cl_image_info",
                                            "cl_pipe_info",
                                            "cl_mem_info",
                                            "cl_sampler_info",
                                            "cl_program_info",
                                            "cl_program_build_info",
                                            "cl_kernel_exec_info",
                                            "cl_kernel_info",
                                            "cl_kernel_work_group_info",
                                            "cl_kernel_sub_group_info",
                                            "cl_kernel_arg_info",
                                            "cl_event_info",
                                            "cl_profiling_info" };

    // Iterate over the versions
    for (xml_node<>* version_node = root_node->first_node("feature"); version_node != nullptr;
         version_node = version_node->next_sibling("feature"))
    {
        for (auto i : enums_list)
        {
            for (xml_node<>* enum_node = version_node->first_node("require"); enum_node != nullptr;
                 enum_node = enum_node->next_sibling("require"))
            {
                if (enum_node->first_attribute("comment") != nullptr)
                {
                    // printf("I have visited %s.\n",
                    //     enum_node->first_attribute("comment")->value());
                    if (strstr(enum_node->first_attribute("comment")->value(), i.c_str())
                        != nullptr)
                    {
                        for (xml_node<>* enum_val = enum_node->first_node("enum");
                             enum_val != nullptr; enum_val = enum_val->next_sibling("enum"))
                        {
                            // remove [] from the type - we want only 1 element for any returned
                            // array
                            std::string type = enum_val->first_attribute("return_type")->value();
                            type = std::regex_replace(type, std::regex("\\[\\]$"), "");

                            code << "  std::conditional_t<property == "
                                 << enum_val->first_attribute("name")->value() << ", " << type
                                 << ",\n";
                            end = "> " + end;
                        }
                    }
                }
            }
        }

        // printf("I have visited %s.\n",
        //     version_node->first_attribute("number")->value());
        //     version_node->value());
    }
    code << "  void" << end << "\n\n";
}

void render_fetch_version(std::stringstream& code,
                          const std::string& handle,
                          const std::string& name)
{
    // Logic replicated from
    // https://github.com/KhronosGroup/OpenCL-ICD-Loader/blob/main/scripts/icd_dispatch_generated.c.mako#L136
    code << "  auto get_version = [=] {\n";
    if (name == "clCreateContext")
    {
        code << "    return get_object_version(devices && num_devices > 0 ? devices[0] : "
                "nullptr)\n;";
    }
    else if (name == "clWaitForEvents")
    {
        code << "    return get_object_version(event_list && num_events > 0 ? event_list[0] : "
                "nullptr)\n;";
    }
    else if (name == "clCreateContextFromType")
    {
        code << "    return get_object_version(get_context_properties_platform(properties));\n";
    }
    else if (name != "clUnloadCompiler" && name != "clGetPlatformIDs")
    {
        code << "    return get_object_version(" << handle << ");\n";
    }
    code << "  };\n";
}

void parse_commands(std::stringstream& code, xml_node<>*& root_node)
{
    ///////////////////////////////////////////////////////////////////////
    // commands
    ///////////////////////////////////////////////////////////////////////

    std::stringstream init_dispatch;

    // Iterate over the commands
    for (xml_node<>* commands_node = root_node->first_node("commands"); commands_node != nullptr;
         commands_node = commands_node->next_sibling("commands"))
    {
        for (xml_node<>* command_node = commands_node->first_node("command");
             command_node != nullptr; command_node = command_node->next_sibling("command"))
        {
            xml_node<>* proto_node = command_node->first_node("proto");
            std::string qual = (proto_node->value() != nullptr)
                ? std::regex_replace(proto_node->value(), std::regex("[ ]+"), " ")
                : "";
            const char* const name = proto_node->first_node("name")->value();
            std::string type = proto_node->first_node("type")->value();

            std::string invoke = std::string("tdispatch->") + name;
            invoke += "(\n";
            const std::string prefix{ "CL_API_ENTRY" }, suffix{ "CL_API_CALL" };
            std::string proto =
                prefix + " " + type + " " + qual + " " + suffix + " " + name + "_layer(\n";
            proto = std::regex_replace(proto, std::regex("[ ]+"), " ");

            init_dispatch << "    dispatch." << name << " = &" << name << "_layer;\n";

            //            if (param_node->value())
            // printf("I have visited %s\n", proto.c_str());
            //            code << proto;

            std::string handle;

            int n = 0;
            func_params.clear();
            for (xml_node<>* param_node = command_node->first_node("param"); param_node != nullptr;
                 param_node = param_node->next_sibling("param"))
            {
                std::string tmp = (n != 0) ? ",\n  " : "  ";
                invoke += tmp + "  ";

                xml_node<>* node = param_node->first_node("type");
                if (node != nullptr)
                    func_params.insert(std::pair<std::string, std::string>(
                        param_node->first_node("name")->value(),
                        param_node->first_node("type")->value()));

                if (n == 0)
                {
                    handle = param_node->first_node("name")->value();
                }

                // read all the contents as text omitting tags - works for 1-level tags only
                node = param_node->first_node();
                while (node != nullptr)
                {
                    if (tmp.back() != ' ') tmp += " ";
                    tmp += std::regex_replace(node->value(), std::regex("[ ]+"), " ");
                    if (strstr(node->name(), "name") != nullptr) invoke += node->value();
                    // printf("%s ", node->value());
                    node = node->next_sibling();
                }
                tmp = std::regex_replace(tmp, std::regex(" \\)"), ")");
                // printf("%s", tmp.c_str());

                proto += tmp;
                ++n;
            }
            proto += ")\n";
            invoke += ");\n";

            code << proto << "{\n";

            std::stringstream body;
            generate_get_version = false;
            bool generate_label = false;

            for (xml_node<>*violation_node = command_node->first_node("if"),
                *result_node = command_node->first_node("then");
                 (violation_node != nullptr) && (result_node != nullptr);
                 violation_node = violation_node->next_sibling("if"),
                result_node = result_node->next_sibling("then"))
            {
                body << "  if " << parse_violation(violation_node->first_node()) << " {\n";

                std::string log_ret;
                std::string log_param;
                for (xml_node<>*name_node = result_node->first_node("name"),
                    *value_node = result_node->first_node("value");
                     (name_node != nullptr) && (value_node != nullptr);
                     name_node = name_node->next_sibling("name"),
                    value_node = value_node->next_sibling("value"))
                {
                    if (strcmp(name, name_node->value()) == 0)
                    {
                        log_ret = value_node->value();
                    }
                    else
                    {
                        if (log_param != "") log_param += ", ";
                        log_param +=
                            std::string("*") + name_node->value() + " = " + value_node->value();
                    }
                }

                body << "    *layer::log_stream << \"In " << name << ": \"\n";

                for (xml_node<>* log_node = result_node->first_node("log"); log_node != nullptr;
                     log_node = log_node->next_sibling("log"))
                {
                    body << "      \"" << log_node->value() << ". \"\n";
                }

                body << "      \"Returning " << log_ret;
                if (log_param != "")
                {
                    if (log_ret != "") body << ", ";
                }
                body << log_param << ".\" << std::endl;\n";

                body << "    if (layer::settings.transparent)\n";
                body << "      goto " << name << "_dispatch;\n";
                generate_label = true;

                std::string ret;
                for (xml_node<>*name_node = result_node->first_node("name"),
                    *value_node = result_node->first_node("value");
                     (name_node != nullptr) && (value_node != nullptr);
                     name_node = name_node->next_sibling("name"),
                    value_node = value_node->next_sibling("value"))
                {
                    if (strcmp(name, name_node->value()) == 0)
                    {
                        ret = value_node->value();
                    }
                    else
                    {
                        body << "    if (" << name_node->value() << " != NULL)\n"
                             << "      *" << name_node->value() << " = " << value_node->value()
                             << ";\n";
                    }
                }

                if (ret != "") body << "    return " << ret << ";\n";
                body << "  }\n\n";
            }

            if (generate_label) body << name << "_dispatch:\n";
            body << "  return " << invoke << "}\n\n";

            if (generate_get_version)
            {
                render_fetch_version(code, handle, name);
            }

            code << body.rdbuf();
        }
    }

    code << "void init_dispatch() {\n" << init_dispatch.rdbuf() << "}\n";
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <path to cl.xml>" << std::endl;
        return EXIT_FAILURE;
    }

    std::ifstream theFile(argv[1]);
    if (!theFile)
    {
        std::cerr << "Error: failed to open '" << argv[1] << "'" << std::endl;
        return EXIT_FAILURE;
    }

    // stream for generated code
    std::stringstream code;
    code << "#ifdef _WIN32\n"
         << "#define NOMINMAX\n"
         << "#endif\n"
         << "#include <CL/cl.h>\n"
         << "#include <string.h>\n"
         << "#include <memory>\n"
         << "#include <vector>\n"
         << "#include <algorithm>\n"
         << "#include <functional>\n"
         << "#include \"param_verification.hpp\"\n\n\n";

    xml_document<> doc;
    xml_node<>* root_node;
    // Read the xml file into a vector
    std::vector<char> buffer((std::istreambuf_iterator<char>(theFile)),
                             std::istreambuf_iterator<char>());
    buffer.push_back('\0');
    // Parse the buffer using the xml file parsing library into doc
    doc.parse<0>(&buffer[0]);

    // Find our root node
    root_node = doc.first_node("registry");

    parse_enums(code, root_node);

    parse_bitfields(code, root_node);

    parse_literal_lists(code, root_node);

    parse_queries(code, root_node);

    // dummy funcs
    code << "template<typename T>\n"
         << "bool array_len_ls(T * ptr, size_t size)\n"
         << "{\n"
         << "  (void)ptr;\n"
         << "  (void)size;\n"
         << "  return false;\n"
         << "}\n\n";

    // real funcs
    code << "template<typename T>\n"
         << "bool any_zero(T * ptr, size_t size)\n"
         << "{\n"
         << "  for (size_t i = 0; i < size; ++i)\n"
         << "    if (ptr[i] == 0) return true;\n"
         << "  return false;\n"
         << "}\n\n";

    code << "template<typename T>\n"
         << "bool any_nullptr(T ** ptr, size_t size)\n"
         << "{\n"
         << "  for (size_t i = 0; i < size; ++i)\n"
         << "    if (ptr[i] == NULL) return true;\n"
         << "  return false;\n"
         << "}\n\n";

    code << "template<typename T>\n"
         << "bool any_invalid(T * ptr, size_t size)\n"
         << "{\n"
         << "  for (size_t i = 0; i < size; ++i)\n"
         << "    if (!object_is_valid(ptr[i])) return true;\n"
         << "  return false;\n"
         << "}\n\n";

    code << "template<typename T>\n"
         << "bool any_non_null_invalid(T * ptr, size_t size)\n"
         << "{\n"
         << "  for (size_t i = 0; i < size; ++i)\n"
         << "    if ((ptr[i] != NULL) && !object_is_valid(ptr[i])) return true;\n"
         << "  return false;\n"
         << "}\n\n";

    code << "bool any_not_available(const cl_device_id * devices, size_t size)\n"
         << "{\n"
         << "  cl_bool avail = false;\n"
         << "  for (size_t i = 0; i < size; ++i) {\n"
         << "    tdispatch->clGetDeviceInfo(devices[i], CL_DEVICE_AVAILABLE, sizeof(cl_bool), "
            "&avail, NULL);\n"
         << "    if (!avail) return true;\n"
         << "    avail = false;\n"
         << "  }\n"
         << "  return false;\n"
         << "}\n\n";

    code << "//////////////////////////////////////////////////////////////////////\n\n";

    code << "#include \"object_is_valid.cpp\"\n"
         << "#include \"list_violation.cpp\"\n"
         << "#include \"struct_violation.cpp\"\n"
         << "\n\n";

    code << "//////////////////////////////////////////////////////////////////////\n\n";

    parse_commands(code, root_node);

    std::ofstream file("res.cpp");
    file << code.str();
    file.close();
    // std::cout << code.str();
    return 0;
}
