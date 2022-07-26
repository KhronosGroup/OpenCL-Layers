#include <map>
#include <string>
#include <iostream>
#include <memory>
#include <CL/cl.h>

namespace ocl_layer_utils {

namespace detail {
template <typename T>
void parse_enumeration(const std::string &option,
                       const std::map<std::string, T> &values_map, T &out) {
  if (values_map.find(option) != values_map.end()) {
    out = values_map.at(option);
  }
}

std::string to_upper(std::string input);

// This function returns a copy of the env var owned by the application
bool get_environment(const std::string& variable, std::string& value);
} // namespace detail

std::string find_settings();

std::map<std::string, std::string> load_settings();

struct settings_parser {
  settings_parser(std::string prefix,
                  const std::map<std::string, std::string> &settings)
      : prefix_{std::move(prefix)}, settings_{&settings} {}

  void get_bool(const char *option_name, bool &out) const;
  void get_filename(const char *option_name, std::string &out) const;

  template <typename T>
  void get_enumeration(const char *option_name,
                       const std::map<std::string, T> &values_map,
                       T &out) const {
    get_option(option_name, [&values_map, &out](const std::string &value) {
      detail::parse_enumeration(value, values_map, out);
    });
  }

private:
  // Fetch the value corresponding to a particular option key. This value can
  // either come from the settings file, or it can be overridden from the environment.
  // If such an argument was found, the function calls the `parse` callback to process it.
  // This callback may be called multiple times, in which case the lattermost option should remain.
  template <typename ParseCallback>
  void get_option(const char *option_name, ParseCallback parse) const {
    const auto full_option_name = prefix_ + "." + option_name;
    auto it = settings_->find(full_option_name);
    if (it != settings_->end() && it->second != "")
    {
      parse(it->second);
    }
    std::string env_option_value;
    const std::string option_env_var = detail::to_upper("OPENCL_" + prefix_ + "_" + option_name);
    if (detail::get_environment(option_env_var, env_option_value))
    {
      parse(env_option_value);
    }
  }

  std::string prefix_;
  const std::map<std::string, std::string> *settings_;
};

namespace detail {
struct stream_deleter {
  void operator()(std::ostream *stream) noexcept {
    if (stream != &std::cout && stream != &std::cerr) {
      delete stream;
    }
  }
};
}

using stream_ptr = std::unique_ptr<std::ostream, detail::stream_deleter>;

cl_version parse_cl_version_string(const char* version_str, cl_version* parsed_version);
} // namespace ocl_layer_utils
