#include <map>
#include <string>

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
const char *get_environment(const char *variable);
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
    const auto full_option_name = prefix_ + "." + option_name;
    if (settings_->find(full_option_name) != settings_->end() &&
        settings_->at(full_option_name) != "") {
      detail::parse_enumeration(settings_->at(full_option_name), values_map,
                                out);
    }
    const std::string env_option = detail::get_environment(
        detail::to_upper("CL_" + prefix_ + "_" + option_name).c_str());
    if (env_option != "") {
      detail::parse_enumeration(env_option, values_map, out);
    }
  }

private:
  std::string prefix_;
  const std::map<std::string, std::string> *settings_;
};

} // namespace ocl_layer_utils