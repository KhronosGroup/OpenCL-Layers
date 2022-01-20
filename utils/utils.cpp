#include "utils.hpp"

#include <algorithm>
#include <fstream>
#include <locale>
#include <map>
#include <string>
#include <utility>

#include <sys/stat.h>

namespace ocl_layer_utils {

namespace detail {
const char *get_environment(const char *variable) {
#ifdef __ANDROID__
  // TODO: Implement get_environment for android
  return "";
#else
  const char *output = std::getenv(variable);
  return output != nullptr ? output : "";
#endif
}

std::string trim(std::string input) {
  constexpr static auto *whitespace_chars = "\t ";
  const auto last_non_whitespace = input.find_last_not_of(whitespace_chars);
  if (last_non_whitespace < input.size() - 1) {
    input.erase(last_non_whitespace + 1);
  }
  auto first_non_whitespace = input.find_first_not_of(whitespace_chars);
  if (first_non_whitespace == std::string::npos) {
    first_non_whitespace = input.size();
  }
  input.erase(input.begin(), input.begin() + first_non_whitespace);
  return input;
}

std::string to_upper(std::string input) {
  const auto &facet = std::use_facet<std::ctype<char>>(std::locale::classic());
  for (auto &c : input) {
    c = facet.toupper(c);
  }
  return input;
}

void parse_bool(const std::string &option, bool &out) {
  static constexpr auto positive_words = {
      "on", "y", "yes", "1", "true",
  };
  static constexpr auto negative_words = {"off", "n", "no", "0", "false"};
  if (std::find(positive_words.begin(), positive_words.end(), option) !=
      positive_words.end()) {
    out = true;
    return;
  }
  if (std::find(negative_words.begin(), negative_words.end(), option) !=
      negative_words.end()) {
    out = false;
    return;
  }
}

} // namespace detail

std::string find_settings() {
  struct stat file_info;
  std::string found_location = "cl_layer_settings.txt";
#ifdef __linux__
  {
    std::string search_path = detail::get_environment("XDG_DATA_HOME");
    if (search_path == "") {
      search_path = detail::get_environment("HOME");
      if (search_path != "") {
        search_path += "/.local/share";
      }
    }
    if (search_path != "") {
      const std::string home_file =
          search_path + "/opencl/settings.d/cl_layer_settings.txt";
      if (stat(home_file.c_str(), &file_info) == 0 &&
          (file_info.st_mode & S_IFREG) > 0) {
        found_location = home_file;
      }
    }
  }
#endif
  std::string config_path =
      detail::get_environment("OPENCL_LAYERS_SETTINGS_PATH");
  if (stat(config_path.c_str(), &file_info) == 0) {
    if ((file_info.st_mode & S_IFDIR) > 0) {
      config_path += "/cl_layer_settings.txt";
    }
    found_location = config_path;
  }
  return found_location;
}

std::map<std::string, std::string> load_settings() {
  auto result = std::map<std::string, std::string>{};
  auto file = std::ifstream{find_settings()};
  if (!file.good()) {
    return result;
  }

  for (std::string line; std::getline(file, line);) {
    const auto comment_pos = line.find_first_of("#");
    if (comment_pos != std::string::npos) {
      line.erase(comment_pos);
    }
    const auto equals_pos = line.find_first_of("=");
    if (equals_pos != std::string::npos) {
      auto option = detail::trim(line.substr(0, equals_pos));
      result[option] = detail::trim(line.substr(equals_pos + 1));
    }
  }
  return result;
}

void settings_parser::get_bool(const char *option_name, bool &out) const {
  if (settings_->find(prefix_ + "." + option_name) != settings_->end()) {
    detail::parse_bool(settings_->at(prefix_ + "." + option_name), out);
  }
  detail::parse_bool(
      detail::get_environment(detail::to_upper("CL_" + prefix_ + "_" + option_name).c_str()),
      out);
}

void settings_parser::get_filename(const char *option_name,
                                   std::string &out) const {
  const auto full_option_name = prefix_ + "." + option_name;
  if (settings_->find(full_option_name) != settings_->end() &&
      settings_->at(full_option_name) != "") {
    out = settings_->at(full_option_name);
  }
  const std::string env_option =
      detail::get_environment(detail::to_upper("CL_" + prefix_ + "_" + option_name).c_str());
  if (env_option != "") {
    out = env_option;
  }
}

} // namespace ocl_layer_utils