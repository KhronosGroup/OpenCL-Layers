#include "utils.hpp"

#include <algorithm>
#include <fstream>
#include <locale>
#include <map>
#include <string>
#include <utility>

#include <initializer_list>
#include <vector>
#include <string>

#include <iostream>

#include <sys/stat.h>

#ifdef _WIN32
#include <windows.h>
#include <stdlib.h> // _dupenv_s
#endif
namespace ocl_layer_utils {

namespace detail {

#ifdef _WIN32
bool is_high_integrity_level()
{
  bool isHighIntegrityLevel = false;

  HANDLE processToken;
  if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY | TOKEN_QUERY_SOURCE, &processToken))
  {
    // Maximum possible size of SID_AND_ATTRIBUTES is maximum size of a SID + size of attributes DWORD.
    char mandatoryLabelBuffer[SECURITY_MAX_SID_SIZE + sizeof(DWORD)] = {0};
    DWORD bufferSize;
    if (GetTokenInformation(processToken, TokenIntegrityLevel, mandatoryLabelBuffer, sizeof(mandatoryLabelBuffer),
                            &bufferSize) != 0)
    {
      const TOKEN_MANDATORY_LABEL* mandatoryLabel = (const TOKEN_MANDATORY_LABEL*)(mandatoryLabelBuffer);
      const DWORD subAuthorityCount = *GetSidSubAuthorityCount(mandatoryLabel->Label.Sid);
      const DWORD integrityLevel = *GetSidSubAuthority(mandatoryLabel->Label.Sid, subAuthorityCount - 1);

      isHighIntegrityLevel = integrityLevel > SECURITY_MANDATORY_MEDIUM_RID;
    }
      CloseHandle(processToken);
  }
  return isHighIntegrityLevel;
}

// NOTE: should be std::initializer_list, but returning it invokes lifetime issues
std::vector<HKEY> hives_to_check()
{
  if (detail::is_high_integrity_level())
    return {HKEY_LOCAL_MACHINE};
  else
    return {HKEY_LOCAL_MACHINE, HKEY_CURRENT_USER};
}
#endif

// NOTE 1: Ideally this should return a std::optional<std::string> instead
//         of bool and an out-var reference, but we're constrained to C++14.
//
// NOTE 2: The implementation needs const_cast<> becasue in C++14 .data()
//         of std::string is const char*. Again, C++17 fixes this.
bool get_environment(const std::string& variable, std::string& value) {
#ifdef __ANDROID__
  // TODO: Implement get_environment for android
  return "";
#elif defined(_WIN32)
  size_t var_size;
  std::string temp;
  errno_t err = getenv_s(&var_size, NULL, 0, variable.c_str());
  if(var_size == 0 || err != 0) return false;
  temp.resize(var_size);
  err = getenv_s(&var_size, const_cast<char*>(temp.data()), var_size, variable.c_str());
  if (err == 0)
  {
    temp.pop_back(); // pop the null-terminator
    value = std::move(temp);
  }
  return err == 0;
#else
  const char *output = std::getenv(variable.c_str());
  if (output != nullptr)
  {
    value.assign(output);
    value = std::string(output);
    return true;
  }
  else return false;
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
  struct stat info;
  std::string found_location = "cl_layer_settings.txt";
  auto use_file_if_exists = [&](const std::string& path)
  {
    bool stat_success = stat(path.c_str(), &info) == 0;
    bool is_regular_file = (info.st_mode & S_IFMT) == S_IFREG;
    if (stat_success && is_regular_file)
    {
      found_location = path;
      return true;
    }
    else return false;
  };
#ifdef __linux__
  {
    std::string search_path;
    if (detail::get_environment("XDG_DATA_HOME", search_path) && !search_path.empty())
    {
      use_file_if_exists(search_path += "/settings.d/opencl/cl_layer_settings.txt");
    }
    else if (detail::get_environment("HOME", search_path))
        use_file_if_exists(search_path += "/.local/share/cl_layer_settings.txt");
  }
#elif defined(_WIN32)
  {
    for(HKEY hive : detail::hives_to_check())
    {
      HKEY key;
      if (ERROR_SUCCESS == RegOpenKeyEx(hive, "Software\\Khronos\\OpenCL\\Settings", 0, KEY_READ, &key))
      {
        // NOTE 1: Should be std::string, but std::string::data() returns const char*.
        //         It returns a char* starting from C++17, needed for RegEnumValue
        //
        // NOTE 2: Querying name_size doesn't work the same as value_size.
        //         We alloc pessimistically for registry name max size as documented.
        std::vector<char> name(32'767);
        DWORD name_size = static_cast<DWORD>(name.capacity()), value, value_size = sizeof(value_size), type;
        LSTATUS err;
        for (DWORD i = 0 ; ERROR_NO_MORE_ITEMS != (err = RegEnumValue(key, i, name.data(), &name_size, nullptr, &type, nullptr, &value_size)); ++i, value_size = sizeof(value_size), name_size = static_cast<DWORD>(name.capacity()))
        {
          // Check if the registry entry is a dword
          if (type != REG_DWORD) continue;

          ++name_size; // ++because subsequent call will write a null-terminator as well

          RegEnumValue(key, i++, name.data(), &name_size, nullptr, &type, reinterpret_cast<LPBYTE>(&value), &(value_size = sizeof(value_size)));

          name.resize(name_size); // ++because subsequent call will write a null-terminator as well

          // Check if the registry entry has value of zero
          if (value != 0) continue;

          if (stat(name.data(), &info) == 0)
          {
            if ((info.st_mode & S_IFDIR) > 0)
            {
              std::string config_path(name.data(), name.size());
              config_path += "\\cl_layer_settings.txt";
              if (use_file_if_exists(config_path)) break;
            }
            else
            {
              if (use_file_if_exists(std::string(name.data(), name.size()))) break;
            }
          }
        }
        RegCloseKey(key);
      }
    }
  }
#endif
  std::string config_path;
  if (detail::get_environment("OPENCL_LAYERS_SETTINGS_PATH", config_path))
    if (stat(config_path.c_str(), &info) == 0) {
      if ((info.st_mode & S_IFDIR) > 0)
        config_path += "/cl_layer_settings.txt";
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
  get_option(option_name, [&out](const std::string &value) {
    detail::parse_bool(value, out);
  });
}

void settings_parser::get_filename(const char *option_name, std::string &out) const {
  get_option(option_name, [&out](const std::string &value) {
    out = value;
  });
}

} // namespace ocl_layer_utils
