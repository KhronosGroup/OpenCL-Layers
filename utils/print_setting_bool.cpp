#include "utils.hpp"

#include <iostream>
#include <string>
#include <cstdlib>

int main(int argc, char* argv[]) {
  auto parse_bool = [](const std::string& str, const char* param, bool& value) {
    if (str == "true")
    {
      value = true;
    }
    else if (str == "false")
    {
      value = false;
    }
    else
    {
      std::cerr << "error: <" << param << "> must be 'true' or 'false'" << std::endl;
      return false;
    }
    return true;
  };

  if (argc <= 4) {
    std::cerr << "usage: " << argv[0] << " <prefix> <setting> <default> <expected>" << std::endl;
    return EXIT_FAILURE;
  }

  const auto settings = ocl_layer_utils::load_settings();
  const auto parser =
    ocl_layer_utils::settings_parser(argv[1], settings);

  bool value;
  if (!parse_bool(argv[3], "default", value))
  {
    return EXIT_FAILURE;
  }

  parser.get_bool(argv[2], value);
  std::cout << (value ? "true" : "false") << std::endl;

  bool expected;
  if (!parse_bool(argv[4], "expected", expected))
  {
    return EXIT_FAILURE;
  }
  return value == expected ? EXIT_SUCCESS : EXIT_FAILURE;
}
