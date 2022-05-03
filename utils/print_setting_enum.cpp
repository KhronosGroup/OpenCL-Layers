#include "utils.hpp"

#include <iostream>
#include <string>
#include <map>
#include <cstdlib>

int main(int argc, char* argv[]) {
  if (argc <= 5) {
    std::cerr << "usage: " << argv[0] << " <prefix> <setting> <default> <expected> <variants...>" << std::endl;
    return EXIT_FAILURE;
  }

  const auto settings = ocl_layer_utils::load_settings();
  const auto parser =
    ocl_layer_utils::settings_parser(argv[1], settings);

  std::map<std::string, std::string> options;
  for (int i = 5; i < argc; ++i) {
    options.insert({std::string(argv[i]), std::string(argv[i])});
  }

  std::string value = argv[3];
  parser.get_enumeration(argv[2], options, value);

  return value == argv[4] ? EXIT_SUCCESS : EXIT_FAILURE;
}
