#include "utils.hpp"

#include <iostream>
#include <string>
#include <cstdlib>

int main(int argc, char* argv[]) {
  if (argc <= 4) {
    std::cerr << "usage: " << argv[0] << " <prefix> <setting> <default> <expected>" << std::endl;
    return EXIT_FAILURE;
  }

  const auto settings = ocl_layer_utils::load_settings();
  const auto parser =
    ocl_layer_utils::settings_parser(argv[1], settings);

  std::string value = argv[3];
  parser.get_filename(argv[2], value);
  std::cout << value << std::endl;

  return value == argv[4] ? EXIT_SUCCESS : EXIT_FAILURE;
}
