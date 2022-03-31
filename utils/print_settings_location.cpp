#include "utils.hpp"

#include <iostream>

int main(int argc, char* argv[]) {
    auto location = ocl_layer_utils::find_settings();
    std::cout << location << std::endl;
    if (argc > 1)
    {
        std::cout << std::string(argv[1]) << std::endl;
        return location == std::string(argv[1]) ? 0 : 1;
    }
    else
        return 0;
}