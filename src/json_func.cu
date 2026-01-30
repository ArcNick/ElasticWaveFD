#include <iostream>
#include <fstream>
#include <string>

std::string readJsonFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "无法打开文件: " << filename << '\n';
        exit(1);
    }
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
