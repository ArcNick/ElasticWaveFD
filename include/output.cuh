#ifndef OUTPUT_CUH
#define OUTPUT_CUH

#include "grid_core.cuh"
#include <vector>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

struct Array_Info {
    float *data[2];
    int lenx, lenz;
    std::string name;
    Array_Info(float* const d[2], int lx, int lz, const std::string &n);
    ~Array_Info() = default;
};

class Snapshot {
public:
    int nz, nx;
    std::vector<Array_Info> arrays;
    fs::path output_dir;

    Snapshot(const Grid_Core &g);
    ~Snapshot() = default;

    void output(int it, float dt, int cur);
};

#endif