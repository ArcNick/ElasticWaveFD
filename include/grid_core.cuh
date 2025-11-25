#ifndef GRID_CORE_CUH
#define GRID_CORE_CUH

#include <cuda_runtime.h>

class Grid_Core {
public:
    float *vx, *vz, *sx, *sz, *txz;
    int nx, nz;
    bool mem_location;  // 0 : host, 1 : device

    struct View {
        float *vx, *vz, *sx, *sz, *txz;
        int nx, nz;
        bool mem_location;
    };
    
    View view() {
        return (View){vx, vz, sx, sz, txz, nx, nz, mem_location};
    }
    
    Grid_Core(int nx, int nz, bool mem_location);
    ~Grid_Core();

    void memcpy_to_host_from(const Grid_Core &rhs);
};

#endif