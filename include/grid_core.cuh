#ifndef GRID_CORE_CUH
#define GRID_CORE_CUH

#include <cstring>

class Grid_Core {
public:
    float *vx, *vz, *sx, *sz, *txz;
    int nx, nz;
    bool mem_location;  // 0 : host, 1 : device

    // 视图结构体用来给核函数传参
    struct View {
        float *vx, *vz, *sx, *sz, *txz;
        int nx, nz;
    };
    
    View view() {
        return (View){vx, vz, sx, sz, txz, nx, nz};
    }
    
    Grid_Core(int nx, int nz, bool mem_location);
    ~Grid_Core();

    void memcpy_to_host_from(const Grid_Core &rhs);
};

#endif
