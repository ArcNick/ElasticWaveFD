#ifndef GRID_CORE_CUH
#define GRID_CORE_CUH

#include <cstring>

class Grid_Core {
public:
    float *vx[2], *vz[2], *sx[2], *sz[2], *txz[2];
    int nx, nz;
    bool mem_location;  // 0 : host, 1 : device

    struct View {
        float *vx[2], *vz[2], *sx[2], *sz[2], *txz[2];
        int nx, nz;
        bool mem_location;
    };
    
    View view() {
        View v;
        std::memcpy(v.vx, vx, sizeof(vx));
        std::memcpy(v.vz, vz, sizeof(vz));
        std::memcpy(v.sx, sx, sizeof(sx));
        std::memcpy(v.sz, sz, sizeof(sz));
        std::memcpy(v.txz, txz, sizeof(txz));
        v.nx = nx;
        v.nz = nz;
        v.mem_location = mem_location;
        return v;
    }
    
    Grid_Core(int nx, int nz, bool mem_location);
    ~Grid_Core();

    void memcpy_to_host_from(const Grid_Core &rhs);
};

#endif
