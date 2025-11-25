#ifndef GRID_MODEL_CUH
#define GRID_MODEL_CUH

#include <cuda_runtime.h>
#include <array>

class Grid_Model {
public:
    float *vp0, *vs0, *rho;
    float *epsilon, *delta, *gamma;
    float *C11, *C13, *C33, *C44, *C66;

    int nx, nz;
    bool mem_location;  // 0 : host, 1 : device

    struct View {
        float *vp0, *vs0, *rho;
        float *epsilon, *delta, *gamma;
        float *C11, *C13, *C33, *C44, *C66;
        int nx, nz;
        bool mem_location;
    };
    
    View view() {
        return (View){
            vp0, vs0, rho, 
            epsilon, delta, gamma, 
            C11, C13, C33, C44, C66, 
            nx, nz, mem_location
        };
    }

    Grid_Model(int nx, int nz, bool mem_location);
    ~Grid_Model();

    void read(const std::array<const char *, 6> &files);
    void memcpy_to_device_from(const Grid_Model &rhs);
    void calc_stiffness();
};

#endif