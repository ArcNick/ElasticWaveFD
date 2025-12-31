#ifndef SIMULATION_UTILS_CUH
#define SIMULATION_UTILS_CUH

#include "grid_model.cuh"
#include "grid_core.cuh"
#include <memory>

std::unique_ptr<float[]> ricker_wavelet(int nt, float dt, float fpeak);
// __global__ void thomsen_to_stiffness(Grid_Model_Thomsen::View gm);

// idx = iz * nx + ix
#define VP(ix, iz) ((gm).vp0[(iz) * (gm).nx + (ix)])
#define VS(ix, iz) ((gm).vs0[(iz) * (gm).nx + (ix)])
#define RHO(ix, iz) ((gm).rho[(iz) * (gm).nx + (ix)])
#define EPS(ix, iz) ((gm).epsilon[(iz) * (gm).nx + (ix)])
#define DEL(ix, iz) ((gm).delta[(iz) * (gm).nx + (ix)])
#define GAM(ix, iz) ((gm).gamma[(iz) * (gm).nx + (ix)])
#define C11(ix, iz) ((gm).C11[(iz) * (gm).nx + (ix)])
#define C13(ix, iz) ((gm).C13[(iz) * (gm).nx + (ix)])
#define C33(ix, iz) ((gm).C33[(iz) * (gm).nx + (ix)])
#define C55(ix, iz) ((gm).C55[(iz) * (gm).nx + (ix)])

#define VX_C(ix, iz) ((gc).vx[(iz) * ((gc).nx - 1) + (ix) + (gc.nx - 1) * (gc.nz) * cur])
#define VZ_C(ix, iz) ((gc).vz[(iz) * (gc).nx + (ix) + (gc.nx) * (gc.nz - 1) * cur])
#define SX_C(ix, iz) ((gc).sx[(iz) * (gc).nx + (ix) + (gc.nx) * (gc.nz) * cur])
#define SZ_C(ix, iz) ((gc).sz[(iz) * (gc).nx + (ix) + (gc.nx) * (gc.nz) * cur])
#define TXZ_C(ix, iz) ((gc).txz[(iz) * ((gc).nx - 1) + (ix) + (gc.nx - 1) * (gc.nz - 1) * cur])
#define VX_P(ix, iz) ((gc).vx[(iz) * ((gc).nx - 1) + (ix) + (gc.nx - 1) * (gc.nz) * pre])
#define VZ_P(ix, iz) ((gc).vz[(iz) * (gc).nx + (ix) + (gc.nx) * (gc.nz - 1) * pre])
#define SX_P(ix, iz) ((gc).sx[(iz) * (gc).nx + (ix) + (gc.nx) * (gc.nz) * pre])
#define SZ_P(ix, iz) ((gc).sz[(iz) * (gc).nx + (ix) + (gc.nx) * (gc.nz) * pre])
#define TXZ_P(ix, iz) ((gc).txz[(iz) * ((gc).nx - 1) + (ix) + (gc.nx - 1) * (gc.nz - 1) * pre])

enum MemoryType {
    HOST_MEM = 0,
    DEVICE_MEM = 1
};

#endif