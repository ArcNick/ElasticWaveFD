#ifndef SIMULATION_UTILS_CUH
#define SIMULATION_UTILS_CUH

#include "grid_model.cuh"
#include "grid_core.cuh"

float *ricker_wave(int nt, float dt, float fpeak);

__global__ void thomsen_to_stiffness(Grid_Model::View gm);

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
#define C44(ix, iz) ((gm).C44[(iz) * (gm).nx + (ix)])
#define C66(ix, iz) ((gm).C66[(iz) * (gm).nx + (ix)])

#define VX(ix, iz) ((gc).vx[(iz) * ((gc).nx - 1) + (ix)])
#define VZ(ix, iz) ((gc).vz[(iz) * (gc).nx + (ix)])
#define SX(ix, iz) ((gc).sx[(iz) * (gc).nx + (ix)])
#define SZ(ix, iz) ((gc).sz[(iz) * (gc).nx + (ix)])
#define TXZ(ix, iz) ((gc).txz[(iz) * ((gc).nx - 1) + (ix)])

#define HOST_MEM 0
#define DEVICE_MEM 1

#endif