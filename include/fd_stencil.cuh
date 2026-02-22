#ifndef FD_STENCIL_CUH
#define FD_STENCIL_CUH

#include <cuda_runtime.h>

__device__ float samp_C55(float *f, int ix, int iz);
__device__ float samp_rho_x(float *f, int ix, int iz);
__device__ float samp_rho_z(float *f, int ix, int iz);

__device__ float samp_vx_z(
    float *f, float ix_global, float iz_global
);

__device__ float samp_vx_x(
    float *f, float ix_global, float iz_global
);

__device__ float samp_vz_x(
    float *f, float ix_global, float iz_global
);

__device__ float samp_vz_z(
    float *f, float ix_global, float iz_global
);

__device__ float samp_sx_z(
    float *f, float ix_global, float iz_global
);

__device__ float samp_sz_x(
    float *f, float ix_global, float iz_global
);

__device__ float samp_txz_x(
    float *f, float ix_global, float iz_global
);

__device__ float samp_txz_z(
    float *f, float ix_global, float iz_global
);

__device__ float dvx_dx_coarse(
    float *f, int ix, int iz, int time
);

__device__ float dvx_dz_coarse(
    float *f, int ix, int iz, int time
);

__device__ float dvz_dx_coarse(
    float *f, int ix, int iz, int time
);

__device__ float dvz_dz_coarse(
    float *f, int ix, int iz, int time
);

__device__ float dsx_dx_coarse(
    float *f, int ix, int iz, int time
);

__device__ float dsz_dz_coarse(
    float *f, int ix, int iz, int time
);

__device__ float dtxz_dx_coarse(
    float *f, int ix, int iz, int time
);

__device__ float dtxz_dz_coarse(
    float *f, int ix, int iz, int time
);

__device__ float dvx_dx_8th(
    float *f, int ix, int iz, int time, int zone
);

__device__ float dvx_dz_8th(
    float *f, int ix, int iz, int time, int zone
);

__device__ float dvz_dx_8th(
    float *f, int ix, int iz, int time, int zone
);

__device__ float dvz_dz_8th(
    float *f, int ix, int iz, int time, int zone
);

__device__ float dsx_dx_8th(
    float *f, int ix, int iz, int time, int zone
);

__device__ float dsz_dz_8th(
    float *f, int ix, int iz, int time, int zone
);

__device__ float dtxz_dx_8th(
    float *f, int ix, int iz, int time, int zone
);

__device__ float dtxz_dz_8th(
    float *f, int ix, int iz, int time, int zone
);

#endif