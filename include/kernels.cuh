#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "grid_manager.cuh"
#include "cpml.cuh"

extern __constant__ float coeff[5][3];

extern __device__ int IdxVxFi(int ix, int iz, int cur, int zone);
extern __device__ int IdxVzFi(int ix, int iz, int cur, int zone);
extern __device__ int IdxSigFi(int ix, int iz, int cur, int zone);
extern __device__ int IdxTxzFi(int ix, int iz, int cur, int zone);

__device__ int get_cpml_idx_x_int(int ix);
__device__ int get_cpml_idx_z_int(int iz);
__device__ int get_cpml_idx_x_half(int ix);
__device__ int get_cpml_idx_z_half(int iz);

__global__ void apply_source(Core core, float src, int cur);
__global__ void update_sigma_coarse(Core core, Model model, PsiVel psi_vel, int cur, int it);
__global__ void update_tau_coarse(Core core, Model model, PsiVel psi_vel, int cur, int it);
__global__ void update_velocity_coarse(Core core, Model model, PsiStr psi_str, int cur, int it);
__global__ void update_sigma_fine(Core core, Model model, int cur, int zone);
__global__ void update_tau_fine(Core core, Model model, int cur, int zone);
__global__ void update_velocity_fine(Core core, Model model, int cur, int zone);
__global__ void apply_fluid_boundary_coarse(Core core, int cur);
__global__ void apply_fluid_boundary_fine(Core core, int cur, int zone);

__global__ void smooth_fine_vx(float *vx, float *temp, int cur, int zone, int lvl);
__global__ void smooth_fine_vz(float *vz, float *temp, int cur, int zone, int lvl);
__global__ void smooth_fine_txz(float *txz, float *temp, int cur, int zone, int lvl);
__global__ void smooth_fine_sig(float *sig, float *temp, int cur, int zone, int lvl);
__global__ void smooth_fine_p(float *p, float *temp, int cur, int zone, int lvl);
__global__ void smooth_fine_rx(float *rx, float *temp, int cur, int zone, int lvl);
__global__ void smooth_fine_rz(float *rz, float *temp, int cur, int zone, int lvl);
__global__ void smooth_fine_rxz(float *rxz, float *temp, int cur, int zone, int lvl);

#endif