#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "grid_manager.cuh"
#include "cpml.cuh"

__device__ int get_cpml_idx_x_int(int ix);
__device__ int get_cpml_idx_z_int(int iz);
__device__ int get_cpml_idx_x_half(int ix);
__device__ int get_cpml_idx_z_half(int iz);

__global__ void apply_source(Core core, float src, int cur);
__global__ void update_stress_coarse(Core core, Model model, PsiVel psi_vel, int cur);
__global__ void update_velocity_coarse(Core core, Model model, PsiStr psi_str, int cur);
__global__ void apply_fluid_boundary_coarse(Core core, int cur);
__global__ void update_stress_fine(Core core, Model model, int cur, int zone);
__global__ void update_velocity_fine(Core core, Model model, int cur, int zone);
__global__ void apply_fluid_boundary_fine(Core core, int cur, int zone);
#endif