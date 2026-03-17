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
__global__ void update_stress_coarse(Core core, Model model, PsiVel psi_vel, int cur);
__global__ void update_velocity_coarse(Core core, Model model, PsiStr psi_str, int cur);
__global__ void apply_fluid_boundary_coarse(Core core, int cur);
__global__ void update_stress_fine(Core core, Model model, int cur, int zone);
__global__ void update_velocity_fine(Core core, Model model, int cur, int zone);
__global__ void apply_fluid_boundary_fine(Core core, int cur, int zone);

template<int LEVEL>
__global__ void smooth_fine_vx(float *vx, float *temp, int cur, int zone) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fines[zone].lenx - 1 || iz >= fines[zone].lenz) return;

    int idx_dst = IdxVxFi(ix, iz, 0, zone) - sum_offset_fine_vx[0];
    int idx_src = IdxVxFi(ix, iz, cur, zone);

    float sum = coeff[LEVEL][0] * vx[idx_src];
    float total_weight = coeff[LEVEL][0];

    // 正交邻点
    if (iz > 0) {
        sum += coeff[LEVEL][1] * vx[IdxVxFi(ix, iz-1, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }
    if (iz < fines[zone].lenz - 1) {
        sum += coeff[LEVEL][1] * vx[IdxVxFi(ix, iz+1, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }
    if (ix > 0) {
        sum += coeff[LEVEL][1] * vx[IdxVxFi(ix-1, iz, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }
    if (ix < fines[zone].lenx - 2) {
        sum += coeff[LEVEL][1] * vx[IdxVxFi(ix+1, iz, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }

    // 对角邻点
    if (ix > 0 && iz > 0) {
        sum += coeff[LEVEL][2] * vx[IdxVxFi(ix-1, iz-1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }
    if (ix < fines[zone].lenx - 2 && iz > 0) {
        sum += coeff[LEVEL][2] * vx[IdxVxFi(ix+1, iz-1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }
    if (ix > 0 && iz < fines[zone].lenz - 1) {
        sum += coeff[LEVEL][2] * vx[IdxVxFi(ix-1, iz+1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }
    if (ix < fines[zone].lenx - 2 && iz < fines[zone].lenz - 1) {
        sum += coeff[LEVEL][2] * vx[IdxVxFi(ix+1, iz+1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }

    temp[idx_dst] = sum / total_weight;
}

template<int LEVEL>
__global__ void smooth_fine_vz(float *vz, float *temp, int cur, int zone) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fines[zone].lenx || iz >= fines[zone].lenz - 1) return;

    int idx_dst = IdxVzFi(ix, iz, 0, zone) - sum_offset_fine_vz[0];
    int idx_src = IdxVzFi(ix, iz, cur, zone);

    float sum = coeff[LEVEL][0] * vz[idx_src];
    float total_weight = coeff[LEVEL][0];

    // 正交邻点
    if (iz > 0) {
        sum += coeff[LEVEL][1] * vz[IdxVzFi(ix, iz-1, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }
    if (iz < fines[zone].lenz - 2) {
        sum += coeff[LEVEL][1] * vz[IdxVzFi(ix, iz+1, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }
    if (ix > 0) {
        sum += coeff[LEVEL][1] * vz[IdxVzFi(ix-1, iz, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }
    if (ix < fines[zone].lenx - 1) {
        sum += coeff[LEVEL][1] * vz[IdxVzFi(ix+1, iz, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }

    // 对角邻点
    if (ix > 0 && iz > 0) {
        sum += coeff[LEVEL][2] * vz[IdxVzFi(ix-1, iz-1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }
    if (ix < fines[zone].lenx - 1 && iz > 0) {
        sum += coeff[LEVEL][2] * vz[IdxVzFi(ix+1, iz-1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }
    if (ix > 0 && iz < fines[zone].lenz - 2) {
        sum += coeff[LEVEL][2] * vz[IdxVzFi(ix-1, iz+1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }
    if (ix < fines[zone].lenx - 1 && iz < fines[zone].lenz - 2) {
        sum += coeff[LEVEL][2] * vz[IdxVzFi(ix+1, iz+1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }

    temp[idx_dst] = sum / total_weight;
}

template<int LEVEL>
__global__ void smooth_fine_sx(float *sx, float *temp, int cur, int zone) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fines[zone].lenx || iz >= fines[zone].lenz) return;

    int idx_dst = IdxSigFi(ix, iz, 0, zone) - sum_offset_fine_sig[0];
    int idx_src = IdxSigFi(ix, iz, cur, zone);

    float sum = coeff[LEVEL][0] * sx[idx_src];
    float total_weight = coeff[LEVEL][0];

    // 正交邻点
    if (iz > 0) {
        sum += coeff[LEVEL][1] * sx[IdxSigFi(ix, iz-1, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }
    if (iz < fines[zone].lenz - 1) {
        sum += coeff[LEVEL][1] * sx[IdxSigFi(ix, iz+1, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }
    if (ix > 0) {
        sum += coeff[LEVEL][1] * sx[IdxSigFi(ix-1, iz, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }
    if (ix < fines[zone].lenx - 1) {
        sum += coeff[LEVEL][1] * sx[IdxSigFi(ix+1, iz, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }

    // 对角邻点
    if (ix > 0 && iz > 0) {
        sum += coeff[LEVEL][2] * sx[IdxSigFi(ix-1, iz-1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }
    if (ix < fines[zone].lenx - 1 && iz > 0) {
        sum += coeff[LEVEL][2] * sx[IdxSigFi(ix+1, iz-1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }
    if (ix > 0 && iz < fines[zone].lenz - 1) {
        sum += coeff[LEVEL][2] * sx[IdxSigFi(ix-1, iz+1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }
    if (ix < fines[zone].lenx - 1 && iz < fines[zone].lenz - 1) {
        sum += coeff[LEVEL][2] * sx[IdxSigFi(ix+1, iz+1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }

    temp[idx_dst] = sum / total_weight;
}

template<int LEVEL>
__global__ void smooth_fine_sz(float *sz, float *temp, int cur, int zone) {
    // 与 sx 完全相同，只是数组名不同
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fines[zone].lenx || iz >= fines[zone].lenz) return;

    int idx_dst = IdxSigFi(ix, iz, 0, zone) - sum_offset_fine_sig[0];
    int idx_src = IdxSigFi(ix, iz, cur, zone);

    float sum = coeff[LEVEL][0] * sz[idx_src];
    float total_weight = coeff[LEVEL][0];

    if (iz > 0) {
        sum += coeff[LEVEL][1] * sz[IdxSigFi(ix, iz-1, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }
    if (iz < fines[zone].lenz - 1) {
        sum += coeff[LEVEL][1] * sz[IdxSigFi(ix, iz+1, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }
    if (ix > 0) {
        sum += coeff[LEVEL][1] * sz[IdxSigFi(ix-1, iz, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }
    if (ix < fines[zone].lenx - 1) {
        sum += coeff[LEVEL][1] * sz[IdxSigFi(ix+1, iz, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }

    if (ix > 0 && iz > 0) {
        sum += coeff[LEVEL][2] * sz[IdxSigFi(ix-1, iz-1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }
    if (ix < fines[zone].lenx - 1 && iz > 0) {
        sum += coeff[LEVEL][2] * sz[IdxSigFi(ix+1, iz-1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }
    if (ix > 0 && iz < fines[zone].lenz - 1) {
        sum += coeff[LEVEL][2] * sz[IdxSigFi(ix-1, iz+1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }
    if (ix < fines[zone].lenx - 1 && iz < fines[zone].lenz - 1) {
        sum += coeff[LEVEL][2] * sz[IdxSigFi(ix+1, iz+1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }

    temp[idx_dst] = sum / total_weight;
}

template<int LEVEL>
__global__ void smooth_fine_txz(float *txz, float *temp, int cur, int zone) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fines[zone].lenx - 1 || iz >= fines[zone].lenz - 1) return;

    int idx_dst = IdxTxzFi(ix, iz, 0, zone) - sum_offset_fine_txz[0];
    int idx_src = IdxTxzFi(ix, iz, cur, zone);

    float sum = coeff[LEVEL][0] * txz[idx_src];
    float total_weight = coeff[LEVEL][0];

    // 正交邻点
    if (iz > 0) {
        sum += coeff[LEVEL][1] * txz[IdxTxzFi(ix, iz-1, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }
    if (iz < fines[zone].lenz - 2) {
        sum += coeff[LEVEL][1] * txz[IdxTxzFi(ix, iz+1, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }
    if (ix > 0) {
        sum += coeff[LEVEL][1] * txz[IdxTxzFi(ix-1, iz, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }
    if (ix < fines[zone].lenx - 2) {
        sum += coeff[LEVEL][1] * txz[IdxTxzFi(ix+1, iz, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }

    // 对角邻点
    if (ix > 0 && iz > 0) {
        sum += coeff[LEVEL][2] * txz[IdxTxzFi(ix-1, iz-1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }
    if (ix < fines[zone].lenx - 2 && iz > 0) {
        sum += coeff[LEVEL][2] * txz[IdxTxzFi(ix+1, iz-1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }
    if (ix > 0 && iz < fines[zone].lenz - 2) {
        sum += coeff[LEVEL][2] * txz[IdxTxzFi(ix-1, iz+1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }
    if (ix < fines[zone].lenx - 2 && iz < fines[zone].lenz - 2) {
        sum += coeff[LEVEL][2] * txz[IdxTxzFi(ix+1, iz+1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }

    temp[idx_dst] = sum / total_weight;
}

template<int LEVEL>
__global__ void smooth_fine_p(float *p, float *temp, int cur, int zone) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fines[zone].lenx || iz >= fines[zone].lenz) return;

    int idx_dst = IdxSigFi(ix, iz, 0, zone) - sum_offset_fine_sig[0];
    int idx_src = IdxSigFi(ix, iz, cur, zone);

    float sum = coeff[LEVEL][0] * p[idx_src];
    float total_weight = coeff[LEVEL][0];

    if (iz > 0) {
        sum += coeff[LEVEL][1] * p[IdxSigFi(ix, iz-1, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }
    if (iz < fines[zone].lenz - 1) {
        sum += coeff[LEVEL][1] * p[IdxSigFi(ix, iz+1, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }
    if (ix > 0) {
        sum += coeff[LEVEL][1] * p[IdxSigFi(ix-1, iz, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }
    if (ix < fines[zone].lenx - 1) {
        sum += coeff[LEVEL][1] * p[IdxSigFi(ix+1, iz, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }

    if (ix > 0 && iz > 0) {
        sum += coeff[LEVEL][2] * p[IdxSigFi(ix-1, iz-1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }
    if (ix < fines[zone].lenx - 1 && iz > 0) {
        sum += coeff[LEVEL][2] * p[IdxSigFi(ix+1, iz-1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }
    if (ix > 0 && iz < fines[zone].lenz - 1) {
        sum += coeff[LEVEL][2] * p[IdxSigFi(ix-1, iz+1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }
    if (ix < fines[zone].lenx - 1 && iz < fines[zone].lenz - 1) {
        sum += coeff[LEVEL][2] * p[IdxSigFi(ix+1, iz+1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }

    temp[idx_dst] = sum / total_weight;
}

template<int LEVEL>
__global__ void smooth_fine_rp(float *rp, float *temp, int cur, int zone) {
    // 与 p 相同，只是数组名不同
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fines[zone].lenx || iz >= fines[zone].lenz) return;

    int idx_dst = IdxSigFi(ix, iz, 0, zone) - sum_offset_fine_sig[0];
    int idx_src = IdxSigFi(ix, iz, cur, zone);

    float sum = coeff[LEVEL][0] * rp[idx_src];
    float total_weight = coeff[LEVEL][0];

    if (iz > 0) {
        sum += coeff[LEVEL][1] * rp[IdxSigFi(ix, iz-1, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }
    if (iz < fines[zone].lenz - 1) {
        sum += coeff[LEVEL][1] * rp[IdxSigFi(ix, iz+1, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }
    if (ix > 0) {
        sum += coeff[LEVEL][1] * rp[IdxSigFi(ix-1, iz, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }
    if (ix < fines[zone].lenx - 1) {
        sum += coeff[LEVEL][1] * rp[IdxSigFi(ix+1, iz, cur, zone)];
        total_weight += coeff[LEVEL][1];
    }

    if (ix > 0 && iz > 0) {
        sum += coeff[LEVEL][2] * rp[IdxSigFi(ix-1, iz-1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }
    if (ix < fines[zone].lenx - 1 && iz > 0) {
        sum += coeff[LEVEL][2] * rp[IdxSigFi(ix+1, iz-1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }
    if (ix > 0 && iz < fines[zone].lenz - 1) {
        sum += coeff[LEVEL][2] * rp[IdxSigFi(ix-1, iz+1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }
    if (ix < fines[zone].lenx - 1 && iz < fines[zone].lenz - 1) {
        sum += coeff[LEVEL][2] * rp[IdxSigFi(ix+1, iz+1, cur, zone)];
        total_weight += coeff[LEVEL][2];
    }

    temp[idx_dst] = sum / total_weight;
}
#endif