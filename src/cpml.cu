#include "common.cuh"
#include "differentiate.cuh"
#include "cpml.cuh"
#include <cstdio>
#include <algorithm>
#include <cmath>

Cpml::Cpml(Params::View par, const char *file) {
    read(file);
    nx = par.nx;
    nz = par.nz;
    mem_location = DEVICE_MEM;
    L = thickness * par.dx;

    cudaMalloc(&alpha_int, thickness * sizeof(float));
    cudaMalloc(&damp_int, thickness * sizeof(float));
    cudaMalloc(&a_int, thickness * sizeof(float));
    cudaMalloc(&b_int, thickness * sizeof(float));
    cudaMalloc(&kappa_int, thickness * sizeof(float));
    cudaMalloc(&alpha_half, thickness * sizeof(float));
    cudaMalloc(&damp_half, thickness * sizeof(float));
    cudaMalloc(&a_half, thickness * sizeof(float));
    cudaMalloc(&b_half, thickness * sizeof(float));
    cudaMalloc(&kappa_half, thickness * sizeof(float));
    cudaMalloc(&psi_vx_x, (nx - 1) * nz * sizeof(float));
    cudaMalloc(&psi_vx_z, (nx - 1) * nz * sizeof(float));
    cudaMalloc(&psi_vz_x, nx * (nz - 1) * sizeof(float));
    cudaMalloc(&psi_vz_z, nx * (nz - 1) * sizeof(float));
    cudaMalloc(&psi_sx_x, nx * nz * sizeof(float));
    cudaMalloc(&psi_sx_z, nx * nz * sizeof(float));
    cudaMalloc(&psi_sz_x, nx * nz * sizeof(float));
    cudaMalloc(&psi_sz_z, nx * nz * sizeof(float));
    cudaMalloc(&psi_txz_x, (nx - 1) * (nz - 1) * sizeof(float));
    cudaMalloc(&psi_txz_z, (nx - 1) * (nz - 1) * sizeof(float));

    cudaMemset(alpha_int, 0, thickness * sizeof(float));
    cudaMemset(damp_int, 0, thickness * sizeof(float));
    cudaMemset(a_int, 0, thickness * sizeof(float));
    cudaMemset(b_int, 0, thickness * sizeof(float));
    cudaMemset(kappa_int, 0, thickness * sizeof(float));
    cudaMemset(alpha_half, 0, thickness * sizeof(float));
    cudaMemset(damp_half, 0, thickness * sizeof(float));
    cudaMemset(a_half, 0, thickness * sizeof(float));
    cudaMemset(b_half, 0, thickness * sizeof(float));
    cudaMemset(kappa_half, 0, thickness * sizeof(float));
    cudaMemset(psi_vx_x, 0, (nx - 1) * nz * sizeof(float));
    cudaMemset(psi_vx_z, 0, (nx - 1) * nz * sizeof(float));
    cudaMemset(psi_vz_x, 0, nx * (nz - 1) * sizeof(float));
    cudaMemset(psi_vz_z, 0, nx * (nz - 1) * sizeof(float));
    cudaMemset(psi_sx_x, 0, nx * nz * sizeof(float));
    cudaMemset(psi_sx_z, 0, nx * nz * sizeof(float));
    cudaMemset(psi_sz_x, 0, nx * nz * sizeof(float));
    cudaMemset(psi_sz_z, 0, nx * nz * sizeof(float));
    cudaMemset(psi_txz_x, 0, (nx - 1) * (nz - 1) * sizeof(float));
    cudaMemset(psi_txz_z, 0, (nx - 1) * (nz - 1) * sizeof(float));

    init(par);
}

Cpml::~Cpml() {
    if (alpha_int) {
        cudaFree(alpha_int);
        alpha_int = nullptr;
    }
    if (damp_int) {
        cudaFree(damp_int);
        damp_int = nullptr;
    }
    if (a_int) {
        cudaFree(a_int);
        a_int = nullptr;
    }
    if (b_int) {
        cudaFree(b_int);
        b_int = nullptr;
    }
    if (kappa_int) {
        cudaFree(kappa_int);
        kappa_int = nullptr;
    }
    if (alpha_half) {
        cudaFree(alpha_half);
        alpha_half = nullptr;
    }
    if (damp_half) {
        cudaFree(damp_half);
        damp_half = nullptr;
    }
    if (a_half) {
        cudaFree(a_half);
        a_half = nullptr;
    }
    if (b_half) {
        cudaFree(b_half);
        b_half = nullptr;
    }
    if (kappa_half) {
        cudaFree(kappa_half);
        kappa_half = nullptr;
    }
    if (psi_vx_x) {
        cudaFree(psi_vx_x);
        psi_vx_x = nullptr;
    }
    if (psi_vx_z) {
        cudaFree(psi_vx_z);
        psi_vx_z = nullptr;
    }
    if (psi_vz_x) {
        cudaFree(psi_vz_x);
        psi_vz_x = nullptr;
    }
    if (psi_vz_z) {
        cudaFree(psi_vz_z);
        psi_vz_z = nullptr;
    }
    if (psi_sx_x) {
        cudaFree(psi_sx_x);
        psi_sx_x = nullptr;
    }
    if (psi_sx_z) {
        cudaFree(psi_sx_z);
        psi_sx_z = nullptr;
    }
    if (psi_sz_x) {
        cudaFree(psi_sz_x);
        psi_sz_x = nullptr;
    }
    if (psi_sz_z) {
        cudaFree(psi_sz_z);
        psi_sz_z = nullptr;
    }
    if (psi_txz_x) {
        cudaFree(psi_txz_x);
        psi_txz_x = nullptr;
    }
    if (psi_txz_z) {
        cudaFree(psi_txz_z);
        psi_txz_z = nullptr;
    }
}

void Cpml::read(const char *file) {
    FILE *fp = fopen(file, "r");
    if (fp == nullptr) {
        printf("Failed to open CPML file %s\n", file);
        exit(1);
    }
    fscanf(fp, "thickness = %d\n", &thickness);
    fscanf(fp, "N = %f\n", &N);
    fscanf(fp, "cp_max = %f\n", &cp_max);
    fscanf(fp, "Rc = %f\n", &Rc);
    fscanf(fp, "kappa0 = %f\n", &kappa0);

    printf("CPML parameters loaded.\n");
    fclose(fp);
}

__global__ void cpml_init_params(
    Cpml::View cpml, float dx, float dz, float dt
) {
    int i = threadIdx.x;
    if (i < cpml.thickness) {
        // 整网格点
        float x_int = 1.0 * i / cpml.thickness;
        cpml.damp_int[i] = cpml.damp0 * powf(x_int, cpml.N);
        cpml.alpha_int[i] = cpml.alpha0 * (1.0f - x_int);
        cpml.kappa_int[i] = 1.0f + (cpml.kappa0 - 1.0f) * powf(x_int, cpml.N);

        cpml.b_int[i] = exp(
            -(cpml.damp_int[i] / cpml.kappa_int[i] + cpml.alpha_int[i]) * dt
        );
        cpml.a_int[i] = cpml.damp_int[i] * (cpml.b_int[i] - 1) / (
            cpml.kappa_int[i] * (cpml.damp_int[i] + cpml.kappa_int[i] * cpml.alpha_int[i])
        );
    }
    if (i < cpml.thickness - 1) {
        // 半网格点
        float x_half = (i + 0.5f) / (cpml.thickness);
        cpml.damp_half[i] = cpml.damp0 * powf(x_half, cpml.N);
        cpml.alpha_half[i] = cpml.alpha0 * (1.0f - x_half);
        cpml.kappa_half[i] = 1.0f + (cpml.kappa0 - 1.0f) * powf(x_half, cpml.N);

        cpml.b_half[i] = exp(
            -(cpml.damp_half[i] / cpml.kappa_half[i] + cpml.alpha_half[i]) * dt
        );
        cpml.a_half[i] = cpml.damp_half[i] * (cpml.b_half[i] - 1) / (
            cpml.kappa_half[i] * (cpml.damp_half[i] + cpml.kappa_half[i] * cpml.alpha_half[i])
        );
    }
}

void Cpml::init(Params::View par) {
    damp0 = -(N + 1) * cp_max * log(Rc) / (2 * L);
    alpha0 = M_PI * par.fpeak;
    cpml_init_params<<<1, thickness>>>(
        view(), par.dx, par.dz, par.dt
    );
}

__device__ int get_cpml_idx_x_int(int lx, int ix, int thickness) {
    int res = -1;
    int arr[] = {ix - 4, lx - 1 - ix - 5};
    for (int i = 0; i < 2; i++) {
        if (0 <= arr[i] && arr[i] < thickness) {
            res = arr[i];
        }
    }
    return thickness - 1 - res;
}
__device__ int get_cpml_idx_z_int(int lz, int iz, int thickness) {
    int res = -1;
    int arr[] = {iz - 4, lz - 1 - iz - 5};
    for (int i = 0; i < 2; i++) {
        if (0 <= arr[i] && arr[i] < thickness) {
            res = arr[i];
        }
    }
    return thickness - 1 - res;
}

__device__ int get_cpml_idx_x_half(int lx, int ix, int thickness) {
    int res = -1;
    int arr[] = {ix - 5, lx - 1 - ix - 4};
    for (int i = 0; i < 2; i++) {
        if (0 <= arr[i] && arr[i] < thickness) {
            res = arr[i];
        }
    }
    return thickness - 1 - res;
}
__device__ int get_cpml_idx_z_half(int lz, int iz, int thickness) {
    int res = -1;
    int arr[] = {iz - 5, lz - 1 - iz - 4};
    for (int i = 0; i < 2; i++) {
        if (0 <= arr[i] && arr[i] < thickness) {
            res = arr[i];
        }
    }
    return thickness - 1 - res;    
}

__global__ void cpml_update_psi_vel(
    Grid_Core::View gc, Cpml::View cpml, float dx, float dz, float dt, int cur
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    int nx = cpml.nx, nz = cpml.nz;
    int thickness = cpml.thickness;

    if (ix < 3 || ix >= nx - 4 || iz < 3 || iz >= nz - 4) {
        return;
    }
    int pml_idx_x_int = get_cpml_idx_x_int(nx, ix, thickness);
    int pml_idx_z_int = get_cpml_idx_z_int(nz, iz, thickness);
    int pml_idx_x_half = get_cpml_idx_x_half(nx - 1, ix, thickness - 1);
    int pml_idx_z_half = get_cpml_idx_z_half(nz - 1, iz, thickness - 1);

    if (pml_idx_x_int < thickness) {
        float a_int = cpml.a_int[pml_idx_x_int];
        float b_int = cpml.b_int[pml_idx_x_int];
        float dvz_dx = Dx_int_8th(gc.vz[cur], ix, iz, nx, dx);
        PVZ_X(ix, iz) = b_int * PVZ_X(ix, iz) + a_int * dvz_dx;
    }
    if (pml_idx_z_int < thickness) {
        float a_int = cpml.a_int[pml_idx_z_int];
        float b_int = cpml.b_int[pml_idx_z_int];
        float dvx_dz = Dz_int_8th(gc.vx[cur], ix, iz, nx - 1, dz);
        PVX_Z(ix, iz) = b_int * PVX_Z(ix, iz) + a_int * dvx_dz;
    } 
    if (pml_idx_x_half < thickness - 1) {
        float a_half = cpml.a_half[pml_idx_x_half];
        float b_half = cpml.b_half[pml_idx_x_half];
        float dvx_dx = (ix <= 3 ? 0 : Dx_half_8th(gc.vx[cur], ix, iz, nx - 1, dx));
        PVX_X(ix, iz) = b_half * PVX_X(ix, iz) + a_half * dvx_dx;
    }
    if (pml_idx_z_half < thickness - 1) {
        float a_half = cpml.a_half[pml_idx_z_half];
        float b_half = cpml.b_half[pml_idx_z_half];
        float dvz_dz = (iz <= 3 ? 0 : Dz_half_8th(gc.vz[cur], ix, iz, nx, dz));
        PVZ_Z(ix, iz) = b_half * PVZ_Z(ix, iz) + a_half * dvz_dz;
    }
}

__global__ void cpml_update_psi_stress(
    Grid_Core::View gc, Cpml::View cpml, float dx, float dz, float dt, int cur
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    int nx = cpml.nx, nz = cpml.nz;
    int thickness = cpml.thickness;

    if (ix < 3 || ix >= nx - 4 || iz < 3 || iz >= nz - 4) {
        return;
    }

    int pml_idx_x_int = get_cpml_idx_x_int(nx, ix, thickness);
    int pml_idx_z_int = get_cpml_idx_z_int(nz, iz, thickness);
    int pml_idx_x_half = get_cpml_idx_x_half(nx - 1, ix, thickness - 1);
    int pml_idx_z_half = get_cpml_idx_z_half(nz - 1, iz, thickness - 1);

    if (pml_idx_x_int < thickness) { 
        float a_int = cpml.a_int[pml_idx_x_int];
        float b_int = cpml.b_int[pml_idx_x_int];
        float dsx_dx = Dx_int_8th(gc.sx[cur], ix, iz, nx, dx);
        float dsz_dx = Dx_int_8th(gc.sz[cur], ix, iz, nx, dx);
        PSX_X(ix, iz) = b_int * PSX_X(ix, iz) + a_int * dsx_dx;
        PSZ_X(ix, iz) = b_int * PSZ_X(ix, iz) + a_int * dsz_dx;
    }
    if (pml_idx_z_int < thickness) {
        float a_int = cpml.a_int[pml_idx_z_int];
        float b_int = cpml.b_int[pml_idx_z_int];
        float dsx_dz = Dz_int_8th(gc.sx[cur], ix, iz, nx, dz);
        float dsz_dz = Dz_int_8th(gc.sz[cur], ix, iz, nx, dz);
        PSX_Z(ix, iz) = b_int * PSX_Z(ix, iz) + a_int * dsx_dz;
        PSZ_Z(ix, iz) = b_int * PSZ_Z(ix, iz) + a_int * dsz_dz;
    }
    if (pml_idx_x_half < thickness - 1) {
        float a_half = cpml.a_half[pml_idx_x_half];
        float b_half = cpml.b_half[pml_idx_x_half];
        float dtxz_dx = (ix <= 3 ? 0 : Dx_half_8th(gc.txz[cur], ix, iz, nx - 1, dx));
        PTXZ_X(ix, iz) = b_half * PTXZ_X(ix, iz) + a_half * dtxz_dx;
    }
    if (pml_idx_z_half < thickness - 1) {
        float a_half = cpml.a_half[pml_idx_z_half];
        float b_half = cpml.b_half[pml_idx_z_half];
        float dtxz_dz = (iz <= 3 ? 0 : Dz_half_8th(gc.txz[cur], ix, iz, nx - 1, dz));
        PTXZ_Z(ix, iz) = b_half * PTXZ_Z(ix, iz) + a_half * dtxz_dz;
    }
}