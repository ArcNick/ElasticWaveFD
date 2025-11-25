#include "common.cuh"
#include "differentiate.cuh"
#include "update.cuh"
#include "cpml.cuh"
#include <cstdio>

__global__ void apply_source(
    Grid_Core::View gc, float src, int posx, int posz
) {
    gc.sx[posz * gc.nx + posx] += src;
    gc.sz[posz * gc.nx + posx] += src;
}

__global__ void update_stress(
    Grid_Core::View gc, Grid_Model::View gm, 
    Cpml::View cpml, float dx, float dz, float dt
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    int nx = gc.nx, nz = gc.nz;

    if (ix < 3 || ix >= nx - 4 || iz < 3 || iz >= nz - 4) {
        return;
    }

    float C11 = C11(ix, iz);
    float C13 = C13(ix, iz);
    float C33 = C33(ix, iz);
    // float C44 = C44(ix, iz);
    
    float dvx_dx = (ix <= 3 ? 0 : Dx_half_8th(gc.vx, ix, iz, nx - 1, dx));
    float dvz_dz = (iz <= 3 ? 0 : Dz_half_8th(gc.vz, ix, iz, nx, dz));
    float dvz_dx = Dx_int_8th(gc.vz, ix, iz, nx, dx);
    float dvx_dz = Dz_int_8th(gc.vx, ix, iz, nx - 1, dz);
    
    int pml_idx_x_int = get_cpml_idx_x_int(nx, ix, cpml.thickness);
    int pml_idx_z_int = get_cpml_idx_z_int(nz, iz, cpml.thickness);
    int pml_idx_x_half = get_cpml_idx_x_half(nx - 1, ix, cpml.thickness - 1);
    int pml_idx_z_half = get_cpml_idx_z_half(nz - 1, iz, cpml.thickness - 1);

    if (pml_idx_x_int < cpml.thickness) {
        dvz_dx = dvz_dx / cpml.kappa_int[pml_idx_x_int] + PVZ_X(ix, iz);
    }
    if (pml_idx_z_int < cpml.thickness) {
        dvx_dz = dvx_dz / cpml.kappa_int[pml_idx_z_int] + PVX_Z(ix, iz);
    }
    if (pml_idx_x_half < cpml.thickness - 1) {
        dvx_dx = dvx_dx / cpml.kappa_half[pml_idx_x_half] + PVX_X(ix, iz);
    }
    if (pml_idx_z_half < cpml.thickness - 1) {
        dvz_dz = dvz_dz / cpml.kappa_half[pml_idx_z_half] + PVZ_Z(ix, iz);
    }

    SX(ix, iz) += dt * (C11 * dvx_dx + C13 * dvz_dz);
    SZ(ix, iz) += dt * (C13 * dvx_dx + C33 * dvz_dz);

    // txz : (nx-1) × (nz-1)
    float C44_half = 0.25 * (
        C44(ix, iz) + C44(ix + 1, iz) + 
        C44(ix, iz + 1) + C44(ix + 1, iz + 1)
    );
    TXZ(ix, iz) += dt * C44_half * (dvz_dx + dvx_dz);
}

__global__ void update_velocity(
    Grid_Core::View gc, Grid_Model::View gm, 
    Cpml::View cpml, float dx, float dz, float dt
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    int nx = gc.nx, nz = gc.nz;

    if (ix < 3 || ix >= nx - 4 || iz < 3 || iz >= nz - 4) {
        return;
    }

    int pml_idx_x_int = get_cpml_idx_x_int(nx, ix, cpml.thickness);
    int pml_idx_z_int = get_cpml_idx_z_int(nz, iz, cpml.thickness);
    int pml_idx_x_half = get_cpml_idx_x_half(nx - 1, ix, cpml.thickness - 1);
    int pml_idx_z_half = get_cpml_idx_z_half(nz - 1, iz, cpml.thickness - 1);

    // vx : (nx-1) × nz
    float rho_half_x = (RHO(ix, iz) + RHO(ix + 1, iz)) * 0.5f;
    float dtxz_dz = (iz <= 3 ? 0 : Dz_half_8th(gc.txz, ix, iz, nx - 1, dz));
    float dsx_dx = Dx_int_8th(gc.sx, ix, iz, nx, dx);

    if (pml_idx_x_int < cpml.thickness) {
        dsx_dx = dsx_dx / cpml.kappa_int[pml_idx_x_int] + PSX_X(ix, iz);
    }
    if (pml_idx_z_half < cpml.thickness - 1) {
        dtxz_dz = dtxz_dz / cpml.kappa_half[pml_idx_z_half] + PTXZ_Z(ix, iz);
    }
    VX(ix, iz) += dt / rho_half_x * (dsx_dx + dtxz_dz);
    
    // vz : nx × (nz-1)
    float rho_half_z = (RHO(ix, iz) + RHO(ix, iz + 1)) * 0.5f;
    float dtxz_dx = (ix <= 3 ? 0 : Dx_half_8th(gc.txz, ix, iz, nx - 1, dx));
    float dsz_dz = Dz_int_8th(gc.sz, ix, iz, nx, dz);

    if (pml_idx_x_half < cpml.thickness - 1) {
        dtxz_dx = dtxz_dx / cpml.kappa_half[pml_idx_x_half] + PTXZ_X(ix, iz);
    }
    if (pml_idx_z_int < cpml.thickness) {
        dsz_dz = dsz_dz / cpml.kappa_int[pml_idx_z_int] + PSZ_Z(ix, iz);
    }
    VZ(ix, iz) += dt / rho_half_z * (dtxz_dx + dsz_dz);
}

__global__ void apply_free_boundary(Grid_Core::View gc) {
    int idx = threadIdx.x;
    int nx = gc.nx, nz = gc.nz;

    // x 在整网格点上
    if (3 <= idx && idx <= nx - 5) {
        SX(idx, 3) = SX(idx, nz - 5) = 0;
        SZ(idx, 3) = SZ(idx, nz - 5) = 0;
        VZ(idx, 4) = VZ(idx, nz - 5) = 0;
    }

    // x 在半网格点上
    if (4 <= idx && idx <= nx - 5) {
        VX(idx, 3) = VX(idx, nz - 5) = 0;
        TXZ(idx, 4) = TXZ(idx, nz - 5) = 0;
    }

    // z 在整网格点上
    if (3 <= idx && idx <= nz - 5) {
        SX(3, idx) = SX(nx - 5, idx) = 0;
        SZ(3, idx) = SZ(nx - 5, idx) = 0;
        VX(4, idx) = VZ(nx - 5, idx) = 0;
    }

    // z 在半网格点上
    if (4 <= idx && idx <= nz - 5) {
        VZ(3, idx) = VZ(nx - 5, idx) = 0;
        TXZ(4, idx) = TXZ(nx - 5, idx) = 0;
    }
}