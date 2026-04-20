#include "kernels.cuh"
#include "fd_stencil.cuh"

extern __constant__ int posx_d, posz_d;
extern __constant__ float dt_d;

__constant__ float coeff[5][3] = {
    {0, 0, 0},
    {0.9, 0.02, 0.005},
    {0.8, 0.04, 0.010},
    {0.5, 0.1, 0.025},
    {0.25, 0.125, 1 / 16.0}
};

__device__ int get_cpml_idx_x_int(int ix) {
    if (ix - 3 >= 0 && ix - 3 < thickness_d) {
        return thickness_d - ix + 2;
    } else if (nx - ix - 5 >= 0 && nx - ix - 5 < thickness_d) {
        return thickness_d - nx + ix + 4;
    } else {
        return thickness_d;
    }
}

__device__ int get_cpml_idx_z_int(int iz) {
    if (iz - 3 >= 0 && iz - 3 < thickness_d) {
        return thickness_d - iz + 2;
    } else if (nz - iz - 5 >= 0 && nz - iz - 5 < thickness_d) {
        return thickness_d - nz + iz + 4;
    } else {
        return thickness_d;
    }
}

__device__ int get_cpml_idx_x_half(int ix) {
    if (ix - 4 >= 0 && ix - 3 < thickness_d) {
        return thickness_d - ix + 2;
    } else if (nx - ix - 5 >= 0 && nx - ix - 4 < thickness_d) {
        return thickness_d - nx + ix + 3;
    } else {
        return thickness_d - 1;
    }
}

__device__ int get_cpml_idx_z_half(int iz) {
    if (iz - 4 >= 0 && iz - 3 < thickness_d) {
        return thickness_d - iz + 2;
    } else if (nz - iz - 5 >= 0 && nz - iz - 4 < thickness_d) {
        return thickness_d - nz + iz + 3;
    } else {
        return thickness_d - 1;
    }
}

__global__ void apply_source(Core core, float src, int cur) {
    core.sx[posz_d * nx + posx_d + cur * offset_sig_all] += src * dt_d;
    core.sz[posz_d * nx + posx_d + cur * offset_sig_all] += src * dt_d;
}

__global__ void update_sigma_coarse(Core core, Model model, PsiVel psi_vel, int cur, int it) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < 3 || ix >= nx - 4 || iz < 3 || iz >= nz - 4) {
        return;
    }
    switch (MAT(iz * nx + ix)) {
    case SOLID:
        if (tex1Dfetch<int>(sig_mask, iz * nx + ix) == -1) {
            float dvx_dx = 0;
            float dvz_dz = 0;
            if (ix > 3) {
                dvx_dx = dvx_dx_coarse(core.vx, ix, iz, cur ^ 1);
            }
            if (iz > 3) {
                dvz_dz = dvz_dz_coarse(core.vz, ix, iz, cur ^ 1);
            }

            int pml_idx;
            // pml_idx_x_half
            pml_idx = get_cpml_idx_x_half(ix);
            if (pml_idx < thickness_d - 1) {
                psi_vel.psi_vx_x[iz * (nx - 1) + ix] = (
                    + b_half_d[pml_idx] * psi_vel.psi_vx_x[iz * (nx - 1) + ix]
                    + a_half_d[pml_idx] * dvx_dx
                );
                dvx_dx = dvx_dx / kappa_half_d[pml_idx] + psi_vel.psi_vx_x[iz * (nx - 1) + ix];
            }

            // pml_idx_z_half
            pml_idx = get_cpml_idx_z_half(iz);
            if (pml_idx < thickness_d - 1) {
                psi_vel.psi_vz_z[iz * nx + ix] = (
                    + b_half_d[pml_idx] * psi_vel.psi_vz_z[iz * nx + ix]
                    + a_half_d[pml_idx] * dvz_dz
                );
                dvz_dz = dvz_dz / kappa_half_d[pml_idx] + psi_vel.psi_vz_z[iz * nx + ix];
            }

            core.sx[IdxSigCo(ix, iz, cur)] = core.sx[IdxSigCo(ix, iz, cur ^ 1)] + dt_d * (
                + model.C11[IdxSigCo(ix, iz, 0)] * dvx_dx 
                + model.C13[IdxSigCo(ix, iz, 0)] * dvz_dz
            );
            core.sz[IdxSigCo(ix, iz, cur)] = core.sz[IdxSigCo(ix, iz, cur ^ 1)] + dt_d * (
                + model.C13[IdxSigCo(ix, iz, 0)] * dvx_dx
                + model.C33[IdxSigCo(ix, iz, 0)] * dvz_dz
            );
        }
        break;

    case FLUID:
        if (tex1Dfetch<int>(sig_mask, iz * nx + ix) == -1) {
            float dvx_dx = 0;
            float dvz_dz = 0;
            if (ix > 3) {
                dvx_dx = dvx_dx_coarse(core.vx, ix, iz, cur ^ 1);
            }
            if (iz > 3) {
                dvz_dz = dvz_dz_coarse(core.vz, ix, iz, cur ^ 1);
            }
            
            core.p[IdxSigCo(ix, iz, cur)] = core.p[IdxSigCo(ix, iz, cur ^ 1)] + dt_d * (
                + model.C11[IdxSigCo(ix, iz, 0)] * (dvx_dx + dvz_dz)
            );
            core.sx[IdxSigCo(ix, iz, cur)] = -core.p[IdxSigCo(ix, iz, cur)] + (
                model.zeta[IdxSigCo(ix, iz, 0)] * (dvx_dx + dvz_dz)
            );
            core.sz[IdxSigCo(ix, iz, cur)] = -core.p[IdxSigCo(ix, iz, cur)] + (
                model.zeta[IdxSigCo(ix, iz, 0)] * (dvx_dx + dvz_dz)
            );
        }
        break;

    case VESOLID:
        if (tex1Dfetch<int>(sig_mask, iz * nx + ix) == -1) {
            float dvx_dx = 0;
            float dvz_dz = 0;
            if (ix > 3) {
                dvx_dx = dvx_dx_coarse(core.vx, ix, iz, cur ^ 1);
            }
            if (iz > 3) {
                dvz_dz = dvz_dz_coarse(core.vz, ix, iz, cur ^ 1);
            }
            
            // 粘弹性区不设置在边界区
            core.sx[IdxSigCo(ix, iz, cur)] = core.sx[IdxSigCo(ix, iz, cur ^ 1)] + dt_d * (
                + model.C11[IdxSigCo(ix, iz, 0)] * (
                    1 + model.taup[IdxSigCo(ix, iz, 0)]
                ) * dvx_dx 
                + model.C13[IdxSigCo(ix, iz, 0)] * (
                    1 + model.taup[IdxSigCo(ix, iz, 0)]
                ) * dvz_dz
                + model.inv_tsig[IdxSigCo(ix, iz, 0)] * core.rx[IdxSigCo(ix, iz, cur ^ 1)]
            );
            core.sz[IdxSigCo(ix, iz, cur)] = core.sz[IdxSigCo(ix, iz, cur ^ 1)] + dt_d * (
                + model.C13[IdxSigCo(ix, iz, 0)] * (
                    1 + model.taup[IdxSigCo(ix, iz, 0)]
                ) * dvx_dx
                + model.C33[IdxSigCo(ix, iz, 0)] * (
                    1 + model.taup[IdxSigCo(ix, iz, 0)]
                ) * dvz_dz
                + model.inv_tsig[IdxSigCo(ix, iz, 0)] * core.rz[IdxSigCo(ix, iz, cur ^ 1)]
            );
            core.rx[IdxSigCo(ix, iz, cur)] = core.rx[IdxSigCo(ix, iz, cur ^ 1)] + dt_d * (
                - model.inv_tsig[IdxSigCo(ix, iz, 0)] * core.rx[IdxSigCo(ix, iz, cur ^ 1)]
                - model.C11[IdxSigCo(ix, iz, 0)] * model.taup[IdxSigCo(ix, iz, 0)] * dvx_dx
                - model.C13[IdxSigCo(ix, iz, 0)] * model.taup[IdxSigCo(ix, iz, 0)] * dvz_dz
            );
            core.rz[IdxSigCo(ix, iz, cur)] = core.rz[IdxSigCo(ix, iz, cur ^ 1)] + dt_d * (
                - model.inv_tsig[IdxSigCo(ix, iz, 0)] * core.rz[IdxSigCo(ix, iz, cur ^ 1)]
                - model.C13[IdxSigCo(ix, iz, 0)] * model.taup[IdxSigCo(ix, iz, 0)] * dvx_dx
                - model.C33[IdxSigCo(ix, iz, 0)] * model.taup[IdxSigCo(ix, iz, 0)] * dvz_dz
            );
        }
        break;
    }
}

__global__ void update_tau_coarse(Core core, Model model, PsiVel psi_vel, int cur, int it) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < 3 || ix >= nx - 4 || iz < 3 || iz >= nz - 4) {
        return;
    }
    auto is_fluid = [&](int x, int z) -> bool {
        return MAT(IdxSigCo(x, z, 0)) == FLUID || MAT(IdxSigCo(x + 1, z, 0)) == FLUID || 
               MAT(IdxSigCo(x, z + 1, 0)) == FLUID || MAT(IdxSigCo(x + 1, z + 1, 0)) == FLUID;
    };
    auto is_vesolid_txz = [&](int x, int z) -> bool {
        return MAT(IdxSigCo(x, z, 0)) == VESOLID || MAT(IdxSigCo(x + 1, z, 0)) == VESOLID || 
               MAT(IdxSigCo(x, z + 1, 0)) == VESOLID || MAT(IdxSigCo(x + 1, z + 1, 0)) == VESOLID;
    };
    if (is_fluid(ix, iz)) {
        return;
    }

    switch (is_vesolid_txz(ix, iz)) {
    case 0:
        if (tex1Dfetch<int>(txz_mask, iz * (nx - 1) + ix) == -1) {
            float dvz_dx = 0;
            float dvx_dz = 0;
            dvx_dz = dvx_dz_coarse(core.vx, ix, iz, cur ^ 1);
            dvz_dx = dvz_dx_coarse(core.vz, ix, iz, cur ^ 1);

            int pml_idx;
            // pml_idx_x_int
            pml_idx = get_cpml_idx_x_int(ix);
            if (pml_idx < thickness_d) {
                psi_vel.psi_vz_x[iz * nx + ix] = (
                    + b_int_d[pml_idx] * psi_vel.psi_vz_x[iz * nx + ix] 
                    + a_int_d[pml_idx] * dvz_dx
                );
                dvz_dx = dvz_dx / kappa_int_d[pml_idx] + psi_vel.psi_vz_x[iz * nx + ix];
            }

            // pml_idx_z_int
            pml_idx = get_cpml_idx_z_int(iz);
            if (pml_idx < thickness_d) {
                psi_vel.psi_vx_z[iz * (nx - 1) + ix] = (
                    + b_int_d[pml_idx] * psi_vel.psi_vx_z[iz * (nx - 1) + ix]
                    + a_int_d[pml_idx] * dvx_dz
                );
                dvx_dz = dvx_dz / kappa_int_d[pml_idx] + psi_vel.psi_vx_z[iz * (nx - 1) + ix];
            }

            core.txz[IdxTxzCo(ix, iz, cur)] = (
                + core.txz[IdxTxzCo(ix, iz, cur ^ 1)]
                + dt_d * samp_C55_coarse(model.C55, ix, iz) * (dvz_dx + dvx_dz)
            );
        }
        break;

    case 1:
        if (tex1Dfetch<int>(txz_mask, iz * (nx - 1) + ix) == -1) {
            float dvz_dx = 0;
            float dvx_dz = 0;
            dvx_dz = dvx_dz_coarse(core.vx, ix, iz, cur ^ 1);
            dvz_dx = dvz_dx_coarse(core.vz, ix, iz, cur ^ 1);

            core.txz[IdxTxzCo(ix, iz, cur)] = core.txz[IdxTxzCo(ix, iz, cur ^ 1)] + dt_d * (
                + samp_C55_coarse(model.C55, ix, iz) * (dvz_dx + dvx_dz) * (
                    1 + samp_taus_coarse(model.taus, ix, iz)
                ) 
                + model.inv_tsig[IdxTxzCo(ix, iz, 0)] * core.rxz[IdxTxzCo(ix, iz, cur ^ 1)]
            );
            core.rxz[IdxTxzCo(ix, iz, cur)] = core.rxz[IdxTxzCo(ix, iz, cur ^ 1)] + dt_d * (
                - model.inv_tsig[IdxTxzCo(ix, iz, 0)] * core.rxz[IdxTxzCo(ix, iz, cur ^ 1)]
                - model.C55[IdxTxzCo(ix, iz, 0)] * model.taus[IdxTxzCo(ix, iz, 0)] * (dvz_dx + dvx_dz)
            );
        }
        break;
    } 
}

__global__ void apply_fluid_boundary_coarse(Core core, int cur) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < 3 || ix >= nx - 4 || iz < 3 || iz >= nz - 4) {
        return;
    }
    if (MAT(iz * nx + ix) == FLUID) {
        core.txz[IdxTxzCo(ix, iz, cur)] = 0;
        core.txz[IdxTxzCo(max(0, ix - 1), iz, cur)] = 0;
        core.txz[IdxTxzCo(ix, max(0, iz - 1), cur)] = 0;
        core.txz[IdxTxzCo(max(0, ix - 1), max(0, iz - 1), cur)] = 0;

        core.rxz[IdxTxzCo(ix, iz, cur)] = 0;
        core.rxz[IdxTxzCo(max(0, ix - 1), iz, cur)] = 0;
        core.rxz[IdxTxzCo(ix, max(0, iz - 1), cur)] = 0;
        core.rxz[IdxTxzCo(max(0, ix - 1), max(0, iz - 1), cur)] = 0;
    }
}

__global__ void update_velocity_coarse(Core core, Model model, PsiStr psi_str, int cur, int it) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < 3 || ix >= nx - 4 || iz < 3 || iz >= nz - 4) {
        return;
    }

    if (tex1Dfetch<int>(vx_mask, iz * (nx - 1) + ix) == -1) {
        float dsx_dx = dsx_dx_coarse(core.sx, ix, iz, cur);
        float dtxz_dz = 0;   // HALO 或者流体内部默认为0
        // 1. 固体内部以及流固边界，使用txz的导数
        // 2. 流体内部，不使用txz的导数
        if (iz > 3 && MAT(iz * nx + ix) + MAT(iz * nx + ix + 1) <= 3) {
            dtxz_dz = dtxz_dz_coarse(core.txz, ix, iz, cur);
        }

        int pml_idx;
        // pml_idx_x_int
        pml_idx = get_cpml_idx_x_int(ix);
        if (pml_idx < thickness_d) {
            psi_str.psi_sx_x[iz * nx + ix] = (
                + b_int_d[pml_idx] * psi_str.psi_sx_x[iz * nx + ix]
                + a_int_d[pml_idx] * dsx_dx
            );
            dsx_dx = dsx_dx / kappa_int_d[pml_idx] + psi_str.psi_sx_x[iz * nx + ix];
        }
        // pml_idx_z_half
        pml_idx = get_cpml_idx_z_half(iz);
        if (pml_idx < thickness_d - 1) {
            psi_str.psi_txz_z[iz * (nx - 1) + ix] = (
                + b_half_d[pml_idx] * psi_str.psi_txz_z[iz * (nx - 1) + ix]
                + a_half_d[pml_idx] * dtxz_dz
            );
            dtxz_dz = dtxz_dz / kappa_half_d[pml_idx] + psi_str.psi_txz_z[iz * (nx - 1) + ix];
        }

        core.vx[IdxVxCo(ix, iz, cur)] = core.vx[IdxVxCo(ix, iz, cur ^ 1)] + (
            dt_d / samp_rho_x_coarse(model.rho, ix, iz) * (dsx_dx + dtxz_dz)
        );
    }

    if (tex1Dfetch<int>(vz_mask, iz * nx + ix) == -1) {
        float dsz_dz = dsz_dz_coarse(core.sz, ix, iz, cur);
        float dtxz_dx = 0; // HALO 或者流体内部默认为0
        if (ix > 3 && MAT(iz * nx + ix) + MAT((iz + 1) * nx + ix) <= 3) {
            dtxz_dx = dtxz_dx_coarse(core.txz, ix, iz, cur);
        }

        int pml_idx;
        // pml_idx_z_int
        pml_idx = get_cpml_idx_z_int(iz);
        if (pml_idx < thickness_d) {
            psi_str.psi_sz_z[iz * nx + ix] = (
                + b_int_d[pml_idx] * psi_str.psi_sz_z[iz * nx + ix]
                + a_int_d[pml_idx] * dsz_dz
            );
            dsz_dz = dsz_dz / kappa_int_d[pml_idx] + psi_str.psi_sz_z[iz * nx + ix];
        }

        // pml_idx_x_half
        pml_idx = get_cpml_idx_x_half(ix);
        if (pml_idx < thickness_d - 1) {
            psi_str.psi_txz_x[iz * (nx - 1) + ix] = (
                + b_half_d[pml_idx] * psi_str.psi_txz_x[iz * (nx - 1) + ix]
                + a_half_d[pml_idx] * dtxz_dx
            );
            dtxz_dx = dtxz_dx / kappa_half_d[pml_idx] + psi_str.psi_txz_x[iz * (nx - 1) + ix];
        }

        core.vz[IdxVzCo(ix, iz, cur)] = core.vz[IdxVzCo(ix, iz, cur ^ 1)] + (
            dt_d / samp_rho_z_coarse(model.rho, ix, iz) * (dtxz_dx + dsz_dz)
        );
    }
}

__global__ void update_sigma_fine(Core core, Model model, int cur, int zone) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= fines[zone].lenx || iz >= fines[zone].lenz) {
        return;
    }
    switch (MAT(IdxSigFi(ix, iz, zone, 0))) {
    case SOLID: {
        float dvx_dx = dvx_dx_8th(core.vx, ix, iz, zone, cur ^ 1);
        float dvz_dz = dvz_dz_8th(core.vz, ix, iz, zone, cur ^ 1);
        core.sx[IdxSigFi(ix, iz, zone, cur)] = core.sx[IdxSigFi(ix, iz, zone, cur ^ 1)] + (
            dt_d * (
                + model.C11[IdxSigFi(ix, iz, zone, 0)] * dvx_dx
                + model.C13[IdxSigFi(ix, iz, zone, 0)] * dvz_dz
            )
        );
        core.sz[IdxSigFi(ix, iz, zone, cur)] = core.sz[IdxSigFi(ix, iz, zone, cur ^ 1)] + (
            dt_d * (
                + model.C13[IdxSigFi(ix, iz, zone, 0)] * dvx_dx
                + model.C33[IdxSigFi(ix, iz, zone, 0)] * dvz_dz
            )
        );
        break;
    }
    case FLUID: {
        float dvx_dx = dvx_dx_8th(core.vx, ix, iz, zone, cur ^ 1);
        float dvz_dz = dvz_dz_8th(core.vz, ix, iz, zone, cur ^ 1);
        core.p[IdxSigFi(ix, iz, zone, cur)] = core.p[IdxSigFi(ix, iz, zone, cur ^ 1)] + dt_d * (
            - model.C11[IdxSigFi(ix, iz, zone, 0)] * (dvx_dx + dvz_dz)
        );
        core.sx[IdxSigFi(ix, iz, zone, cur)] = -core.p[IdxSigFi(ix, iz, zone, cur)] + (
            model.zeta[IdxSigFi(ix, iz, zone, 0)] * (dvx_dx + dvz_dz)
        );
        core.sz[IdxSigFi(ix, iz, zone, cur)] = -core.p[IdxSigFi(ix, iz, zone, cur)] + (
            model.zeta[IdxSigFi(ix, iz, zone, 0)] * (dvx_dx + dvz_dz)
        );

        break;
    }
    case VESOLID: {
        float dvx_dx = dvx_dx_8th(core.vx, ix, iz, zone, cur ^ 1);
        float dvz_dz = dvz_dz_8th(core.vz, ix, iz, zone, cur ^ 1);
        core.sx[IdxSigFi(ix, iz, zone, cur)] = core.sx[IdxSigFi(ix, iz, zone, cur ^ 1)] + dt_d * (
            + model.C11[IdxSigFi(ix, iz, zone, 0)] * (1 + model.taup[IdxSigFi(ix, iz, zone, 0)]) * dvx_dx
            + model.C13[IdxSigFi(ix, iz, zone, 0)] * (1 + model.taup[IdxSigFi(ix, iz, zone, 0)]) * dvz_dz
            + model.inv_tsig[IdxSigFi(ix, iz, zone, 0)] * core.rx[IdxSigFi(ix, iz, zone, cur ^ 1)]
        );
        core.sz[IdxSigFi(ix, iz, zone, cur)] = core.sz[IdxSigFi(ix, iz, zone, cur ^ 1)] + dt_d * (
            + model.C13[IdxSigFi(ix, iz, zone, 0)] * (1 + model.taup[IdxSigFi(ix, iz, zone, 0)]) * dvx_dx
            + model.C33[IdxSigFi(ix, iz, zone, 0)] * (1 + model.taup[IdxSigFi(ix, iz, zone, 0)]) * dvz_dz
            + model.inv_tsig[IdxSigFi(ix, iz, zone, 0)] * core.rz[IdxSigFi(ix, iz, zone, cur ^ 1)]
        );
        core.rx[IdxSigFi(ix, iz, zone, cur)] = core.rx[IdxSigFi(ix, iz, zone, cur ^ 1)] + dt_d * (
            - model.inv_tsig[IdxSigFi(ix, iz, zone, 0)] * core.rx[IdxSigFi(ix, iz, zone, cur ^ 1)]
            - model.C11[IdxSigFi(ix, iz, zone, 0)] * model.taup[IdxSigFi(ix, iz, zone, 0)] * dvx_dx
            - model.C13[IdxSigFi(ix, iz, zone, 0)] * model.taup[IdxSigFi(ix, iz, zone, 0)] * dvz_dz
        );
        core.rz[IdxSigFi(ix, iz, zone, cur)] = core.rz[IdxSigFi(ix, iz, zone, cur ^ 1)] + dt_d * (
            - model.inv_tsig[IdxSigFi(ix, iz, zone, 0)] * core.rz[IdxSigFi(ix, iz, zone, cur ^ 1)]
            - model.C13[IdxSigFi(ix, iz, zone, 0)] * model.taup[IdxSigFi(ix, iz, zone, 0)] * dvx_dx
            - model.C33[IdxSigFi(ix, iz, zone, 0)] * model.taup[IdxSigFi(ix, iz, zone, 0)] * dvz_dz
        );
        break;
    }
    }
}

__global__ void update_tau_fine(Core core, Model model, int cur, int zone) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= fines[zone].lenx - 1 || iz >= fines[zone].lenz - 1) {
        return;
    }
    auto is_fluid = [&](int x, int z) -> bool {
        return MAT(IdxSigFi(x, z, zone, 0)) == FLUID || MAT(IdxSigFi(x + 1, z, zone, 0)) == FLUID || 
               MAT(IdxSigFi(x, z + 1, zone, 0)) == FLUID || MAT(IdxSigFi(x + 1, z + 1, zone, 0)) == FLUID;
    };
    auto is_vesolid_txz = [&](int x, int z) -> bool {
        return MAT(IdxSigFi(x, z, zone, 0)) == VESOLID || MAT(IdxSigFi(x + 1, z, zone, 0)) == VESOLID || 
               MAT(IdxSigFi(x, z + 1, zone, 0)) == VESOLID || MAT(IdxSigFi(x + 1, z + 1, zone, 0)) == VESOLID;
    };
    if (is_fluid(ix, iz)) {
        return;
    }

    switch (is_vesolid_txz(ix, iz)) {
    case 0: {
        float dvx_dz = dvx_dz_8th(core.vx, ix, iz, zone, cur ^ 1);
        float dvz_dx = dvz_dx_8th(core.vz, ix, iz, zone, cur ^ 1);
        core.txz[IdxTxzFi(ix, iz, zone, cur)] = core.txz[IdxTxzFi(ix, iz, zone, cur ^ 1)] + (
            dt_d * samp_C55_fine(model.C55, ix, iz, zone) * (dvz_dx + dvx_dz)
        );
        break;
    }
    case 1: {
        float dvx_dz = dvx_dz_8th(core.vx, ix, iz, zone, cur ^ 1);
        float dvz_dx = dvz_dx_8th(core.vz, ix, iz, zone, cur ^ 1);
        core.txz[IdxTxzFi(ix, iz, zone, cur)] = core.txz[IdxTxzFi(ix, iz, zone, cur ^ 1)] + dt_d * (
            + samp_C55_fine(model.C55, ix, iz, zone) * (dvz_dx + dvx_dz) * (
                1 + samp_taus_fine(model.taus, ix, iz, zone)
            ) 
            + model.inv_tsig[IdxTxzFi(ix, iz, zone, 0)] * core.rxz[IdxTxzFi(ix, iz, zone, cur ^ 1)]
        );
        core.rxz[IdxTxzFi(ix, iz, zone, cur)] = core.rxz[IdxTxzFi(ix, iz, zone, cur ^ 1)] + dt_d * (
            - model.inv_tsig[IdxTxzFi(ix, iz, zone, 0)] * core.rxz[IdxTxzFi(ix, iz, zone, cur ^ 1)]
            - model.C55[IdxTxzFi(ix, iz, zone, 0)] * model.taus[IdxTxzFi(ix, iz, zone, 0)] * (dvz_dx + dvx_dz)
        );
        break;
    }
    }
}

__global__ void update_velocity_fine(Core core, Model model, int cur, int zone) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= fines[zone].lenx || iz >= fines[zone].lenz) {
        return;
    }

    float dsx_dx = 0;
    float dsz_dz = 0;
    float dtxz_dx = 0;
    float dtxz_dz = 0;
    
    if (ix < fines[zone].lenx - 1) {
        if (MAT(IdxSigFi(ix, iz, zone, 0)) + MAT(IdxSigFi(ix + 1, iz, zone, 0)) <= 3) {
            dtxz_dz = dtxz_dz_8th(core.txz, ix, iz, zone, cur);
        } 
        dsx_dx = dsx_dx_8th(core.sx, ix, iz, zone, cur);
        core.vx[IdxVxFi(ix, iz, zone, cur)] = core.vx[IdxVxFi(ix, iz, zone, cur ^ 1)] + (
            dt_d / samp_rho_x_fine(model.rho, ix, iz, zone) * (dsx_dx + dtxz_dz)
        );
    }

    if (iz < fines[zone].lenz - 1) {
        if (MAT(IdxSigFi(ix, iz, zone, 0)) + MAT(IdxSigFi(ix, iz + 1, zone, 0)) <= 3) {
            dtxz_dx = dtxz_dx_8th(core.txz, ix, iz, zone, cur);
        } 
        dsz_dz = dsz_dz_8th(core.sz, ix, iz, zone, cur);
        core.vz[IdxVzFi(ix, iz, zone, cur)] = core.vz[IdxVzFi(ix, iz, zone, cur ^ 1)] + (
            dt_d / samp_rho_z_fine(model.rho, ix, iz, zone) * (dtxz_dx + dsz_dz)
        );
    }
}

__global__ void apply_fluid_boundary_fine(Core core, int cur, int zone) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= fines[zone].lenx || iz >= fines[zone].lenz) {
        return;
    }
    if (tex1Dfetch<int>(mat_tex, IdxSigFi(ix, iz, zone, 0)) == FLUID) {
        core.txz[IdxTxzFi(ix, iz, zone, cur)] = 0;
        core.txz[IdxTxzFi(max(0, ix - 1), iz, zone, cur)] = 0;
        core.txz[IdxTxzFi(ix, max(0, iz - 1), zone, cur)] = 0;
        core.txz[IdxTxzFi(max(0, ix - 1), max(0, iz - 1), zone, cur)] = 0;
        core.rxz[IdxTxzFi(ix, iz, zone, cur)] = 0;
        core.rxz[IdxTxzFi(max(0, ix - 1), iz, zone, cur)] = 0;
        core.rxz[IdxTxzFi(ix, max(0, iz - 1), zone, cur)] = 0;
        core.rxz[IdxTxzFi(max(0, ix - 1), max(0, iz - 1), zone, cur)] = 0;
    }
}

__global__ void smooth_fine_vx(float *vx, float *temp, int cur, int zone, int lvl) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fines[zone].lenx - 1 || iz >= fines[zone].lenz) return;

    int idx_dst = IdxVxFi(ix, iz, zone, 0) - sum_offset_fine_vx[0];
    int idx_src = IdxVxFi(ix, iz, zone, cur);

    float sum = coeff[lvl][0] * vx[idx_src];
    float total_weight = coeff[lvl][0];

    // 正交邻点
    if (iz > 0) {
        sum += coeff[lvl][1] * vx[IdxVxFi(ix, iz-1, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (iz < fines[zone].lenz - 1) {
        sum += coeff[lvl][1] * vx[IdxVxFi(ix, iz+1, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (ix > 0) {
        sum += coeff[lvl][1] * vx[IdxVxFi(ix-1, iz, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (ix < fines[zone].lenx - 2) {
        sum += coeff[lvl][1] * vx[IdxVxFi(ix+1, iz, zone, cur)];
        total_weight += coeff[lvl][1];
    }

    // 对角邻点
    if (ix > 0 && iz > 0) {
        sum += coeff[lvl][2] * vx[IdxVxFi(ix-1, iz-1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 2 && iz > 0) {
        sum += coeff[lvl][2] * vx[IdxVxFi(ix+1, iz-1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix > 0 && iz < fines[zone].lenz - 1) {
        sum += coeff[lvl][2] * vx[IdxVxFi(ix-1, iz+1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 2 && iz < fines[zone].lenz - 1) {
        sum += coeff[lvl][2] * vx[IdxVxFi(ix+1, iz+1, zone, cur)];
        total_weight += coeff[lvl][2];
    }

    temp[idx_dst] = sum / total_weight;
}

__global__ void smooth_fine_vz(float *vz, float *temp, int cur, int zone, int lvl) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fines[zone].lenx || iz >= fines[zone].lenz - 1) return;

    int idx_dst = IdxVzFi(ix, iz, zone, 0) - sum_offset_fine_vz[0];
    int idx_src = IdxVzFi(ix, iz, zone, cur);

    float sum = coeff[lvl][0] * vz[idx_src];
    float total_weight = coeff[lvl][0];

    // 正交邻点
    if (iz > 0) {
        sum += coeff[lvl][1] * vz[IdxVzFi(ix, iz-1, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (iz < fines[zone].lenz - 2) {
        sum += coeff[lvl][1] * vz[IdxVzFi(ix, iz+1, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (ix > 0) {
        sum += coeff[lvl][1] * vz[IdxVzFi(ix-1, iz, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (ix < fines[zone].lenx - 1) {
        sum += coeff[lvl][1] * vz[IdxVzFi(ix+1, iz, zone, cur)];
        total_weight += coeff[lvl][1];
    }

    // 对角邻点
    if (ix > 0 && iz > 0) {
        sum += coeff[lvl][2] * vz[IdxVzFi(ix-1, iz-1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 1 && iz > 0) {
        sum += coeff[lvl][2] * vz[IdxVzFi(ix+1, iz-1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix > 0 && iz < fines[zone].lenz - 2) {
        sum += coeff[lvl][2] * vz[IdxVzFi(ix-1, iz+1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 1 && iz < fines[zone].lenz - 2) {
        sum += coeff[lvl][2] * vz[IdxVzFi(ix+1, iz+1, zone, cur)];
        total_weight += coeff[lvl][2];
    }

    temp[idx_dst] = sum / total_weight;
}

__global__ void smooth_fine_sig(float *sig, float *temp, int cur, int zone, int lvl) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fines[zone].lenx || iz >= fines[zone].lenz) return;

    int idx_dst = IdxSigFi(ix, iz, zone, 0) - sum_offset_fine_sig[0];
    int idx_src = IdxSigFi(ix, iz, zone, cur);

    float sum = coeff[lvl][0] * sig[idx_src];
    float total_weight = coeff[lvl][0];

    // 正交邻点
    if (iz > 0) {
        sum += coeff[lvl][1] * sig[IdxSigFi(ix, iz-1, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (iz < fines[zone].lenz - 1) {
        sum += coeff[lvl][1] * sig[IdxSigFi(ix, iz+1, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (ix > 0) {
        sum += coeff[lvl][1] * sig[IdxSigFi(ix-1, iz, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (ix < fines[zone].lenx - 1) {
        sum += coeff[lvl][1] * sig[IdxSigFi(ix+1, iz, zone, cur)];
        total_weight += coeff[lvl][1];
    }

    // 对角邻点
    if (ix > 0 && iz > 0) {
        sum += coeff[lvl][2] * sig[IdxSigFi(ix-1, iz-1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 1 && iz > 0) {
        sum += coeff[lvl][2] * sig[IdxSigFi(ix+1, iz-1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix > 0 && iz < fines[zone].lenz - 1) {
        sum += coeff[lvl][2] * sig[IdxSigFi(ix-1, iz+1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 1 && iz < fines[zone].lenz - 1) {
        sum += coeff[lvl][2] * sig[IdxSigFi(ix+1, iz+1, zone, cur)];
        total_weight += coeff[lvl][2];
    }

    temp[idx_dst] = sum / total_weight;
}

__global__ void smooth_fine_txz(float *txz, float *temp, int cur, int zone, int lvl) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fines[zone].lenx - 1 || iz >= fines[zone].lenz - 1) return;

    int idx_dst = IdxTxzFi(ix, iz, zone, 0) - sum_offset_fine_txz[0];
    int idx_src = IdxTxzFi(ix, iz, zone, cur);

    float sum = coeff[lvl][0] * txz[idx_src];
    float total_weight = coeff[lvl][0];

    // 正交邻点
    if (iz > 0) {
        sum += coeff[lvl][1] * txz[IdxTxzFi(ix, iz-1, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (iz < fines[zone].lenz - 2) {
        sum += coeff[lvl][1] * txz[IdxTxzFi(ix, iz+1, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (ix > 0) {
        sum += coeff[lvl][1] * txz[IdxTxzFi(ix-1, iz, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (ix < fines[zone].lenx - 2) {
        sum += coeff[lvl][1] * txz[IdxTxzFi(ix+1, iz, zone, cur)];
        total_weight += coeff[lvl][1];
    }

    // 对角邻点
    if (ix > 0 && iz > 0) {
        sum += coeff[lvl][2] * txz[IdxTxzFi(ix-1, iz-1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 2 && iz > 0) {
        sum += coeff[lvl][2] * txz[IdxTxzFi(ix+1, iz-1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix > 0 && iz < fines[zone].lenz - 2) {
        sum += coeff[lvl][2] * txz[IdxTxzFi(ix-1, iz+1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 2 && iz < fines[zone].lenz - 2) {
        sum += coeff[lvl][2] * txz[IdxTxzFi(ix+1, iz+1, zone, cur)];
        total_weight += coeff[lvl][2];
    }

    temp[idx_dst] = sum / total_weight;
}

// 平滑压力场 p（流体专用）
__global__ void smooth_fine_p(float *p, float *temp, int cur, int zone, int lvl) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fines[zone].lenx || iz >= fines[zone].lenz) return;

    int idx_src = IdxSigFi(ix, iz, zone, cur);
    int idx_dst = IdxSigFi(ix, iz, zone, 0) - sum_offset_fine_sig[0];

    // 只处理流体点
    if (tex1Dfetch<int>(mat_tex, idx_src) != FLUID) {
        temp[idx_dst] = p[idx_src];  // 非流体点直接复制
        return;
    }

    float sum = coeff[lvl][0] * p[idx_src];
    float total_weight = coeff[lvl][0];

    // 辅助判断邻点是否为流体
    auto is_fluid = [&](int x, int y) -> bool {
        if (x < 0 || x >= fines[zone].lenx || y < 0 || y >= fines[zone].lenz) return false;
        return tex1Dfetch<int>(mat_tex, IdxSigFi(x, y, zone, 0)) == FLUID;
    };

    // 正交邻点
    if (iz > 0 && is_fluid(ix, iz-1)) {
        sum += coeff[lvl][1] * p[IdxSigFi(ix, iz-1, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (iz < fines[zone].lenz - 1 && is_fluid(ix, iz+1)) {
        sum += coeff[lvl][1] * p[IdxSigFi(ix, iz+1, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (ix > 0 && is_fluid(ix-1, iz)) {
        sum += coeff[lvl][1] * p[IdxSigFi(ix-1, iz, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (ix < fines[zone].lenx - 1 && is_fluid(ix+1, iz)) {
        sum += coeff[lvl][1] * p[IdxSigFi(ix+1, iz, zone, cur)];
        total_weight += coeff[lvl][1];
    }

    // 对角邻点
    if (ix > 0 && iz > 0 && is_fluid(ix-1, iz-1)) {
        sum += coeff[lvl][2] * p[IdxSigFi(ix-1, iz-1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 1 && iz > 0 && is_fluid(ix+1, iz-1)) {
        sum += coeff[lvl][2] * p[IdxSigFi(ix+1, iz-1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix > 0 && iz < fines[zone].lenz - 1 && is_fluid(ix-1, iz+1)) {
        sum += coeff[lvl][2] * p[IdxSigFi(ix-1, iz+1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 1 && iz < fines[zone].lenz - 1 && is_fluid(ix+1, iz+1)) {
        sum += coeff[lvl][2] * p[IdxSigFi(ix+1, iz+1, zone, cur)];
        total_weight += coeff[lvl][2];
    }

    temp[idx_dst] = sum / total_weight;
}

// 平滑记忆变量 rx
__global__ void smooth_fine_rx(float *rx, float *temp, int cur, int zone, int lvl) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fines[zone].lenx || iz >= fines[zone].lenz) return;

    int idx_src = IdxSigFi(ix, iz, zone, cur);
    int idx_dst = IdxSigFi(ix, iz, zone, 0) - sum_offset_fine_sig[0];

    // 只处理 VESOLID 点
    if (tex1Dfetch<int>(mat_tex, idx_src) != VESOLID) {
        temp[idx_dst] = rx[idx_src];
        return;
    }

    float sum = coeff[lvl][0] * rx[idx_src];
    float total_weight = coeff[lvl][0];

    auto is_vesolid = [&](int x, int y) -> bool {
        if (x < 0 || x >= fines[zone].lenx || y < 0 || y >= fines[zone].lenz) return false;
        return tex1Dfetch<int>(mat_tex, IdxSigFi(x, y, zone, 0)) == VESOLID;
    };

    // 正交邻点
    if (iz > 0 && is_vesolid(ix, iz-1)) {
        sum += coeff[lvl][1] * rx[IdxSigFi(ix, iz-1, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (iz < fines[zone].lenz - 1 && is_vesolid(ix, iz+1)) {
        sum += coeff[lvl][1] * rx[IdxSigFi(ix, iz+1, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (ix > 0 && is_vesolid(ix-1, iz)) {
        sum += coeff[lvl][1] * rx[IdxSigFi(ix-1, iz, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (ix < fines[zone].lenx - 1 && is_vesolid(ix+1, iz)) {
        sum += coeff[lvl][1] * rx[IdxSigFi(ix+1, iz, zone, cur)];
        total_weight += coeff[lvl][1];
    }

    // 对角邻点
    if (ix > 0 && iz > 0 && is_vesolid(ix-1, iz-1)) {
        sum += coeff[lvl][2] * rx[IdxSigFi(ix-1, iz-1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 1 && iz > 0 && is_vesolid(ix+1, iz-1)) {
        sum += coeff[lvl][2] * rx[IdxSigFi(ix+1, iz-1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix > 0 && iz < fines[zone].lenz - 1 && is_vesolid(ix-1, iz+1)) {
        sum += coeff[lvl][2] * rx[IdxSigFi(ix-1, iz+1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 1 && iz < fines[zone].lenz - 1 && is_vesolid(ix+1, iz+1)) {
        sum += coeff[lvl][2] * rx[IdxSigFi(ix+1, iz+1, zone, cur)];
        total_weight += coeff[lvl][2];
    }

    temp[idx_dst] = sum / total_weight;
}

// 平滑记忆变量 rz（整网格，与 sx 同位置）
__global__ void smooth_fine_rz(float *rz, float *temp, int cur, int zone, int lvl) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fines[zone].lenx || iz >= fines[zone].lenz) return;

    int idx_src = IdxSigFi(ix, iz, zone, cur);
    int idx_dst = IdxSigFi(ix, iz, zone, 0) - sum_offset_fine_sig[0];

    if (tex1Dfetch<int>(mat_tex, idx_src) != VESOLID) {
        temp[idx_dst] = rz[idx_src];
        return;
    }

    float sum = coeff[lvl][0] * rz[idx_src];
    float total_weight = coeff[lvl][0];

    auto is_vesolid = [&](int x, int z) -> bool {
        if (x < 0 || x >= fines[zone].lenx || z < 0 || z >= fines[zone].lenz) return false;
        return tex1Dfetch<int>(mat_tex, IdxSigFi(x, z, zone, 0)) == VESOLID;
    };

    // 正交邻点
    if (iz > 0 && is_vesolid(ix, iz-1)) {
        sum += coeff[lvl][1] * rz[IdxSigFi(ix, iz-1, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (iz < fines[zone].lenz - 1 && is_vesolid(ix, iz+1)) {
        sum += coeff[lvl][1] * rz[IdxSigFi(ix, iz+1, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (ix > 0 && is_vesolid(ix-1, iz)) {
        sum += coeff[lvl][1] * rz[IdxSigFi(ix-1, iz, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (ix < fines[zone].lenx - 1 && is_vesolid(ix+1, iz)) {
        sum += coeff[lvl][1] * rz[IdxSigFi(ix+1, iz, zone, cur)];
        total_weight += coeff[lvl][1];
    }

    // 对角邻点
    if (ix > 0 && iz > 0 && is_vesolid(ix-1, iz-1)) {
        sum += coeff[lvl][2] * rz[IdxSigFi(ix-1, iz-1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 1 && iz > 0 && is_vesolid(ix+1, iz-1)) {
        sum += coeff[lvl][2] * rz[IdxSigFi(ix+1, iz-1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix > 0 && iz < fines[zone].lenz - 1 && is_vesolid(ix-1, iz+1)) {
        sum += coeff[lvl][2] * rz[IdxSigFi(ix-1, iz+1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 1 && iz < fines[zone].lenz - 1 && is_vesolid(ix+1, iz+1)) {
        sum += coeff[lvl][2] * rz[IdxSigFi(ix+1, iz+1, zone, cur)];
        total_weight += coeff[lvl][2];
    }

    temp[idx_dst] = sum / total_weight;
}

__global__ void smooth_fine_rxz(float *rxz, float *temp, int cur, int zone, int lvl) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fines[zone].lenx - 1 || iz >= fines[zone].lenz - 1) return;


    int idx_src = IdxTxzFi(ix, iz, zone, cur);
    int idx_dst = IdxTxzFi(ix, iz, zone, 0) - sum_offset_fine_txz[0];

    float sum = coeff[lvl][0] * rxz[idx_src];
    float total_weight = coeff[lvl][0];

    auto is_vesolid_txz = [&](int x, int z) -> bool {
        if (x < 0 || x >= fines[zone].lenx - 1 || z < 0 || z >= fines[zone].lenz - 1) return false;
        return MAT(IdxSigFi(x, z, zone, 0)) == VESOLID || MAT(IdxSigFi(x + 1, z, zone, 0)) == VESOLID || 
               MAT(IdxSigFi(x, z + 1, zone, 0)) == VESOLID || MAT(IdxSigFi(x + 1, z + 1, zone, 0)) == VESOLID;
    };

    if (!is_vesolid_txz(ix, iz)) {
        temp[idx_dst] = rxz[idx_src];
        return;
    }

    // 正交邻点
    if (iz > 0 && is_vesolid_txz(ix, iz-1)) {
        sum += coeff[lvl][1] * rxz[IdxTxzFi(ix, iz-1, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (iz < fines[zone].lenz - 2 && is_vesolid_txz(ix, iz+1)) {
        sum += coeff[lvl][1] * rxz[IdxTxzFi(ix, iz+1, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (ix > 0 && is_vesolid_txz(ix-1, iz)) {
        sum += coeff[lvl][1] * rxz[IdxTxzFi(ix-1, iz, zone, cur)];
        total_weight += coeff[lvl][1];
    }
    if (ix < fines[zone].lenx - 2 && is_vesolid_txz(ix+1, iz)) {
        sum += coeff[lvl][1] * rxz[IdxTxzFi(ix+1, iz, zone, cur)];
        total_weight += coeff[lvl][1];
    }

    // 对角邻点
    if (ix > 0 && iz > 0 && is_vesolid_txz(ix-1, iz-1)) {
        sum += coeff[lvl][2] * rxz[IdxTxzFi(ix-1, iz-1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 2 && iz > 0 && is_vesolid_txz(ix+1, iz-1)) {
        sum += coeff[lvl][2] * rxz[IdxTxzFi(ix+1, iz-1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix > 0 && iz < fines[zone].lenz - 2 && is_vesolid_txz(ix-1, iz+1)) {
        sum += coeff[lvl][2] * rxz[IdxTxzFi(ix-1, iz+1, zone, cur)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 2 && iz < fines[zone].lenz - 2 && is_vesolid_txz(ix+1, iz+1)) {
        sum += coeff[lvl][2] * rxz[IdxTxzFi(ix+1, iz+1, zone, cur)];
        total_weight += coeff[lvl][2];
    }

    temp[idx_dst] = sum / total_weight;
}
