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

__global__ void update_stress_coarse(Core core, Model model, PsiVel psi_vel, int cur) {
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

            core.sx[IdxSigCo(ix, iz, cur)] = (
                    + core.sx[IdxSigCo(ix, iz, cur ^ 1)]
                    + dt_d * (
                        + model.C11[IdxSigCo(ix, iz, 0)] * dvx_dx 
                        + model.C13[IdxSigCo(ix, iz, 0)] * dvz_dz
                    )
                );
            core.sz[IdxSigCo(ix, iz, cur)] = (
                + core.sz[IdxSigCo(ix, iz, cur ^ 1)]
                + dt_d * (
                    + model.C13[IdxSigCo(ix, iz, 0)] * dvx_dx
                    + model.C33[IdxSigCo(ix, iz, 0)] * dvz_dz
                )
            );
        }

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
            
            core.p[IdxSigCo(ix, iz, cur)] = core.p[IdxSigCo(ix, iz, cur ^ 1)] + (
                - dt_d * ( 
                    + model.C11[IdxSigCo(ix, iz, 0)] * (1 + model.tau[IdxSigCo(ix, iz, 0)]) * (
                        dvx_dx + dvz_dz
                    )
                    - core.r[IdxSigCo(ix, iz, cur ^ 1)] * model.inv_taus[IdxSigCo(ix, iz, cur ^ 1)]
                )
            );
            core.sx[IdxSigCo(ix, iz, cur)] = -core.p[IdxSigCo(ix, iz, cur)];
            core.sz[IdxSigCo(ix, iz, cur)] = -core.p[IdxSigCo(ix, iz, cur)];

            core.r[IdxSigCo(ix, iz, cur)] = core.r[IdxSigCo(ix, iz, cur ^ 1)] + (
                - dt_d * (
                    + core.r[IdxSigCo(ix, iz, cur ^ 1)] * model.inv_taus[IdxSigCo(ix, iz, cur ^ 1)]
                    + model.C11[IdxSigCo(ix, iz, 0)] * model.tau[IdxSigCo(ix, iz, 0)] * (
                        dvx_dx + dvz_dz
                    )
                )
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
    }
}

__global__ void update_velocity_coarse(Core core, Model model, PsiStr psi_str, int cur) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < 3 || ix >= nx - 4 || iz < 3 || iz >= nz - 4) {
        return;
    }

    if (tex1Dfetch<int>(vx_mask, iz * (nx - 1) + ix) == -1) {
        float dsx_dx = dsx_dx_coarse(core.sx, ix, iz, cur);
        float dtxz_dz = 0;                                        // HALO 或者流体内部默认为0
        // 1. 固体内部以及流固边界，使用txz的导数
        // 2. 流体内部，不使用txz的导数
        if (iz > 3 &&
            ((MAT(iz * nx + ix) == SOLID && MAT(iz * nx + ix + 1) == SOLID) || // 1. 固体内部
            MAT(iz * nx + ix) != MAT(iz * nx + ix + 1))) {                     // 2. 流固边界
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
        float dtxz_dx = 0;                                      // HALO 或者流体内部默认为0
        if (ix > 3 && 
            ((MAT(iz * nx + ix) == SOLID && MAT((iz + 1) * nx + ix) == SOLID) || // 1. 固体内部
            MAT(iz * nx + ix) != MAT((iz + 1) * nx + ix))) {                     // 2. 流固边界
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

__global__ void update_stress_fine(Core core, Model model, int cur, int zone) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= fines[zone].lenx || iz >= fines[zone].lenz) {
        return;
    }
    float dvx_dx = 0;
    float dvz_dz = 0;

    switch (MAT(IdxSigFi(ix, iz, 0, zone))) {
    default:
        dvx_dx = dvx_dx_8th(core.vx, ix, iz, cur ^ 1, zone);
        dvz_dz = dvz_dz_8th(core.vz, ix, iz, cur ^ 1, zone);
    case SOLID:
        core.sx[IdxSigFi(ix, iz, cur, zone)] = core.sx[IdxSigFi(ix, iz, cur ^ 1, zone)] + (
            dt_d * (
                + model.C11[IdxSigFi(ix, iz, 0, zone)] * dvx_dx
                + model.C13[IdxSigFi(ix, iz, 0, zone)] * dvz_dz
            )
        );
        core.sz[IdxSigFi(ix, iz, cur, zone)] = core.sz[IdxSigFi(ix, iz, cur ^ 1, zone)] + (
            dt_d * (
                + model.C13[IdxSigFi(ix, iz, 0, zone)] * dvx_dx
                + model.C33[IdxSigFi(ix, iz, 0, zone)] * dvz_dz
            )
        );

        if (ix < fines[zone].lenx - 1 && iz < fines[zone].lenz - 1) {
            float dvx_dz = dvx_dz_8th(core.vx, ix, iz, cur ^ 1, zone);
            float dvz_dx = dvz_dx_8th(core.vz, ix, iz, cur ^ 1, zone);
            core.txz[IdxTxzFi(ix, iz, cur, zone)] = core.txz[IdxTxzFi(ix, iz, cur ^ 1, zone)] + (
                dt_d * samp_C55_fine(model.C55, ix, iz, zone) * (dvz_dx + dvx_dz)
            );
        }
        break;

    case FLUID:
        core.p[IdxSigFi(ix, iz, cur, zone)] = core.p[IdxSigFi(ix, iz, cur ^ 1, zone)] + dt_d * (
            - model.C11[IdxSigFi(ix, iz, 0, zone)] * (1 + model.tau[IdxSigFi(ix, iz, 0, zone)]) * (
                dvx_dx + dvz_dz
            ) + model.inv_taus[IdxSigFi(ix, iz, 0, zone)] * core.r[IdxSigFi(ix, iz, cur ^ 1, zone)]
        );
        core.sx[IdxSigFi(ix, iz, cur, zone)] = -core.p[IdxSigFi(ix, iz, cur, zone)];
        core.sz[IdxSigFi(ix, iz, cur, zone)] = -core.p[IdxSigFi(ix, iz, cur, zone)];

        core.r[IdxSigFi(ix, iz, cur, zone)] = core.r[IdxSigFi(ix, iz, cur ^ 1, zone)] + dt_d * (
            - model.inv_taus[IdxSigFi(ix, iz, 0, zone)] * core.r[IdxSigFi(ix, iz, cur ^ 1, zone)]
            - model.C11[IdxSigFi(ix, iz, 0, zone)] * model.tau[IdxSigFi(ix, iz, 0, zone)] * (dvx_dx + dvz_dz)
        );
        break;
    }
}

__global__ void apply_fluid_boundary_fine(Core core, int cur, int zone) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= fines[zone].lenx || iz >= fines[zone].lenz) {
        return;
    }
    if (tex1Dfetch<int>(mat_tex, IdxSigFi(ix, iz, 0, zone)) == FLUID) {
        core.txz[IdxTxzFi(ix, iz, cur, zone)] = 0;
        core.txz[IdxTxzFi(max(0, ix - 1), iz, cur, zone)] = 0;
        core.txz[IdxTxzFi(ix, max(0, iz - 1), cur, zone)] = 0;
        core.txz[IdxTxzFi(max(0, ix - 1), max(0, iz - 1), cur, zone)] = 0;
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
        if ((MAT(IdxSigFi(ix, iz, 0, zone)) == SOLID && MAT(IdxSigFi(ix + 1, iz, 0, zone)) == SOLID) || 
            MAT(IdxSigFi(ix, iz, 0, zone)) != MAT(IdxSigFi(ix + 1, iz, 0, zone))) {
            dtxz_dz = dtxz_dz_8th(core.txz, ix, iz, cur, zone);
        } 
        dsx_dx = dsx_dx_8th(core.sx, ix, iz, cur, zone);
        core.vx[IdxVxFi(ix, iz, cur, zone)] = core.vx[IdxVxFi(ix, iz, cur ^ 1, zone)] + (
            dt_d / samp_rho_x_fine(model.rho, ix, iz, zone) * (dsx_dx + dtxz_dz)
        );
    }

    if (iz < fines[zone].lenz - 1) {
        if ((MAT(IdxSigFi(ix, iz, 0, zone)) == SOLID && MAT(IdxSigFi(ix, iz + 1, 0, zone)) == SOLID) || 
            MAT(IdxSigFi(ix, iz, 0, zone)) != MAT(IdxSigFi(ix, iz + 1, 0, zone))) {
            dtxz_dx = dtxz_dx_8th(core.txz, ix, iz, cur, zone);
        } 
        dsz_dz = dsz_dz_8th(core.sz, ix, iz, cur, zone);
        core.vz[IdxVzFi(ix, iz, cur, zone)] = core.vz[IdxVzFi(ix, iz, cur ^ 1, zone)] + (
            dt_d / samp_rho_z_fine(model.rho, ix, iz, zone) * (dtxz_dx + dsz_dz)
        );
    }
}

__global__ void smooth_fine_vx(float *vx, float *temp, int cur, int zone, int lvl) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fines[zone].lenx - 1 || iz >= fines[zone].lenz) return;

    int idx_dst = IdxVxFi(ix, iz, 0, zone) - sum_offset_fine_vx[0];
    int idx_src = IdxVxFi(ix, iz, cur, zone);

    float sum = coeff[lvl][0] * vx[idx_src];
    float total_weight = coeff[lvl][0];

    // 正交邻点
    if (iz > 0) {
        sum += coeff[lvl][1] * vx[IdxVxFi(ix, iz-1, cur, zone)];
        total_weight += coeff[lvl][1];
    }
    if (iz < fines[zone].lenz - 1) {
        sum += coeff[lvl][1] * vx[IdxVxFi(ix, iz+1, cur, zone)];
        total_weight += coeff[lvl][1];
    }
    if (ix > 0) {
        sum += coeff[lvl][1] * vx[IdxVxFi(ix-1, iz, cur, zone)];
        total_weight += coeff[lvl][1];
    }
    if (ix < fines[zone].lenx - 2) {
        sum += coeff[lvl][1] * vx[IdxVxFi(ix+1, iz, cur, zone)];
        total_weight += coeff[lvl][1];
    }

    // 对角邻点
    if (ix > 0 && iz > 0) {
        sum += coeff[lvl][2] * vx[IdxVxFi(ix-1, iz-1, cur, zone)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 2 && iz > 0) {
        sum += coeff[lvl][2] * vx[IdxVxFi(ix+1, iz-1, cur, zone)];
        total_weight += coeff[lvl][2];
    }
    if (ix > 0 && iz < fines[zone].lenz - 1) {
        sum += coeff[lvl][2] * vx[IdxVxFi(ix-1, iz+1, cur, zone)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 2 && iz < fines[zone].lenz - 1) {
        sum += coeff[lvl][2] * vx[IdxVxFi(ix+1, iz+1, cur, zone)];
        total_weight += coeff[lvl][2];
    }

    temp[idx_dst] = sum / total_weight;
}

__global__ void smooth_fine_vz(float *vz, float *temp, int cur, int zone, int lvl) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fines[zone].lenx || iz >= fines[zone].lenz - 1) return;

    int idx_dst = IdxVzFi(ix, iz, 0, zone) - sum_offset_fine_vz[0];
    int idx_src = IdxVzFi(ix, iz, cur, zone);

    float sum = coeff[lvl][0] * vz[idx_src];
    float total_weight = coeff[lvl][0];

    // 正交邻点
    if (iz > 0) {
        sum += coeff[lvl][1] * vz[IdxVzFi(ix, iz-1, cur, zone)];
        total_weight += coeff[lvl][1];
    }
    if (iz < fines[zone].lenz - 2) {
        sum += coeff[lvl][1] * vz[IdxVzFi(ix, iz+1, cur, zone)];
        total_weight += coeff[lvl][1];
    }
    if (ix > 0) {
        sum += coeff[lvl][1] * vz[IdxVzFi(ix-1, iz, cur, zone)];
        total_weight += coeff[lvl][1];
    }
    if (ix < fines[zone].lenx - 1) {
        sum += coeff[lvl][1] * vz[IdxVzFi(ix+1, iz, cur, zone)];
        total_weight += coeff[lvl][1];
    }

    // 对角邻点
    if (ix > 0 && iz > 0) {
        sum += coeff[lvl][2] * vz[IdxVzFi(ix-1, iz-1, cur, zone)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 1 && iz > 0) {
        sum += coeff[lvl][2] * vz[IdxVzFi(ix+1, iz-1, cur, zone)];
        total_weight += coeff[lvl][2];
    }
    if (ix > 0 && iz < fines[zone].lenz - 2) {
        sum += coeff[lvl][2] * vz[IdxVzFi(ix-1, iz+1, cur, zone)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 1 && iz < fines[zone].lenz - 2) {
        sum += coeff[lvl][2] * vz[IdxVzFi(ix+1, iz+1, cur, zone)];
        total_weight += coeff[lvl][2];
    }

    temp[idx_dst] = sum / total_weight;
}

__global__ void smooth_fine_sig(float *sig, float *temp, int cur, int zone, int lvl) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fines[zone].lenx || iz >= fines[zone].lenz) return;

    int idx_dst = IdxSigFi(ix, iz, 0, zone) - sum_offset_fine_sig[0];
    int idx_src = IdxSigFi(ix, iz, cur, zone);

    float sum = coeff[lvl][0] * sig[idx_src];
    float total_weight = coeff[lvl][0];

    // 正交邻点
    if (iz > 0) {
        sum += coeff[lvl][1] * sig[IdxSigFi(ix, iz-1, cur, zone)];
        total_weight += coeff[lvl][1];
    }
    if (iz < fines[zone].lenz - 1) {
        sum += coeff[lvl][1] * sig[IdxSigFi(ix, iz+1, cur, zone)];
        total_weight += coeff[lvl][1];
    }
    if (ix > 0) {
        sum += coeff[lvl][1] * sig[IdxSigFi(ix-1, iz, cur, zone)];
        total_weight += coeff[lvl][1];
    }
    if (ix < fines[zone].lenx - 1) {
        sum += coeff[lvl][1] * sig[IdxSigFi(ix+1, iz, cur, zone)];
        total_weight += coeff[lvl][1];
    }

    // 对角邻点
    if (ix > 0 && iz > 0) {
        sum += coeff[lvl][2] * sig[IdxSigFi(ix-1, iz-1, cur, zone)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 1 && iz > 0) {
        sum += coeff[lvl][2] * sig[IdxSigFi(ix+1, iz-1, cur, zone)];
        total_weight += coeff[lvl][2];
    }
    if (ix > 0 && iz < fines[zone].lenz - 1) {
        sum += coeff[lvl][2] * sig[IdxSigFi(ix-1, iz+1, cur, zone)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 1 && iz < fines[zone].lenz - 1) {
        sum += coeff[lvl][2] * sig[IdxSigFi(ix+1, iz+1, cur, zone)];
        total_weight += coeff[lvl][2];
    }

    temp[idx_dst] = sum / total_weight;
}

__global__ void smooth_fine_txz(float *txz, float *temp, int cur, int zone, int lvl) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fines[zone].lenx - 1 || iz >= fines[zone].lenz - 1) return;

    int idx_dst = IdxTxzFi(ix, iz, 0, zone) - sum_offset_fine_txz[0];
    int idx_src = IdxTxzFi(ix, iz, cur, zone);

    float sum = coeff[lvl][0] * txz[idx_src];
    float total_weight = coeff[lvl][0];

    // 正交邻点
    if (iz > 0) {
        sum += coeff[lvl][1] * txz[IdxTxzFi(ix, iz-1, cur, zone)];
        total_weight += coeff[lvl][1];
    }
    if (iz < fines[zone].lenz - 2) {
        sum += coeff[lvl][1] * txz[IdxTxzFi(ix, iz+1, cur, zone)];
        total_weight += coeff[lvl][1];
    }
    if (ix > 0) {
        sum += coeff[lvl][1] * txz[IdxTxzFi(ix-1, iz, cur, zone)];
        total_weight += coeff[lvl][1];
    }
    if (ix < fines[zone].lenx - 2) {
        sum += coeff[lvl][1] * txz[IdxTxzFi(ix+1, iz, cur, zone)];
        total_weight += coeff[lvl][1];
    }

    // 对角邻点
    if (ix > 0 && iz > 0) {
        sum += coeff[lvl][2] * txz[IdxTxzFi(ix-1, iz-1, cur, zone)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 2 && iz > 0) {
        sum += coeff[lvl][2] * txz[IdxTxzFi(ix+1, iz-1, cur, zone)];
        total_weight += coeff[lvl][2];
    }
    if (ix > 0 && iz < fines[zone].lenz - 2) {
        sum += coeff[lvl][2] * txz[IdxTxzFi(ix-1, iz+1, cur, zone)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 2 && iz < fines[zone].lenz - 2) {
        sum += coeff[lvl][2] * txz[IdxTxzFi(ix+1, iz+1, cur, zone)];
        total_weight += coeff[lvl][2];
    }

    temp[idx_dst] = sum / total_weight;
}

// 平滑压力场 p（流体专用）
__global__ void smooth_fine_p(float *p, float *temp, int cur, int zone, int lvl) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fines[zone].lenx || iz >= fines[zone].lenz) return;

    int idx_src = IdxSigFi(ix, iz, cur, zone);
    int idx_dst = IdxSigFi(ix, iz, 0, zone) - sum_offset_fine_sig[0];

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
        return tex1Dfetch<int>(mat_tex, IdxSigFi(x, y, 0, zone)) == FLUID;
    };

    // 正交邻点
    if (iz > 0 && is_fluid(ix, iz-1)) {
        sum += coeff[lvl][1] * p[IdxSigFi(ix, iz-1, cur, zone)];
        total_weight += coeff[lvl][1];
    }
    if (iz < fines[zone].lenz - 1 && is_fluid(ix, iz+1)) {
        sum += coeff[lvl][1] * p[IdxSigFi(ix, iz+1, cur, zone)];
        total_weight += coeff[lvl][1];
    }
    if (ix > 0 && is_fluid(ix-1, iz)) {
        sum += coeff[lvl][1] * p[IdxSigFi(ix-1, iz, cur, zone)];
        total_weight += coeff[lvl][1];
    }
    if (ix < fines[zone].lenx - 1 && is_fluid(ix+1, iz)) {
        sum += coeff[lvl][1] * p[IdxSigFi(ix+1, iz, cur, zone)];
        total_weight += coeff[lvl][1];
    }

    // 对角邻点
    if (ix > 0 && iz > 0 && is_fluid(ix-1, iz-1)) {
        sum += coeff[lvl][2] * p[IdxSigFi(ix-1, iz-1, cur, zone)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 1 && iz > 0 && is_fluid(ix+1, iz-1)) {
        sum += coeff[lvl][2] * p[IdxSigFi(ix+1, iz-1, cur, zone)];
        total_weight += coeff[lvl][2];
    }
    if (ix > 0 && iz < fines[zone].lenz - 1 && is_fluid(ix-1, iz+1)) {
        sum += coeff[lvl][2] * p[IdxSigFi(ix-1, iz+1, cur, zone)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 1 && iz < fines[zone].lenz - 1 && is_fluid(ix+1, iz+1)) {
        sum += coeff[lvl][2] * p[IdxSigFi(ix+1, iz+1, cur, zone)];
        total_weight += coeff[lvl][2];
    }

    temp[idx_dst] = sum / total_weight;
}

__global__ void smooth_fine_rp(float *rp, float *temp, int cur, int zone, int lvl) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fines[zone].lenx || iz >= fines[zone].lenz) return;

    int idx_src = IdxSigFi(ix, iz, cur, zone);
    int idx_dst = IdxSigFi(ix, iz, 0, zone) - sum_offset_fine_sig[0];

    if (tex1Dfetch<int>(mat_tex, idx_src) != FLUID) {
        temp[idx_dst] = rp[idx_src];
        return;
    }

    float sum = coeff[lvl][0] * rp[idx_src];
    float total_weight = coeff[lvl][0];

    auto is_fluid = [&](int x, int y) -> bool {
        if (x < 0 || x >= fines[zone].lenx || y < 0 || y >= fines[zone].lenz) return false;
        return tex1Dfetch<int>(mat_tex, IdxSigFi(x, y, 0, zone)) == FLUID;
    };

    if (iz > 0 && is_fluid(ix, iz-1)) {
        sum += coeff[lvl][1] * rp[IdxSigFi(ix, iz-1, cur, zone)];
        total_weight += coeff[lvl][1];
    }
    if (iz < fines[zone].lenz - 1 && is_fluid(ix, iz+1)) {
        sum += coeff[lvl][1] * rp[IdxSigFi(ix, iz+1, cur, zone)];
        total_weight += coeff[lvl][1];
    }
    if (ix > 0 && is_fluid(ix-1, iz)) {
        sum += coeff[lvl][1] * rp[IdxSigFi(ix-1, iz, cur, zone)];
        total_weight += coeff[lvl][1];
    }
    if (ix < fines[zone].lenx - 1 && is_fluid(ix+1, iz)) {
        sum += coeff[lvl][1] * rp[IdxSigFi(ix+1, iz, cur, zone)];
        total_weight += coeff[lvl][1];
    }

    if (ix > 0 && iz > 0 && is_fluid(ix-1, iz-1)) {
        sum += coeff[lvl][2] * rp[IdxSigFi(ix-1, iz-1, cur, zone)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 1 && iz > 0 && is_fluid(ix+1, iz-1)) {
        sum += coeff[lvl][2] * rp[IdxSigFi(ix+1, iz-1, cur, zone)];
        total_weight += coeff[lvl][2];
    }
    if (ix > 0 && iz < fines[zone].lenz - 1 && is_fluid(ix-1, iz+1)) {
        sum += coeff[lvl][2] * rp[IdxSigFi(ix-1, iz+1, cur, zone)];
        total_weight += coeff[lvl][2];
    }
    if (ix < fines[zone].lenx - 1 && iz < fines[zone].lenz - 1 && is_fluid(ix+1, iz+1)) {
        sum += coeff[lvl][2] * rp[IdxSigFi(ix+1, iz+1, cur, zone)];
        total_weight += coeff[lvl][2];
    }

    temp[idx_dst] = sum / total_weight;
}
