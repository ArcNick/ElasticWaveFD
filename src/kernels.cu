#include "kernels.cuh"
#include "fd_stencil.cuh"

extern __constant__ int posx_d, posz_d;
extern __constant__ float dt_d;

__constant__ float coeff[5][3] = {
    {0, 0, 0},
    {0.95, 0.01, 0.0025},
    {0.9, 0.02, 0.005},
    {0.8, 0.04, 0.010},
    {0.5, 0.125, 0}
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
    core.sx[posz_d * nx + posx_d + cur * offset_sig_all] += src;
    core.sz[posz_d * nx + posx_d + cur * offset_sig_all] += src;
}

__global__ void update_stress_coarse(Core core, Model model, PsiVel psi_vel, int cur) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < 3 || ix >= nx - 4 || iz < 3 || iz >= nz - 4) {
        return;
    }
    if (tex1Dfetch<int>(mat_tex, iz * nx + ix) == SOLID) {
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
    } else {
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
    
    if (MAT(IdxSigFi(ix, iz, 0, zone)) == SOLID) {
        float dvx_dx = dvx_dx_8th(core.vx, ix, iz, cur ^ 1, zone);
        float dvz_dz = dvz_dz_8th(core.vz, ix, iz, cur ^ 1, zone);

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
    } else {
        float dvx_dx = dvx_dx_8th(core.vx, ix, iz, cur ^ 1, zone);
        float dvz_dz = dvz_dz_8th(core.vz, ix, iz, cur ^ 1, zone);

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

// __global__ void smooth_fine_lv1(Core core, Core temp, int cur, int zone) {
//     int ix = blockIdx.x * blockDim.x + threadIdx.x;
//     int iz = blockIdx.y * blockDim.y + threadIdx.y;

//     if (ix >= fines[zone].lenx || iz >= fines[zone].lenz) {
//         return;
//     }

//     if (0 < ix && ix < fines[zone].lenx - 2 && 0 < iz && iz < fines[zone].lenz - 1) {
//         temp.vx[IdxVxFi(ix, iz, cur, zone) - sum_offset_fine_vx[0]] = (
//             + 0.9 * core.vx[IdxVxFi(ix, iz, cur, zone)]
//             + 0.02 * (
//                 + core.vx[IdxVxFi(ix - 1, iz, cur, zone)]
//                 + core.vx[IdxVxFi(ix + 1, iz, cur, zone)]
//                 + core.vx[IdxVxFi(ix, iz - 1, cur, zone)]
//                 + core.vx[IdxVxFi(ix, iz + 1, cur, zone)]
//             )
//             + 0.005 * (
//                 + core.vx[IdxVxFi(ix - 1, iz - 1, cur, zone)]
//                 + core.vx[IdxVxFi(ix + 1, iz - 1, cur, zone)]
//                 + core.vx[IdxVxFi(ix - 1, iz + 1, cur, zone)]
//                 + core.vx[IdxVxFi(ix + 1, iz + 1, cur, zone)]
//             )
//         );
//     }
//     if (0 < ix && ix < fines[zone].lenx - 1 && 0 < iz && iz < fines[zone].lenz - 2) {
//         temp.vz[IdxVzFi(ix, iz, cur, zone) - sum_offset_fine_vz[0]] = (
//             + 0.9 * core.vz[IdxVzFi(ix, iz, cur, zone)]
//             + 0.02 * (
//                 + core.vz[IdxVzFi(ix - 1, iz, cur, zone)]
//                 + core.vz[IdxVzFi(ix + 1, iz, cur, zone)]
//                 + core.vz[IdxVzFi(ix, iz - 1, cur, zone)]
//                 + core.vz[IdxVzFi(ix, iz + 1, cur, zone)]
//             )
//             + 0.005 * (
//                 + core.vz[IdxVzFi(ix - 1, iz - 1, cur, zone)]
//                 + core.vz[IdxVzFi(ix + 1, iz - 1, cur, zone)]
//                 + core.vz[IdxVzFi(ix - 1, iz + 1, cur, zone)]
//                 + core.vz[IdxVzFi(ix + 1, iz + 1, cur, zone)]
//             )
//         );
//     }

//     if (0 < ix && ix < fines[zone].lenx - 2 && 0 < iz && iz < fines[zone].lenz - 2) {
//         temp.txz[IdxTxzFi(ix, iz, cur, zone) - sum_offset_fine_txz[0]] = (
//             + 0.9 * core.txz[IdxTxzFi(ix, iz, cur, zone)]
//             + 0.02 * (
//                 + core.txz[IdxTxzFi(ix - 1, iz, cur, zone)]
//                 + core.txz[IdxTxzFi(ix + 1, iz, cur, zone)]
//                 + core.txz[IdxTxzFi(ix, iz - 1, cur, zone)]
//                 + core.txz[IdxTxzFi(ix, iz + 1, cur, zone)]
//             )
//             + 0.005 * (
//                 + core.txz[IdxTxzFi(ix - 1, iz - 1, cur, zone)]
//                 + core.txz[IdxTxzFi(ix + 1, iz - 1, cur, zone)]
//                 + core.txz[IdxTxzFi(ix - 1, iz + 1, cur, zone)]
//                 + core.txz[IdxTxzFi(ix + 1, iz + 1, cur, zone)]
//             )
//         );
//     }

//     if (0 < ix && ix < fines[zone].lenx - 1 && 0 < iz && iz < fines[zone].lenz - 1) {
//         temp.sx[IdxSigFi(ix, iz, cur, zone) - sum_offset_fine_sig[0]] = (
//             + 0.9 * core.sx[IdxSigFi(ix, iz, cur, zone)]
//             + 0.02 * (
//                 + core.sx[IdxSigFi(ix - 1, iz, cur, zone)]
//                 + core.sx[IdxSigFi(ix + 1, iz, cur, zone)]
//                 + core.sx[IdxSigFi(ix, iz - 1, cur, zone)]
//                 + core.sx[IdxSigFi(ix, iz + 1, cur, zone)]
//             )
//             + 0.005 * (
//                 + core.sx[IdxSigFi(ix - 1, iz - 1, cur, zone)]
//                 + core.sx[IdxSigFi(ix + 1, iz - 1, cur, zone)]
//                 + core.sx[IdxSigFi(ix - 1, iz + 1, cur, zone)]
//                 + core.sx[IdxSigFi(ix + 1, iz + 1, cur, zone)]
//             )
//         );

//         temp.sz[IdxSigFi(ix, iz, cur, zone) - sum_offset_fine_sig[0]] = (
//             + 0.9 * core.sz[IdxSigFi(ix, iz, cur, zone)]
//             + 0.02 * (
//                 + core.sz[IdxSigFi(ix - 1, iz, cur, zone)]
//                 + core.sz[IdxSigFi(ix + 1, iz, cur, zone)]
//                 + core.sz[IdxSigFi(ix, iz - 1, cur, zone)]
//                 + core.sz[IdxSigFi(ix, iz + 1, cur, zone)]
//             )
//             + 0.005 * (
//                 + core.sz[IdxSigFi(ix - 1, iz - 1, cur, zone)]
//                 + core.sz[IdxSigFi(ix + 1, iz - 1, cur, zone)]
//                 + core.sz[IdxSigFi(ix - 1, iz + 1, cur, zone)]
//                 + core.sz[IdxSigFi(ix + 1, iz + 1, cur, zone)]
//             )
//         );

//         temp.p[IdxSigFi(ix, iz, cur, zone) - sum_offset_fine_sig[0]] = (
//             + 0.9 * core.p[IdxSigFi(ix, iz, cur, zone)]
//             + 0.02 * (
//                 + core.p[IdxSigFi(ix - 1, iz, cur, zone)]
//                 + core.p[IdxSigFi(ix + 1, iz, cur, zone)]
//                 + core.p[IdxSigFi(ix, iz - 1, cur, zone)]
//                 + core.p[IdxSigFi(ix, iz + 1, cur, zone)]
//             )
//             + 0.005 * (
//                 + core.p[IdxSigFi(ix - 1, iz - 1, cur, zone)]
//                 + core.p[IdxSigFi(ix + 1, iz - 1, cur, zone)]
//                 + core.p[IdxSigFi(ix - 1, iz + 1, cur, zone)]
//                 + core.p[IdxSigFi(ix + 1, iz + 1, cur, zone)]
//             )
//         );

//         temp.r[IdxSigFi(ix, iz, cur, zone) - sum_offset_fine_sig[0]] = (
//             + 0.9 * core.r[IdxSigFi(ix, iz, cur, zone)]
//             + 0.02 * (
//                 + core.r[IdxSigFi(ix - 1, iz, cur, zone)]
//                 + core.r[IdxSigFi(ix + 1, iz, cur, zone)]
//                 + core.r[IdxSigFi(ix, iz - 1, cur, zone)]
//                 + core.r[IdxSigFi(ix, iz + 1, cur, zone)]
//             )
//             + 0.005 * (
//                 + core.r[IdxSigFi(ix - 1, iz - 1, cur, zone)]
//                 + core.r[IdxSigFi(ix + 1, iz - 1, cur, zone)]
//                 + core.r[IdxSigFi(ix - 1, iz + 1, cur, zone)]
//                 + core.r[IdxSigFi(ix + 1, iz + 1, cur, zone)]
//             )
//         );
//     }
// }

// __global__ void smooth_fine_lv2(Core core, Core temp, int cur, int zone) {
//     int ix = blockIdx.x * blockDim.x + threadIdx.x;
//     int iz = blockIdx.y * blockDim.y + threadIdx.y;

//     if (ix >= fines[zone].lenx || iz >= fines[zone].lenz) {
//         return;
//     }

//     if (0 < ix && ix < fines[zone].lenx - 2 && 0 < iz && iz < fines[zone].lenz - 1) {
//         temp.vx[IdxVxFi(ix, iz, cur, zone)] = (
//             + 0.8 * core.vx[IdxVxFi(ix, iz, cur, zone)]
//             + 0.04 * (
//                 + core.vx[IdxVxFi(ix - 1, iz, cur, zone)]
//                 + core.vx[IdxVxFi(ix + 1, iz, cur, zone)]
//                 + core.vx[IdxVxFi(ix, iz - 1, cur, zone)]
//                 + core.vx[IdxVxFi(ix, iz + 1, cur, zone)]
//             )
//             + 0.01 * (
//                 + core.vx[IdxVxFi(ix - 1, iz - 1, cur, zone)]
//                 + core.vx[IdxVxFi(ix + 1, iz - 1, cur, zone)]
//                 + core.vx[IdxVxFi(ix - 1, iz + 1, cur, zone)]
//                 + core.vx[IdxVxFi(ix + 1, iz + 1, cur, zone)]
//             )
//         );
//     }
//     if (0 < ix && ix < fines[zone].lenx - 1 && 0 < iz && iz < fines[zone].lenz - 2) {
//         temp.vz[IdxVzFi(ix, iz, cur, zone)] = (
//             + 0.8 * core.vz[IdxVzFi(ix, iz, cur, zone)]
//             + 0.04 * (
//                 + core.vz[IdxVzFi(ix - 1, iz, cur, zone)]
//                 + core.vz[IdxVzFi(ix + 1, iz, cur, zone)]
//                 + core.vz[IdxVzFi(ix, iz - 1, cur, zone)]
//                 + core.vz[IdxVzFi(ix, iz + 1, cur, zone)]
//             )
//             + 0.01 * (
//                 + core.vz[IdxVzFi(ix - 1, iz - 1, cur, zone)]
//                 + core.vz[IdxVzFi(ix + 1, iz - 1, cur, zone)]
//                 + core.vz[IdxVzFi(ix - 1, iz + 1, cur, zone)]
//                 + core.vz[IdxVzFi(ix + 1, iz + 1, cur, zone)]
//             )
//         );
//     }

//     if (0 < ix && ix < fines[zone].lenx - 2 && 0 < iz && iz < fines[zone].lenz - 2) {
//         temp.txz[IdxTxzFi(ix, iz, cur, zone)] = (
//             + 0.8 * core.txz[IdxTxzFi(ix, iz, cur, zone)]
//             + 0.04 * (
//                 + core.txz[IdxTxzFi(ix - 1, iz, cur, zone)]
//                 + core.txz[IdxTxzFi(ix + 1, iz, cur, zone)]
//                 + core.txz[IdxTxzFi(ix, iz - 1, cur, zone)]
//                 + core.txz[IdxTxzFi(ix, iz + 1, cur, zone)]
//             )
//             + 0.01 * (
//                 + core.txz[IdxTxzFi(ix - 1, iz - 1, cur, zone)]
//                 + core.txz[IdxTxzFi(ix + 1, iz - 1, cur, zone)]
//                 + core.txz[IdxTxzFi(ix - 1, iz + 1, cur, zone)]
//                 + core.txz[IdxTxzFi(ix + 1, iz + 1, cur, zone)]
//             )
//         );
//     }

//     if (0 < ix && ix < fines[zone].lenx - 1 && 0 < iz && iz < fines[zone].lenz - 1) {
//         temp.sx[IdxSigFi(ix, iz, cur, zone)] = (
//             + 0.8 * core.sx[IdxSigFi(ix, iz, cur, zone)]
//             + 0.04 * (
//                 + core.sx[IdxSigFi(ix - 1, iz, cur, zone)]
//                 + core.sx[IdxSigFi(ix + 1, iz, cur, zone)]
//                 + core.sx[IdxSigFi(ix, iz - 1, cur, zone)]
//                 + core.sx[IdxSigFi(ix, iz + 1, cur, zone)]
//             )
//             + 0.01 * (
//                 + core.sx[IdxSigFi(ix - 1, iz - 1, cur, zone)]
//                 + core.sx[IdxSigFi(ix + 1, iz - 1, cur, zone)]
//                 + core.sx[IdxSigFi(ix - 1, iz + 1, cur, zone)]
//                 + core.sx[IdxSigFi(ix + 1, iz + 1, cur, zone)]
//             )
//         );

//         temp.sz[IdxSigFi(ix, iz, cur, zone)] = (
//             + 0.8 * core.sz[IdxSigFi(ix, iz, cur, zone)]
//             + 0.04 * (
//                 + core.sz[IdxSigFi(ix - 1, iz, cur, zone)]
//                 + core.sz[IdxSigFi(ix + 1, iz, cur, zone)]
//                 + core.sz[IdxSigFi(ix, iz - 1, cur, zone)]
//                 + core.sz[IdxSigFi(ix, iz + 1, cur, zone)]
//             )
//             + 0.01 * (
//                 + core.sz[IdxSigFi(ix - 1, iz - 1, cur, zone)]
//                 + core.sz[IdxSigFi(ix + 1, iz - 1, cur, zone)]
//                 + core.sz[IdxSigFi(ix - 1, iz + 1, cur, zone)]
//                 + core.sz[IdxSigFi(ix + 1, iz + 1, cur, zone)]
//             )
//         );

//         temp.p[IdxSigFi(ix, iz, cur, zone)] = (
//             + 0.8 * core.p[IdxSigFi(ix, iz, cur, zone)]
//             + 0.04 * (
//                 + core.p[IdxSigFi(ix - 1, iz, cur, zone)]
//                 + core.p[IdxSigFi(ix + 1, iz, cur, zone)]
//                 + core.p[IdxSigFi(ix, iz - 1, cur, zone)]
//                 + core.p[IdxSigFi(ix, iz + 1, cur, zone)]
//             )
//             + 0.01 * (
//                 + core.p[IdxSigFi(ix - 1, iz - 1, cur, zone)]
//                 + core.p[IdxSigFi(ix + 1, iz - 1, cur, zone)]
//                 + core.p[IdxSigFi(ix - 1, iz + 1, cur, zone)]
//                 + core.p[IdxSigFi(ix + 1, iz + 1, cur, zone)]
//             )
//         );

//         temp.r[IdxSigFi(ix, iz, cur, zone)] = (
//             + 0.8 * core.r[IdxSigFi(ix, iz, cur, zone)]
//             + 0.04 * (
//                 + core.r[IdxSigFi(ix - 1, iz, cur, zone)]
//                 + core.r[IdxSigFi(ix + 1, iz, cur, zone)]
//                 + core.r[IdxSigFi(ix, iz - 1, cur, zone)]
//                 + core.r[IdxSigFi(ix, iz + 1, cur, zone)]
//             )
//             + 0.01 * (
//                 + core.r[IdxSigFi(ix - 1, iz - 1, cur, zone)]
//                 + core.r[IdxSigFi(ix + 1, iz - 1, cur, zone)]
//                 + core.r[IdxSigFi(ix - 1, iz + 1, cur, zone)]
//                 + core.r[IdxSigFi(ix + 1, iz + 1, cur, zone)]
//             )
//         );
//     }
// }

// __global__ void sync_fine_to_coarse_str(Core core, int cur, int zone) {
//     int ix = blockIdx.x * blockDim.x + threadIdx.x;
//     int iz = blockIdx.y * blockDim.y + threadIdx.y;

//     int ix_in = ix + fines[zone].x_start;
//     int iz_in = iz + fines[zone].z_start;
//     int N = fines[zone].N;
//     if (ix_in > fines[zone].x_end || iz_in > fines[zone].z_end) {
//         return;
//     }

//     if (N == 1) {
//         if (tex1Dfetch<int>(sig_mask, iz_in * nx + ix_in) == -1) {
//             core.sx[IdxSigCo(ix_in, iz_in, cur)] = core.sx[IdxSigFi(ix, iz, cur, zone)];
//             core.sz[IdxSigCo(ix_in, iz_in, cur)] = core.sz[IdxSigFi(ix, iz, cur, zone)];
//         }
//         if (ix_in < fines[zone].x_end && iz_in < fines[zone].z_end && tex1Dfetch<int>(txz_mask, iz_in * (nx - 1) + ix_in) == -1) {
//             core.txz[IdxTxzCo(ix_in, iz_in, cur)] = core.txz[IdxTxzFi(ix, iz, cur, zone)];
//         }
//     } else {
//         if (tex1Dfetch<int>(sig_mask, iz_in * nx + ix_in) == -1) {
//             core.sx[IdxSigCo(ix_in, iz_in, cur)] = core.sx[IdxSigFi(ix * N, iz * N, cur, zone)];
//             core.sz[IdxSigCo(ix_in, iz_in, cur)] = core.sz[IdxSigFi(ix * N, iz * N, cur, zone)];
//         }
//         if (ix_in < fines[zone].x_end && iz_in < fines[zone].z_end && tex1Dfetch<int>(txz_mask, iz_in * (nx - 1) + ix_in) == -1) {
//             core.txz[IdxTxzCo(ix_in, iz_in, cur)] = core.txz[IdxTxzFi(ix * N + 1, iz * N + 1, cur, zone)];
//         }
//     }
// }

// __global__ void sync_fine_to_coarse_vel(Core core, int cur, int zone) {
//     int ix = blockIdx.x * blockDim.x + threadIdx.x;
//     int iz = blockIdx.y * blockDim.y + threadIdx.y;

//     int ix_in = ix + fines[zone].x_start;
//     int iz_in = iz + fines[zone].z_start;
//     int N = fines[zone].N;
//     if (ix_in > fines[zone].x_end || iz_in > fines[zone].z_end) {
//         return;
//     }
    
//     if (N == 1) {
//         if (ix_in < fines[zone].x_end && tex1Dfetch<int>(vx_mask, iz_in * (nx - 1) + ix_in) == -1) {
//             core.vx[IdxVxCo(ix_in, iz_in, cur)] = core.vx[IdxVxFi(ix, iz, cur, zone)];
//         }
//         if (iz_in < fines[zone].z_end && tex1Dfetch<int>(vz_mask, iz_in * nx + ix_in) == -1) {
//             core.vz[IdxVzCo(ix_in, iz_in, cur)] = core.vz[IdxVzFi(ix, iz, cur, zone)];
//         }
//     } else {
//         if (ix_in < fines[zone].x_end && tex1Dfetch<int>(vx_mask, iz_in * (nx - 1) + ix_in) == -1) {
//             core.vx[IdxVxCo(ix_in, iz_in, cur)] = core.vx[IdxVxFi(ix * N + 1, iz * N, cur, zone)];
//         }
//         if (iz_in < fines[zone].z_end && tex1Dfetch<int>(vz_mask, iz_in * nx + ix_in) == -1) {
//             core.vz[IdxVzCo(ix_in, iz_in, cur)] = core.vz[IdxVzFi(ix * N, iz * N + 1, cur, zone)];
//         }
//     }
// }