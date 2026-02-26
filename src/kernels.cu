#include "kernels.cuh"
#include "fd_stencil.cuh"

extern __constant__ int posx_d, posz_d;
extern __constant__ float dt_d;

__device__ int IdxVxCo(int ix, int iz, int time) {
    return iz * (nx - 1) + ix + time * offset_vx_all;
}
__device__ int IdxVzCo(int ix, int iz, int time) {
    return iz * nx + ix + time * offset_vz_all;
}
__device__ int IdxSxCo(int ix, int iz, int time) {
    return iz * nx + ix + time * offset_sx_all;
}
__device__ int IdxSzCo(int ix, int iz, int time) {
    return iz * nx + ix + time * offset_sz_all;
}
__device__ int IdxTxzCo(int ix, int iz, int time) {
    return iz * (nx - 1) + ix + time * offset_txz_all;
}
__device__ int IdxVxFi(int ix, int iz, int time, int zone) {
    return iz * (fines[zone].lenx - 1) + ix + time * offset_vx_all + sum_offset_fine_vx[zone];
}
__device__ int IdxVzFi(int ix, int iz, int time, int zone) {
    return iz * fines[zone].lenx + ix + time * offset_vz_all + sum_offset_fine_vz[zone];
}
__device__ int IdxSxFi(int ix, int iz, int time, int zone) {
    return iz * fines[zone].lenx + ix + time * offset_sx_all + sum_offset_fine_sx[zone];
}
__device__ int IdxSzFi(int ix, int iz, int time, int zone) {
    return iz * fines[zone].lenx + ix + time * offset_sz_all + sum_offset_fine_sz[zone];
}
__device__ int IdxTxzFi(int ix, int iz, int time, int zone) {
    return iz * (fines[zone].lenx - 1) + ix + time * offset_txz_all + sum_offset_fine_txz[zone];
}

//======== original ========//
// __device__ int get_cpml_idx_x_int(int ix) {
//     if (ix - 3 >= 0 && ix - 3 < thickness_d) {
//         return thickness_d - 1 - (ix - 3);
//     } else if (nx - 1 - ix - 4 >= 0 && nx - 1 - ix - 4 < thickness_d) {
//         return thickness_d - 1 - (nx - 1 - ix - 4);
//     } else {
//         return thickness_d;
//     }
// }

// __device__ int get_cpml_idx_z_int(int iz) {
//     if (iz - 3 >= 0 && iz - 3 < thickness_d) {
//         return thickness_d - 1 - (iz - 3);
//     } else if (nz - 1 - iz - 4 >= 0 && nz - 1 - iz - 4) {
//         return thickness_d - 1 - (nz - 1 - iz - 4);
//     } else {
//         return thickness_d;
//     }
// }

// __device__ int get_cpml_idx_x_half(int ix) {
//     if (ix - 4 >= 0 && ix - 4 < thickness_d - 1) {
//         return thickness_d - 1 - (ix - 4);
//     } else if (nx - 2 - ix - 3 >= 0 && nx - 2 - ix - 3 < thickness_d - 1) {
//         return thickness_d - 1 - (nx - 2 - ix - 3);
//     } else {
//         return thickness_d - 1;
//     }
// }

// __device__ int get_cpml_idx_z_half(int iz) {
//     if (iz - 4 >= 0 && iz - 4 < thickness_d - 1) {
//         return thickness_d - 1 - (iz - 4);
//     } else if (nz - 2 - iz - 3 >= 0 && nz - 2 - iz - 3 < thickness_d - 1) {
//         return thickness_d - 1 - (nz - 2 - iz - 3);
//     } else {
//         return thickness_d - 1;
//     }
// }

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
    core.sx[posz_d * nx + posx_d + cur * offset_sx_all] += src;
    core.sz[posz_d * nx + posx_d + cur * offset_sz_all] += src;
}

__global__ void update_stress_coarse(Core core, Model model, PsiVel psi_vel, int cur) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < 3 || ix >= nx - 4 || iz < 3 || iz >= nz - 4) {
        return;
    }

    if (tex1Dfetch<int>(sx_mask, iz * nx + ix) == -1) {
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

        core.sx[IdxSxCo(ix, iz, cur)] = (
            + core.sx[IdxSxCo(ix, iz, cur ^ 1)]
            + dt_d * (
                + model.C11[IdxSxCo(ix, iz, 0)] * dvx_dx 
                + model.C13[IdxSxCo(ix, iz, 0)] * dvz_dz
            )
        );
        core.sz[IdxSzCo(ix, iz, cur)] = (
            + core.sz[IdxSzCo(ix, iz, cur ^ 1)]
            + dt_d * (
                + model.C13[IdxSzCo(ix, iz, 0)] * dvx_dx
                + model.C33[IdxSzCo(ix, iz, 0)] * dvz_dz
            )
        );

        // NaN checks
        // if (isnan(dvx_dx)) {
        //     printf("update_stress_coarse dvx_dx (%d,%d)\n", ix, iz);
        // }
        // if (isnan(dvz_dz)) {
        //     printf("update_stress_coarse dvz_dz (%d,%d)\n", ix, iz);
        // }
    } else {
        printf("mask!!!!\n");
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
            + dt_d * samp_C55(model.C55, ix, iz) * (dvz_dx + dvx_dz)
        );

        // NaN checks
        // if (isnan(dvz_dx)) {
        //     printf("update_stress_coarse dvz_dx (%d,%d)\n", ix, iz);
        // }
        // if (isnan(dvx_dz)) {
        //     printf("update_stress_coarse dvx_dz (%d,%d)\n", ix, iz);
        // }
    }else {
        printf("mask!!!!\n");
    }

}

__global__ void update_velocity_coarse(Core core, Model model, PsiStr psi_str, int cur) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < 3 || ix >= nx - 4 || iz < 3 || iz >= nz - 4) {
        return;
    }

    if (tex1Dfetch<int>(vx_mask, iz * (nx - 1) + ix) == -1) {
        float dsx_dx = 0;
        float dtxz_dz = 0;
        dsx_dx = dsx_dx_coarse(core.sx, ix, iz, cur);
        if (iz > 3) {
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
            dt_d / samp_rho_x(model.rho, ix, iz) * (dsx_dx + dtxz_dz)
        );

        // NaN checks
        // if (isnan(dsx_dx)) {
        //     printf("update_velocity_coarse dsx_dx (%d,%d)\n", ix, iz);
        // }
        // if (isnan(dtxz_dz)) {
        //     printf("update_velocity_coarse dtxz_dz (%d,%d)\n", ix, iz);
        // }
    }else {
        printf("mask!!!!\n");
    }


    if (tex1Dfetch<int>(vz_mask, iz * (nx - 1) + ix) == -1) {
        float dsz_dz = 0;
        float dtxz_dx = 0;
        dsz_dz = dsz_dz_coarse(core.sz, ix, iz, cur);
        if (ix > 3) {
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
            dt_d / samp_rho_z(model.rho, ix, iz) * (dtxz_dx + dsz_dz)
        );

        // NaN checks
        // if (isnan(dsz_dz)) {
        //     printf("update_velocity_coarse dsz_dz (%d,%d)\n", ix, iz);
        // }
        // if (isnan(dtxz_dx)) {
        //     printf("update_velocity_coarse dtxz_dx (%d,%d)\n", ix, iz);
        // }
    }else {
        printf("mask!!!!\n");
    }

}

__global__ void update_stress_fine(Core core, Model model, int cur, int zone) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= fines[zone].lenx || iz >= fines[zone].lenz) {
        return;
    }

    float dvx_dx = 0;
    float dvx_dz = 0;
    float dvz_dx = 0;
    float dvz_dz = 0;
    
    if (ix < fines[zone].lenx - 1) {
        dvx_dx = dvx_dx_8th(core.vx, ix, iz, cur ^ 1, zone);
        dvx_dz = dvx_dz_8th(core.vx, ix, iz, cur ^ 1, zone);
    }
    if (iz < fines[zone].lenz - 1) {
        dvz_dx = dvz_dx_8th(core.vz, ix, iz, cur ^ 1, zone);
        dvz_dz = dvz_dz_8th(core.vz, ix, iz, cur ^ 1, zone);
    }

    core.sx[IdxSxFi(ix, iz, cur, zone)] = core.sx[IdxSxFi(ix, iz, cur ^ 1, zone)] + (
        dt_d * (
            + model.C11[IdxSxFi(ix, iz, 0, zone)] * dvx_dx
            + model.C13[IdxSxFi(ix, iz, 0, zone)] * dvz_dz
        )
    );
    core.sz[IdxSzFi(ix, iz, cur, zone)] = core.sz[IdxSzFi(ix, iz, cur ^ 1, zone)] + (
        dt_d * (
            + model.C13[IdxSzFi(ix, iz, 0, zone)] * dvx_dx
            + model.C33[IdxSzFi(ix, iz, 0, zone)] * dvz_dz
        )
    );

    // NaN checks
    if (isnan(dvx_dx)) {
        printf("update_stress_fine dvx_dx (%d,%d)\n", ix, iz);
    }
    if (isnan(dvz_dz)) {
        printf("update_stress_fine dvz_dz (%d,%d)\n", ix, iz);
    }

    if (ix < fines[zone].lenx - 1 && iz < fines[zone].lenz - 1) {
        core.txz[IdxTxzFi(ix, iz, cur, zone)] = core.txz[IdxTxzFi(ix, iz, cur ^ 1, zone)] + (
            dt_d * 0.25 * (
                + model.C55[IdxSxFi(ix, iz, 0, zone)]
                + model.C55[IdxSxFi(ix + 1, iz, 0, zone)]
                + model.C55[IdxSxFi(ix, iz + 1, 0, zone)]
                + model.C55[IdxSxFi(ix + 1, iz + 1, 0, zone)]
            ) * (dvz_dx + dvx_dz)
        );

        // NaN checks
        if (isnan(dvz_dx)) {
            printf("update_stress_fine dvz_dx (%d,%d)\n", ix, iz);
        }
        if (isnan(dvx_dz)) {
            printf("update_stress_fine dvx_dz (%d,%d)\n", ix, iz);
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
    
    dsx_dx = dsx_dx_8th(core.sx, ix, iz, cur, zone);
    dsz_dz = dsz_dz_8th(core.sz, ix, iz, cur, zone);
    if (ix < fines[zone].lenx - 1 && iz < fines[zone].lenz - 1) {
        dtxz_dx = dtxz_dx_8th(core.txz, ix, iz, cur, zone);
        dtxz_dz = dtxz_dz_8th(core.txz, ix, iz, cur, zone);
    }
    
    if (ix < fines[zone].lenx - 1) {
        core.vx[IdxVxFi(ix, iz, cur, zone)] = core.vx[IdxVxFi(ix, iz, cur ^ 1, zone)] + (
            dt_d * 2 / (
                + model.rho[IdxSxFi(ix, iz, 0, zone)]
                + model.rho[IdxSxFi(ix + 1, iz, 0, zone)]
            ) * (dsx_dx + dtxz_dz)
        );

        // NaN checks
        if (isnan(dsx_dx)) {
            printf("update_velocity_fine dsx_dx (%d,%d)\n", ix, iz);
        }
        if (isnan(dtxz_dz)) {
            printf("update_velocity_fine dtxz_dz (%d,%d)\n", ix, iz);
        }
    }
    if (iz < fines[zone].lenz - 1) {
        core.vz[IdxVzFi(ix, iz, cur, zone)] = core.vz[IdxVzFi(ix, iz, cur ^ 1, zone)] + (
            dt_d * 2 / (
                + model.rho[IdxSxFi(ix, iz, 0, zone)]
                + model.rho[IdxSxFi(ix, iz + 1, 0, zone)]
            ) * (dtxz_dx + dsz_dz)
        );

        // NaN checks
        if (isnan(dtxz_dx)) {
            printf("update_velocity_fine dtxz_dx (%d,%d)\n", ix, iz);
        }
        if (isnan(dsz_dz)) {
            printf("update_velocity_fine dsz_dz (%d,%d)\n", ix, iz);
        }
    }
}