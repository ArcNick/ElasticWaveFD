#include "fd_stencil.cuh"
#include "coeffs.cuh"
#include "grid_manager.cuh"

__device__ int IdxVxCo(int ix, int iz, int time) {
    return iz * (nx - 1) + ix + time * offset_vx_all;
}
__device__ int IdxVzCo(int ix, int iz, int time) {
    return iz * nx + ix + time * offset_vz_all;
}
__device__ int IdxSigCo(int ix, int iz, int time) {
    return iz * nx + ix + time * offset_sig_all;
}
__device__ int IdxTxzCo(int ix, int iz, int time) {
    return iz * (nx - 1) + ix + time * offset_txz_all;
}
__device__ int IdxVxFi(int ix, int iz, int zone, int time) {
    return iz * (fines[zone].lenx - 1) + ix + time * offset_vx_all + sum_offset_fine_vx[zone];
}
__device__ int IdxVzFi(int ix, int iz, int zone, int time) {
    return iz * fines[zone].lenx + ix + time * offset_vz_all + sum_offset_fine_vz[zone];
}
__device__ int IdxSigFi(int ix, int iz, int zone, int time) {
    return iz * fines[zone].lenx + ix + time * offset_sig_all + sum_offset_fine_sig[zone];
}
__device__ int IdxTxzFi(int ix, int iz, int zone, int time) {
    return iz * (fines[zone].lenx - 1) + ix + time * offset_txz_all + sum_offset_fine_txz[zone];
}

// interpolation
__device__ float samp_rho_x_coarse(float *f, int ix, int iz) {
    float val1 = 0;
    float val2 = 0;
    int zone;
    int ix_fine;
    int iz_fine;
    zone = tex1Dfetch<int>(sig_mask, iz * nx + ix);
    if (zone == -1) {
        val1 = f[iz * nx + ix];
    } else {
        ix_fine = (ix - fines[zone].x_start) * fines[zone].N;
        iz_fine = (iz - fines[zone].z_start) * fines[zone].N;
        val1 = f[IdxSigFi(ix_fine, iz_fine, zone, 0)];
    }

    zone = tex1Dfetch<int>(sig_mask, iz * nx + ix + 1);
    if (zone == -1) {
        val2 = f[iz * nx + ix + 1];
    } else {
        ix_fine = (ix + 1 - fines[zone].x_start) * fines[zone].N;
        iz_fine = (iz - fines[zone].z_start) * fines[zone].N;
        val2 = f[IdxSigFi(ix_fine, iz_fine, zone, 0)];
    }
    
    return 2 / (1 / val1 + 1 / val2);
}

__device__ float samp_rho_z_coarse(float *f, int ix, int iz) {
    float val1 = 0;
    float val2 = 0;
    int zone;
    int ix_fine;
    int iz_fine;
    zone = tex1Dfetch<int>(sig_mask, iz * nx + ix);
    if (zone == -1) {
        val1 = f[iz * nx + ix];
    } else {
        ix_fine = (ix - fines[zone].x_start) * fines[zone].N;
        iz_fine = (iz - fines[zone].z_start) * fines[zone].N;
        val1 = f[IdxSigFi(ix_fine, iz_fine, zone, 0)];
    }

    zone = tex1Dfetch<int>(sig_mask, (iz + 1) * nx + ix);
    if (zone == -1) {
        val2 = f[(iz + 1) * nx + ix];
    } else {
        ix_fine = (ix - fines[zone].x_start) * fines[zone].N;
        iz_fine = (iz + 1 - fines[zone].z_start) * fines[zone].N;
        val2 = f[IdxSigFi(ix_fine, iz_fine, zone, 0)];
    }
    
    return 2 / (1 / val1 + 1 / val2);
}

__device__ float samp_rho_x_fine(float *f, int ix, int iz, int zone) {
    float val1 = f[IdxSigFi(ix, iz, zone, 0)];
    float val2 = f[IdxSigFi(ix + 1, iz, zone, 0)];
    return 2 / (1 / val1 + 1 / val2);
}

__device__ float samp_rho_z_fine(float *f, int ix, int iz, int zone) {
    float val1 = f[IdxSigFi(ix, iz, zone, 0)];
    float val2 = f[IdxSigFi(ix, iz + 1, zone, 0)];
    return 2 / (1 / val1 + 1 / val2);
}

__device__ float samp_C55_coarse(
    float *f, int ix, int iz
) {
    float sum = 0;
    int zone;
    int ix_fine;
    int iz_fine;

    zone = tex1Dfetch<int>(sig_mask, iz * nx + ix);
    if (zone == -1) {
        if (MAT(iz * nx + ix) <= VESOLID) {
            sum += f[iz * nx + ix];
        } else {
            return 0;
        }
    } else {
        ix_fine = (ix - fines[zone].x_start) * fines[zone].N;
        iz_fine = (iz - fines[zone].z_start) * fines[zone].N;
        if (MAT(IdxSigFi(ix_fine, iz_fine, zone, 0)) <= VESOLID) {
            sum += f[IdxSigFi(ix_fine, iz_fine, zone, 0)];
        } else {
            return 0;
        }
    }

    zone = tex1Dfetch<int>(sig_mask, (iz + 1) * nx + ix);
    if (zone == -1) {
        if (MAT((iz + 1) * nx + ix) <= VESOLID) {
            sum += f[(iz + 1) * nx + ix];
        } else {
            return 0;
        }
    } else {
        ix_fine = (ix - fines[zone].x_start) * fines[zone].N;
        iz_fine = (iz + 1 - fines[zone].z_start) * fines[zone].N;
        if (MAT(IdxSigFi(ix_fine, iz_fine, zone, 0)) <= VESOLID) {
            sum += f[IdxSigFi(ix_fine, iz_fine, zone, 0)];
        } else {
            return 0;
        }
    }

    zone = tex1Dfetch<int>(sig_mask, iz * nx + ix + 1);
    if (zone == -1) {
        if (MAT(iz * nx + ix + 1) <= VESOLID) {
            sum += f[iz * nx + ix + 1];
        } else {
            return 0;
        }
    } else {
        ix_fine = (ix + 1 - fines[zone].x_start) * fines[zone].N;
        iz_fine = (iz - fines[zone].z_start) * fines[zone].N;
        if (MAT(IdxSigFi(ix_fine, iz_fine, zone, 0)) <= VESOLID) {
            sum += f[IdxSigFi(ix_fine, iz_fine, zone, 0)];
        } else {
            return 0;
        }
    }

    zone = tex1Dfetch<int>(sig_mask, (iz + 1) * nx + ix + 1);
    if (zone == -1) {
        if (MAT((iz + 1) * nx + ix + 1) <= VESOLID) {
            sum += f[(iz + 1) * nx + ix + 1];
        } else {
            return 0;
        }
    } else {
        ix_fine = (ix + 1 - fines[zone].x_start) * fines[zone].N;
        iz_fine = (iz + 1 - fines[zone].z_start) * fines[zone].N;
        if (MAT(IdxSigFi(ix_fine, iz_fine, zone, 0)) <= VESOLID) {
            sum += f[IdxSigFi(ix_fine, iz_fine, zone, 0)];
        } else {
            return 0;
        }
    }
    return 0.25 * sum;
}

__device__ float samp_C55_fine(float *f, int ix, int iz, int zone) {
    float sum = 0;
    if (MAT(IdxSigFi(ix, iz, zone, 0)) <= VESOLID) {
        sum += f[IdxSigFi(ix, iz, zone, 0)];
    } else {
        return 0;
    }
    if (MAT(IdxSigFi(ix + 1, iz, zone, 0)) <= VESOLID) {
        sum += f[IdxSigFi(ix + 1, iz, zone, 0)];
    } else {
        return 0;
    }
    if (MAT(IdxSigFi(ix, iz + 1, zone, 0)) <= VESOLID) {
        sum += f[IdxSigFi(ix, iz + 1, zone, 0)];
    } else {
        return 0;
    }
    if (MAT(IdxSigFi(ix + 1, iz + 1, zone, 0)) <= VESOLID) {
        sum += f[IdxSigFi(ix + 1, iz + 1, zone, 0)];
    } else {
        return 0;
    }
    return 0.25 * sum;
}

__device__ float samp_sls_coarse(float *f, int ix, int iz) {
    float sum = 0;
    int zone;
    int ix_fine;
    int iz_fine;

    zone = tex1Dfetch<int>(sig_mask, iz * nx + ix);
    if (zone == -1) {
        if (MAT(iz * nx + ix) <= VESOLID) {
            sum += f[iz * nx + ix];
        } else {
            return 0;
        }
    } else {
        ix_fine = (ix - fines[zone].x_start) * fines[zone].N;
        iz_fine = (iz - fines[zone].z_start) * fines[zone].N;
        if (MAT(IdxSigFi(ix_fine, iz_fine, zone, 0)) <= VESOLID) {
            sum += f[IdxSigFi(ix_fine, iz_fine, zone, 0)];
        } else {
            return 0;
        }
    }

    zone = tex1Dfetch<int>(sig_mask, (iz + 1) * nx + ix);
    if (zone == -1) {
        if (MAT((iz + 1) * nx + ix) <= VESOLID) {
            sum += f[(iz + 1) * nx + ix];
        } else {
            return 0;
        }
    } else {
        ix_fine = (ix - fines[zone].x_start) * fines[zone].N;
        iz_fine = (iz + 1 - fines[zone].z_start) * fines[zone].N;
        if (MAT(IdxSigFi(ix_fine, iz_fine, zone, 0)) <= VESOLID) {
            sum += f[IdxSigFi(ix_fine, iz_fine, zone, 0)];
        } else {
            return 0;
        }
    }

    zone = tex1Dfetch<int>(sig_mask, iz * nx + ix + 1);
    if (zone == -1) {
        if (MAT(iz * nx + ix + 1) <= VESOLID) {
            sum += f[iz * nx + ix + 1];
        } else {
            return 0;
        }
    } else {
        ix_fine = (ix + 1 - fines[zone].x_start) * fines[zone].N;
        iz_fine = (iz - fines[zone].z_start) * fines[zone].N;
        if (MAT(IdxSigFi(ix_fine, iz_fine, zone, 0)) <= VESOLID) {
            sum += f[IdxSigFi(ix_fine, iz_fine, zone, 0)];
        } else {
            return 0;
        }
    }

    zone = tex1Dfetch<int>(sig_mask, (iz + 1) * nx + ix + 1);
    if (zone == -1) {
        if (MAT((iz + 1) * nx + ix + 1) <= VESOLID) {
            sum += f[(iz + 1) * nx + ix + 1];
        } else {
            return 0;
        }
    } else {
        ix_fine = (ix + 1 - fines[zone].x_start) * fines[zone].N;
        iz_fine = (iz + 1 - fines[zone].z_start) * fines[zone].N;
        if (MAT(IdxSigFi(ix_fine, iz_fine, zone, 0)) <= VESOLID) {
            sum += f[IdxSigFi(ix_fine, iz_fine, zone, 0)];
        } else {
            return 0;
        }
    }
    return sum * 0.25;
}
__device__ float samp_sls_fine(float *f, int ix, int iz, int zone) {
    float sum = 0;
    if (MAT(IdxSigFi(ix, iz, zone, 0)) <= VESOLID) {
        sum += f[IdxSigFi(ix, iz, zone, 0)];
    } else {
        return 0;
    }
    if (MAT(IdxSigFi(ix + 1, iz, zone, 0)) <= VESOLID) {
        sum += f[IdxSigFi(ix + 1, iz, zone, 0)];
    } else {
        return 0;
    }
    if (MAT(IdxSigFi(ix, iz + 1, zone, 0)) <= VESOLID) {
        sum += f[IdxSigFi(ix, iz + 1, zone, 0)];
    } else {
        return 0;
    }
    if (MAT(IdxSigFi(ix + 1, iz + 1, zone, 0)) <= VESOLID) {
        sum += f[IdxSigFi(ix + 1, iz + 1, zone, 0)];
    } else {
        return 0;
    }
    return sum * 0.25;
}

// vx 沿 z 方向插值（x 固定，z 变化）
__device__ float samp_vx_z(
    float *f, float ix_global, float iz_global
) {
    int ix = int(ix_global - 0.5f + 1e-5f);   // 修正：减去 0.5
    ix = max(0, min(ix, nx - 2));

    int iz0 = int(iz_global);
    float tz = iz_global - iz0;
    int tz_idx = int(tz * LUT_SIZE + 0.5f);
    tz_idx = (tz_idx < 0) ? 0 : (tz_idx >= LUT_SIZE ? LUT_SIZE - 1 : tz_idx);
    const float *coeff = &lagrange_coeff[tz_idx * LAGRANGE_ORDER];

    const int offset = LAGRANGE_ORDER / 2;
    float sum = 0.0f;
    int iz = 0;
    for (int k = 0; k < LAGRANGE_ORDER; ++k) {
        iz = iz0 + (k - offset);
        iz = max(0, min(iz, nz - 1));
        sum += coeff[k] * __ldg(f + iz * (nx - 1) + ix);
    }
    return sum;
}

// vx 沿 x 方向插值（z 固定，x 变化）
__device__ float samp_vx_x(
    float *f, float ix_global, float iz_global
) {
    int ix0 = int(ix_global - 0.5f + 1e-5f);
    ix0 = max(0, min(ix0, nx - 2));
    float tx = (ix_global - 0.5f) - ix0;

    int iz = int(iz_global + 1e-5f);
    iz = max(0, min(iz, nz - 1));

    int tx_idx = int(tx * LUT_SIZE + 0.5f);
    tx_idx = (tx_idx < 0) ? 0 : (tx_idx >= LUT_SIZE ? LUT_SIZE - 1 : tx_idx);
    const float* coeff = &lagrange_coeff[tx_idx * LAGRANGE_ORDER];

    const int offset = LAGRANGE_ORDER / 2;
    float sum = 0.0f;
    int ix = 0;
    for (int k = 0; k < LAGRANGE_ORDER; ++k) {
        ix = ix0 + (k - offset);
        ix = max(0, min(ix, nx - 2));
        sum += coeff[k] * __ldg(f + iz * (nx - 1) + ix);
    }
    return sum;
}

// vz 沿 x 方向插值（z 固定，x 变化）
__device__ float samp_vz_x(
    float *f, float ix_global, float iz_global
) {
    int iz = int(iz_global - 0.5f + 1e-5f);
    iz = max(0, min(iz, nz - 2));

    int ix0 = int(ix_global + 1e-5f);
    ix0 = max(0, min(ix0, nx - 1));
    float tx = ix_global - ix0;

    int tx_idx = int(tx * LUT_SIZE + 0.5f);
    tx_idx = (tx_idx < 0) ? 0 : (tx_idx >= LUT_SIZE ? LUT_SIZE - 1 : tx_idx);
    const float* coeff = &lagrange_coeff[tx_idx * LAGRANGE_ORDER];

    const int offset = LAGRANGE_ORDER / 2;
    float sum = 0.0f;
    int ix = 0;
    for (int k = 0; k < LAGRANGE_ORDER; ++k) {
        ix = ix0 + (k - offset);
        ix = max(0, min(ix, nx - 1));
        sum += coeff[k] * __ldg(f + iz * nx + ix);
    }
    return sum;
}

// vz 沿 z 方向插值（x 固定，z 变化）
__device__ float samp_vz_z(
    float *f, float ix_global, float iz_global
) {
    int ix = int(ix_global + 1e-5f);
    ix = max(0, min(ix, nx - 1));

    int iz0 = int(iz_global - 0.5f + 1e-5f);
    iz0 = max(0, min(iz0, nz - 2));
    float tz = (iz_global - 0.5f) - iz0;

    int tz_idx = int(tz * LUT_SIZE + 0.5f);
    tz_idx = (tz_idx < 0) ? 0 : (tz_idx >= LUT_SIZE ? LUT_SIZE - 1 : tz_idx);
    const float* coeff = &lagrange_coeff[tz_idx * LAGRANGE_ORDER];

    const int offset = LAGRANGE_ORDER / 2;
    float sum = 0.0f;
    int iz = 0;
    for (int k = 0; k < LAGRANGE_ORDER; ++k) {
        iz = iz0 + (k - offset);
        iz = max(0, min(iz, nz - 2));
        sum += coeff[k] * __ldg(f + iz * nx + ix);
    }
    return sum;
}

// sx 沿 z 方向插值（x 固定）
__device__ float samp_sx_z(
    float *f, float ix_global, float iz_global
) {
    int ix = int(ix_global);

    int iz0 = int(iz_global);
    float tz = iz_global - iz0;

    int tz_idx = int(tz * LUT_SIZE + 0.5f);
    tz_idx = (tz_idx < 0) ? 0 : (tz_idx >= LUT_SIZE ? LUT_SIZE - 1 : tz_idx);
    const float* coeff = &lagrange_coeff[tz_idx * LAGRANGE_ORDER];

    const int offset = LAGRANGE_ORDER / 2;
    float sum = 0.0f;
    int iz = 0;
    for (int k = 0; k < LAGRANGE_ORDER; ++k) {
        iz = iz0 + (k - offset);
        iz = max(0, min(iz, nz - 1));
        sum += coeff[k] * __ldg(f + iz * nx + ix);
    }
    return sum;
}

// sz 沿 x 方向插值（z 固定）
__device__ float samp_sz_x(
    float *f, float ix_global, float iz_global
) {
    int ix0 = int(ix_global);
    int iz = int(iz_global);

    float tx = ix_global - ix0;
    int tx_idx = int(tx * LUT_SIZE + 0.5f);
    tx_idx = (tx_idx < 0) ? 0 : (tx_idx >= LUT_SIZE ? LUT_SIZE - 1 : tx_idx);
    const float* coeff = &lagrange_coeff[tx_idx * LAGRANGE_ORDER];

    const int offset = LAGRANGE_ORDER / 2;
    float sum = 0.0f;
    int ix = 0;
    for (int k = 0; k < LAGRANGE_ORDER; ++k) {
        ix = ix0 + (k - offset);
        ix = max(0, min(ix, nx - 1));
        sum += coeff[k] * __ldg(f + iz * nx + ix);
    }
    return sum;
}

// txz 沿 x 方向插值（z 固定）
__device__ float samp_txz_x(
    float *f, float ix_global, float iz_global
) {
    int iz0 = int(iz_global - 0.5f);
    iz0 = max(0, min(iz0, nz - 2));

    int ix0 = int(ix_global - 0.5f);
    ix0 = max(0, min(ix0, nx - 2));
    float tx = (ix_global - 0.5f) - ix0;

    int tx_idx = int(tx * LUT_SIZE + 0.5f);
    tx_idx = (tx_idx < 0) ? 0 : (tx_idx >= LUT_SIZE ? LUT_SIZE - 1 : tx_idx);
    const float* coeff = &lagrange_coeff[tx_idx * LAGRANGE_ORDER];

    const int offset = LAGRANGE_ORDER / 2;
    float sum = 0.0f;
    int ix = 0;
    for (int k = 0; k < LAGRANGE_ORDER; ++k) {
        ix = ix0 + (k - offset);
        ix = max(0, min(ix, nx - 2));
        sum += coeff[k] * __ldg(f + iz0 * (nx - 1) + ix);
    }
    return sum;
}

// txz 沿 z 方向插值（x 固定）
__device__ float samp_txz_z(
    float *f, float ix_global, float iz_global
) {
    int ix = int(ix_global - 0.5f + 1e-5f);
    ix = max(0, min(ix, nx - 2));

    int iz0 = int(iz_global - 0.5f + 1e-5f);
    iz0 = max(0, min(iz0, nz - 2));
    float tz = (iz_global - 0.5f) - iz0;

    int tz_idx = int(tz * LUT_SIZE + 0.5f);
    tz_idx = (tz_idx < 0) ? 0 : (tz_idx >= LUT_SIZE ? LUT_SIZE - 1 : tz_idx);
    const float* coeff = &lagrange_coeff[tz_idx * LAGRANGE_ORDER];

    const int offset = LAGRANGE_ORDER / 2;
    float sum = 0.0f;
    int iz = 0;
    for (int k = 0; k < LAGRANGE_ORDER; ++k) {
        iz = iz0 + (k - offset);
        iz = max(0, min(iz, nz - 2));
        sum += coeff[k] * __ldg(f + iz * (nx - 1) + ix);
    }
    return sum;
}

// coarse
__device__ float dvx_dx_coarse(
    float *f, int ix, int iz, int time
) {
    float sum = 0;
    f += time * offset_vx_all;
    int mask;
    int local_ix;
    int local_iz;
    for (int i = 1; i <= 4; i++) {
        // samp1
        mask = tex1Dfetch<int>(vx_mask, (iz * (nx - 1) + ix + i - 1));
        if (mask != -1) {
            local_ix = (
                ix + i - 1 + 0.5 - fines[mask].x_start - 1.0 / (2 * fines[mask].N)
            ) * fines[mask].N;
            local_iz = (iz - fines[mask].z_start) * fines[mask].N;
            sum += c_normal[i] * f[IdxVxFi(local_ix, local_iz, mask, 0)];
        } else {
            sum += c_normal[i] * f[iz * (nx - 1) + (ix + i - 1)];
        }

        // samp2
        mask = tex1Dfetch<int>(vx_mask, (iz * (nx - 1) + ix - i));
        if (mask != -1) {
            local_ix = (
                ix - i + 0.5 - fines[mask].x_start - 1.0 / (2 * fines[mask].N)
            ) * fines[mask].N;
            local_iz = (iz - fines[mask].z_start) * fines[mask].N;
            sum -= c_normal[i] * f[IdxVxFi(local_ix, local_iz, mask, 0)];
        } else {
            sum -= c_normal[i] * f[iz * (nx - 1) + (ix - i)];
        }
    }
    return sum / dx;
}

__device__ float dvx_dz_coarse(
    float *f, int ix, int iz, int time
) {
    float sum = 0;
    f += time * offset_vx_all;
    int mask;
    int local_ix;
    int local_iz;
    for (int i = 1; i <= 4; i++) {
        mask = tex1Dfetch<int>(vx_mask, (iz + i) * (nx - 1) + ix);
        if (mask != -1) {
            local_ix = (
                ix + 0.5 - fines[mask].x_start - 1.0 / (2 * fines[mask].N)
            ) * fines[mask].N;
            local_iz = (iz + i - fines[mask].z_start) * fines[mask].N;
            sum += d_normal[i] * f[IdxVxFi(local_ix, local_iz, mask, 0)];
        } else {
            sum += d_normal[i] * f[(iz + i) * (nx - 1) + ix];
        }

        mask = tex1Dfetch<int>(vx_mask, (iz - i + 1) * (nx - 1) + ix);
        if (mask != -1) {
            local_ix = (
                ix + 0.5 - fines[mask].x_start - 1.0 / (2 * fines[mask].N)
            ) * fines[mask].N;
            local_iz = (iz - i + 1 - fines[mask].z_start) * fines[mask].N;
            sum -= d_normal[i] * f[IdxVxFi(local_ix, local_iz, mask, 0)];
        } else {
            sum -= d_normal[i] * f[(iz - i + 1) * (nx - 1) + ix];
        }
    }
    return sum / dz;
}

__device__ float dvz_dx_coarse(
    float *f, int ix, int iz, int time
) {
    float sum = 0;
    f += time * offset_vz_all;
    int mask;
    int local_ix;
    int local_iz;
    for (int i = 1; i <= 4; i++) {
        mask = tex1Dfetch<int>(vz_mask, (iz * nx + ix + i));
        if (mask != -1) {
            local_ix = (ix + i - fines[mask].x_start) * fines[mask].N;
            local_iz = (iz + 0.5 - fines[mask].z_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            sum += d_normal[i] * f[IdxVzFi(local_ix, local_iz, mask, 0)];
        } else {
            sum += d_normal[i] * f[iz * nx + ix + i];
        }

        mask = tex1Dfetch<int>(vz_mask, (iz * nx + ix - i + 1));
        if (mask != -1) {
            local_ix = (ix - i + 1 - fines[mask].x_start) * fines[mask].N;
            local_iz = (iz + 0.5 - fines[mask].z_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            sum -= d_normal[i] * f[IdxVzFi(local_ix, local_iz, mask, 0)];
        } else {
            sum -= d_normal[i] * f[iz * nx + ix - i + 1];
        }
    }
    return sum / dx;
}

__device__ float dvz_dz_coarse(
    float *f, int ix, int iz, int time
) {
    float sum = 0;
    f += time * offset_vz_all;
    int mask;
    int local_ix;
    int local_iz;
    for (int i = 1; i <= 4; i++) {
        mask = tex1Dfetch<int>(vz_mask, (iz + i - 1) * nx + ix);
        if (mask != -1) {
            local_iz = (iz + i - 1 + 0.5 - fines[mask].z_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            local_ix = (ix - fines[mask].x_start) * fines[mask].N;
            sum += c_normal[i] * f[IdxVzFi(local_ix, local_iz, mask, 0)];
        } else {
            sum += c_normal[i] * f[(iz + i - 1) * nx + ix];
        }

        mask = tex1Dfetch<int>(vz_mask, (iz - i) * nx + ix);
        if (mask != -1) {
            local_iz = (iz - i + 0.5 - fines[mask].z_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            local_ix = (ix - fines[mask].x_start) * fines[mask].N;
            sum -= c_normal[i] * f[IdxVzFi(local_ix, local_iz, mask, 0)];
        } else {
            sum -= c_normal[i] * f[(iz - i) * nx + ix];
        }
    }
    return sum / dz;
}

__device__ float dsx_dx_coarse(
    float *f, int ix, int iz, int time
) {
    float sum = 0;
    f += time * offset_sig_all;
    int mask;
    int local_ix;
    int local_iz;
    for (int i = 1; i <= 4; i++) {
        mask = tex1Dfetch<int>(sig_mask, (iz * nx + ix + i));
        if (mask != -1) {
            local_ix = (ix + i - fines[mask].x_start) * fines[mask].N;
            local_iz = (iz - fines[mask].z_start) * fines[mask].N;
            sum += d_normal[i] * f[IdxSigFi(local_ix, local_iz, mask, 0)];
        } else {
            sum += d_normal[i] * f[iz * nx + ix + i];
        }

        mask = tex1Dfetch<int>(sig_mask, (iz * nx + ix - i + 1));
        if (mask != -1) {
            local_ix = (ix - i + 1 - fines[mask].x_start) * fines[mask].N;
            local_iz = (iz - fines[mask].z_start) * fines[mask].N;
            sum -= d_normal[i] * f[IdxSigFi(local_ix, local_iz, mask, 0)];
        } else {
            sum -= d_normal[i] * f[iz * nx + ix - i + 1];
        }
    }
    return sum / dx;
}

__device__ float dsz_dz_coarse(
    float *f, int ix, int iz, int time
) {
    float sum = 0;
    f += time * offset_sig_all;
    int mask;
    int local_ix;
    int local_iz;
    for (int i = 1; i <= 4; i++) {
        mask = tex1Dfetch<int>(sig_mask, (iz + i) * nx + ix);
        if (mask != -1) {
            local_ix = (ix - fines[mask].x_start) * fines[mask].N;
            local_iz = (iz + i - fines[mask].z_start) * fines[mask].N;
            sum += d_normal[i] * f[IdxSigFi(local_ix, local_iz, mask, 0)];
        } else {
            sum += d_normal[i] * f[(iz + i) * nx + ix];
        }

        mask = tex1Dfetch<int>(sig_mask, (iz - i + 1) * nx + ix);
        if (mask != -1) {
            local_ix = (ix - fines[mask].x_start) * fines[mask].N;
            local_iz = (iz - i + 1 - fines[mask].z_start) * fines[mask].N;
            sum -= d_normal[i] * f[IdxSigFi(local_ix, local_iz, mask, 0)];
        } else {
            sum -= d_normal[i] * f[(iz - i + 1) * nx + ix];
        }
    }
    return sum / dz;
}

__device__ float dtxz_dx_coarse(
    float *f, int ix, int iz, int time
) {
    float sum = 0;
    f += time * offset_txz_all;
    int mask;
    int local_ix;
    int local_iz;
    for (int i = 1; i <= 4; i++) {
        mask = tex1Dfetch<int>(txz_mask, (iz * (nx - 1) + ix + i - 1));
        if (mask != -1) {
            local_ix = (ix + i - 1 + 0.5 - fines[mask].x_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            local_iz = (iz + 0.5 - fines[mask].z_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            sum += c_normal[i] * f[IdxTxzFi(local_ix, local_iz, mask, 0)];
        } else {
            sum += c_normal[i] * f[iz * (nx - 1) + ix + i - 1];
        }

        mask = tex1Dfetch<int>(txz_mask, (iz * (nx - 1) + ix - i));
        if (mask != -1) {
            local_ix = (ix - i + 0.5 - fines[mask].x_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            local_iz = (iz + 0.5 - fines[mask].z_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            sum -= c_normal[i] * f[IdxTxzFi(local_ix, local_iz, mask, 0)];
        } else {
            sum -= c_normal[i] * f[iz * (nx - 1) + ix - i];
        }
    }
    return sum / dx;
}

__device__ float dtxz_dz_coarse(
    float *f, int ix, int iz, int time
) {
    float sum = 0;
    f += time * offset_txz_all;
    int mask;
    int local_ix;
    int local_iz;
    for (int i = 1; i <= 4; i++) {
        mask = tex1Dfetch<int>(txz_mask, (iz + i - 1) * (nx - 1) + ix);
        if (mask != -1) {
            local_ix = (ix + 0.5 - fines[mask].x_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            local_iz = (iz + i - 1 + 0.5 - fines[mask].z_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            sum += c_normal[i] * f[IdxTxzFi(local_ix, local_iz, mask, 0)];
        } else {
            sum += c_normal[i] * f[(iz + i - 1) * (nx - 1) + ix];
        }

        mask = tex1Dfetch<int>(txz_mask, (iz - i) * (nx - 1) + ix);
        if (mask != -1) {
            local_ix = (ix + 0.5 - fines[mask].x_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            local_iz = (iz - i + 0.5 - fines[mask].z_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            sum -= c_normal[i] * f[IdxTxzFi(local_ix, local_iz, mask, 0)];
        } else {
            sum -= c_normal[i] * f[(iz - i) * (nx - 1) + ix];
        }
    }
    return sum / dz;
}

// fine
__device__ float dvx_dx_8th(
    float *f, int ix, int iz, int zone, int time
) {
    float sum = 0;
    f += time * offset_vx_all;
    int lenx = fines[zone].lenx;
    int n_x = tex1Dfetch<int2>(
        vx_n_tex, sum_offset_fine_sig[zone] - sum_offset_fine_sig[0] + iz * lenx + ix
    ).x;

    float ix_global, iz_global;
    int ix_local;
    
    for (int i = 1; i <= n_x; i++) {
        sum += c[fines[zone].N][n_x][i] * (
            + f[sum_offset_fine_vx[zone] + iz * (lenx - 1) + ix + i - 1]
            - f[sum_offset_fine_vx[zone] + iz * (lenx - 1) + ix - i]
        );
    }
    
    for (int i = n_x + 1; i <= 4; i++) {
        // samp1
        ix_global = (
            + 1.0 * ix / fines[zone].N + fines[zone].x_start 
            + 1.0 * n_x / fines[zone].N + (i - n_x - 0.5)//
        );
        iz_global = 1.0 * iz / fines[zone].N + fines[zone].z_start;

        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end && fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
            ix_local = (
                ix_global - fines[zone].x_start - 1.0 / (2 * fines[zone].N)
            ) * fines[zone].N;
            sum += c[fines[zone].N][n_x][i] * f[sum_offset_fine_vx[zone] + iz * (lenx - 1) + ix_local];
        } else {
            sum += c[fines[zone].N][n_x][i] * samp_vx_z(f, ix_global, iz_global);
        }

        // samp2
        ix_global = (
            + 1.0 * ix / fines[zone].N + fines[zone].x_start
            - 1.0 * n_x / fines[zone].N - (i - n_x - 0.5)
        );
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end && fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
            ix_local = (
                ix_global - fines[zone].x_start - 1.0 / (2 * fines[zone].N)
            ) * fines[zone].N;
            sum -= c[fines[zone].N][n_x][i] * f[sum_offset_fine_vx[zone] + iz * (lenx - 1) + ix_local];
        } else {
            sum -= c[fines[zone].N][n_x][i] * samp_vx_z(f, ix_global, iz_global);
        }
    }
    return sum / fines[zone].dx_fine;
}

__device__ float dvx_dz_8th(
    float *f, int ix, int iz, int zone, int time
) {
    float sum = 0;
    f += time * offset_vx_all;
    int lenx = fines[zone].lenx;
    int n_z = tex1Dfetch<int2>(
        vx_n_tex, sum_offset_fine_sig[zone] - sum_offset_fine_sig[0] + iz * lenx + ix
    ).y;
    float ix_global, iz_global;
    int iz_local;

    for (int i = 1; i <= n_z; i++) {
        sum += d[fines[zone].N][n_z][i] * (
            + f[sum_offset_fine_vx[zone] + (iz + i) * (lenx - 1) + ix]
            - f[sum_offset_fine_vx[zone] + (iz - i + 1) * (lenx - 1) + ix]
        ); 
    }

    for (int i = n_z + 1; i <= 4; i++) {
        // samp1
        ix_global = (ix + 0.5) / fines[zone].N + fines[zone].x_start;
        iz_global = (
            + (iz + 0.5) / fines[zone].N + fines[zone].z_start
            + (n_z - 0.5) / fines[zone].N + (i - n_z)
        );
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end && fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
            iz_local = (iz_global - fines[zone].z_start) * fines[zone].N;
            sum += d[fines[zone].N][n_z][i] * f[sum_offset_fine_vx[zone] + iz_local * (lenx - 1) + ix];
        } else {
            sum += d[fines[zone].N][n_z][i] * samp_vx_x(f, ix_global, iz_global);
        }

        // samp2
        iz_global = (
            + (iz + 0.5) / fines[zone].N + fines[zone].z_start
            - (n_z - 0.5) / fines[zone].N - (i - n_z)
        );
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end && fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
            iz_local = (iz_global - fines[zone].z_start) * fines[zone].N;
            sum -= d[fines[zone].N][n_z][i] * f[sum_offset_fine_vx[zone] + iz_local * (lenx - 1) + ix];
        } else {
            sum -= d[fines[zone].N][n_z][i] * samp_vx_x(f, ix_global, iz_global);
        }
    }
    return sum / fines[zone].dz_fine;
}

__device__ float dvz_dx_8th(
    float *f, int ix, int iz, int zone, int time
) {
    float sum = 0;
    f += time * offset_vz_all;
    int lenx = fines[zone].lenx;
    int n_x = tex1Dfetch<int2>(
        vz_n_tex, sum_offset_fine_sig[zone] - sum_offset_fine_sig[0] + iz * lenx + ix
    ).x;
    float ix_global, iz_global;
    int ix_local;

    for (int i = 1; i <= n_x; i++) {
        sum += d[fines[zone].N][n_x][i] * (
            + f[sum_offset_fine_vz[zone] + iz * lenx + ix + i]
            - f[sum_offset_fine_vz[zone] + iz * lenx + ix - i + 1]
        );
    }

    for (int i = n_x + 1; i <= 4; i++) {
        // samp1
        ix_global = (
            + (ix + 0.5) / fines[zone].N + fines[zone].x_start
            + (n_x - 0.5) / fines[zone].N + (i - n_x)
        );
        iz_global = (iz + 0.5) / fines[zone].N + fines[zone].z_start;
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end && fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
            ix_local = (ix_global - fines[zone].x_start) * fines[zone].N;
            sum += d[fines[zone].N][n_x][i] * f[sum_offset_fine_vz[zone] + iz * lenx + ix_local];
        } else {
            sum += d[fines[zone].N][n_x][i] * samp_vz_z(f, ix_global, iz_global);
        }

        // samp2
        ix_global = (
            + (ix + 0.5) / fines[zone].N + fines[zone].x_start
            - (n_x - 0.5) / fines[zone].N - (i - n_x)
        );
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end && fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
            ix_local = (ix_global - fines[zone].x_start) * fines[zone].N;
            sum -= d[fines[zone].N][n_x][i] * f[sum_offset_fine_vz[zone] + iz * lenx + ix_local];
        } else {
            sum -= d[fines[zone].N][n_x][i] * samp_vz_z(f, ix_global, iz_global);
        }
    }
    return sum / fines[zone].dx_fine;
}

__device__ float dvz_dz_8th(
    float *f, int ix, int iz, int zone, int time
) {
    float sum = 0;
    f += time * offset_vz_all;
    int lenx = fines[zone].lenx;
    int n_z = tex1Dfetch<int2>(
        vz_n_tex, sum_offset_fine_sig[zone] - sum_offset_fine_sig[0] + iz * lenx + ix
    ).y;
    float ix_global, iz_global;
    int iz_local;

    for (int i = 1; i <= n_z; i++) {
        sum += c[fines[zone].N][n_z][i] * (
            + f[sum_offset_fine_vz[zone] + (iz + i - 1) * lenx + ix]
            - f[sum_offset_fine_vz[zone] + (iz - i) * lenx + ix]
        );
    }

    for (int i = n_z + 1; i <= 4; i++) {
        // samp1
        ix_global = 1.0 * ix / fines[zone].N + fines[zone].x_start;
        iz_global = (
            + 1.0 * iz / fines[zone].N + fines[zone].z_start
            + 1.0 * n_z / fines[zone].N + (i - n_z - 0.5)
        );
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end && fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
            iz_local = (
                iz_global - fines[zone].z_start - 1.0 / (2 * fines[zone].N)
            ) * fines[zone].N;
            sum += c[fines[zone].N][n_z][i] * f[sum_offset_fine_vz[zone] + iz_local * lenx + ix];
        } else {
            sum += c[fines[zone].N][n_z][i] * samp_vz_x(f, ix_global, iz_global);
        }

        // samp2
        iz_global = (
            + 1.0 * iz / fines[zone].N + fines[zone].z_start
            - 1.0 * n_z / fines[zone].N - (i - n_z - 0.5)
        );
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end && fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
            iz_local = (
                iz_global - fines[zone].z_start - 1.0 / (2 * fines[zone].N)
            ) * fines[zone].N;
            sum -= c[fines[zone].N][n_z][i] * f[sum_offset_fine_vz[zone] + iz_local * lenx + ix];
        } else {
            sum -= c[fines[zone].N][n_z][i] * samp_vz_x(f, ix_global, iz_global);
        }
    }
    return sum / fines[zone].dz_fine;
}

__device__ float dsx_dx_8th(
    float *f, int ix, int iz, int zone, int time
) {
    float sum = 0;
    f += time * offset_sig_all;
    int lenx = fines[zone].lenx;
    int n_x = tex1Dfetch<int2>(
        sig_n_tex, sum_offset_fine_sig[zone] - sum_offset_fine_sig[0] + iz * lenx + ix
    ).x;
    float ix_global, iz_global;
    int ix_local;
    
    for (int i = 1; i <= n_x; i++) {
        sum += d[fines[zone].N][n_x][i] * (
            + f[sum_offset_fine_sig[zone] + iz * lenx + ix + i]
            - f[sum_offset_fine_sig[zone] + iz * lenx + ix - i + 1]
        );
    }

    for (int i = n_x + 1; i <= 4; i++) {
        // samp1
        ix_global = (
            + (ix + 0.5) / fines[zone].N + fines[zone].x_start
            + (n_x - 0.5) / fines[zone].N + (i - n_x)
        );
        iz_global = 1.0 * iz / fines[zone].N + fines[zone].z_start;
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end && fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
            ix_local = (ix_global - fines[zone].x_start) * fines[zone].N;
            sum += d[fines[zone].N][n_x][i] * f[sum_offset_fine_sig[zone] + iz * lenx + ix_local];
        } else {
            sum += d[fines[zone].N][n_x][i] * samp_sx_z(f, ix_global, iz_global);
        }

        // samp2
        ix_global = (
            + (ix + 0.5) / fines[zone].N + fines[zone].x_start
            - (n_x - 0.5) / fines[zone].N - (i - n_x)
        );
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end && fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
            ix_local = (ix_global - fines[zone].x_start) * fines[zone].N;
            sum -= d[fines[zone].N][n_x][i] * f[sum_offset_fine_sig[zone] + iz * lenx + ix_local];
        } else {
            sum -= d[fines[zone].N][n_x][i] * samp_sx_z(f, ix_global, iz_global);
        }   
    }
    return sum / fines[zone].dx_fine;
}

__device__ float dsz_dz_8th(
    float *f, int ix, int iz, int zone, int time
) {
    float sum = 0;
    f += time * offset_sig_all;
    int lenx = fines[zone].lenx;
    int n_z = tex1Dfetch<int2>(
        sig_n_tex, sum_offset_fine_sig[zone] - sum_offset_fine_sig[0] + iz * lenx + ix
    ).y;
    float ix_global, iz_global;
    int iz_local;

    for (int i = 1; i <= n_z; i++) {
        sum += d[fines[zone].N][n_z][i] * (
            + f[sum_offset_fine_sig[zone] + (iz + i) * lenx + ix]
            - f[sum_offset_fine_sig[zone] + (iz - i + 1) * lenx + ix]
        );
    }

    for (int i = n_z + 1; i <= 4; i++) {
        // samp1
        ix_global = 1.0 * ix / fines[zone].N + fines[zone].x_start;
        iz_global = (
            + (iz + 0.5) / fines[zone].N + fines[zone].z_start
            + (n_z - 0.5) / fines[zone].N + (i - n_z)
        );
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end && fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
            iz_local = (iz_global - fines[zone].z_start) * fines[zone].N;
            sum += d[fines[zone].N][n_z][i] * f[sum_offset_fine_sig[zone] + iz_local * lenx + ix];
        } else {
            sum += d[fines[zone].N][n_z][i] * samp_sz_x(f, ix_global, iz_global);
        }

        // samp2
        iz_global = (
            + (iz + 0.5) / fines[zone].N + fines[zone].z_start
            - (n_z - 0.5) / fines[zone].N - (i - n_z)
        );
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end && fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
            iz_local = (iz_global - fines[zone].z_start) * fines[zone].N;
            sum -= d[fines[zone].N][n_z][i] * f[sum_offset_fine_sig[zone] + iz_local * lenx + ix];
        } else {
            sum -= d[fines[zone].N][n_z][i] * samp_sz_x(f, ix_global, iz_global);
        }
    }
    return sum / fines[zone].dz_fine;
}

__device__ float dtxz_dx_8th(
    float *f, int ix, int iz, int zone, int time
) {
    float sum = 0;
    f += time * offset_txz_all;
    int lenx = fines[zone].lenx;
    int n_x = tex1Dfetch<int2>(
        txz_n_tex, sum_offset_fine_sig[zone] - sum_offset_fine_sig[0] + iz * lenx + ix
    ).x;
    float ix_global, iz_global;
    int ix_local;

    for (int i = 1; i <= n_x; i++) {
        sum += c[fines[zone].N][n_x][i] * (
            + f[sum_offset_fine_txz[zone] + iz * (lenx - 1) + ix + i - 1]
            - f[sum_offset_fine_txz[zone] + iz * (lenx - 1) + ix - i]
        );
    }

    for (int i = n_x + 1; i <= 4; i++) {
        // samp1
        ix_global = (
            + 1.0 * ix / fines[zone].N + fines[zone].x_start
            + 1.0 * n_x / fines[zone].N + (i - n_x - 0.5)
        );
        iz_global = (iz + 0.5) / fines[zone].N + fines[zone].z_start;
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end && fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
            ix_local = (
                ix_global - fines[zone].x_start - 1.0 / (2 * fines[zone].N)
            ) * fines[zone].N;
            sum += d[fines[zone].N][n_x][i] * f[sum_offset_fine_txz[zone] + iz * (lenx - 1) + ix_local];
        } else {
            sum += d[fines[zone].N][n_x][i] * samp_txz_z(f, ix_global, iz_global);
        }

        // samp2
        ix_global = (
            + 1.0 * ix / fines[zone].N + fines[zone].x_start
            - 1.0 * (n_x) / fines[zone].N - (i - n_x - 0.5)
        );
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end && fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
            ix_local = (
                ix_global - fines[zone].x_start - 1.0 / (2 * fines[zone].N)
            ) * fines[zone].N;
            sum -= d[fines[zone].N][n_x][i] * f[sum_offset_fine_txz[zone] + iz * (lenx - 1) + ix_local];
        } else {
            sum -= d[fines[zone].N][n_x][i] * samp_txz_z(f, ix_global, iz_global);
        }
    }
    return sum / fines[zone].dx_fine;
}

__device__ float dtxz_dz_8th(
    float *f, int ix, int iz, int zone, int time
) {
    float sum = 0;
    f += time * offset_txz_all;
    int lenx = fines[zone].lenx;
    int n_z = tex1Dfetch<int2>(
        txz_n_tex, sum_offset_fine_sig[zone] - sum_offset_fine_sig[0] + iz * lenx + ix
    ).y;
    float ix_global, iz_global;
    int iz_local;

    for (int i = 1; i <= n_z; i++) {
        sum += c[fines[zone].N][n_z][i] * (
            + f[sum_offset_fine_txz[zone] + (iz + i - 1) * (lenx - 1) + ix]
            - f[sum_offset_fine_txz[zone] + (iz - i) * (lenx - 1) + ix]
        );
    }

    for (int i = n_z + 1; i <= 4; i++) {
        // samp1
        ix_global = (ix + 0.5) / fines[zone].N + fines[zone].x_start;
        iz_global = (
            + 1.0 * iz / fines[zone].N + fines[zone].z_start
            + n_z / fines[zone].N + (i - n_z - 0.5)
        );
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end && fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
            iz_local = (
                iz_global - fines[zone].z_start - 1.0 / (2 * fines[zone].N)
            ) * fines[zone].N;
            sum += c[fines[zone].N][n_z][i] * f[sum_offset_fine_txz[zone] + iz_local * (lenx - 1) + ix];
        } else {
            sum += c[fines[zone].N][n_z][i] * samp_txz_x(f, ix_global, iz_global);
        }

        // samp2
        iz_global = (
            1.0 * iz / fines[zone].N + fines[zone].z_start
            - n_z / fines[zone].N - (i - n_z - 0.5)
        );
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end && fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
            iz_local = (
                iz_global - fines[zone].z_start - 1.0 / (2 * fines[zone].N)
            ) * fines[zone].N;
            sum -= c[fines[zone].N][n_z][i] * f[sum_offset_fine_txz[zone] + iz_local * (lenx - 1) + ix];
        } else {
            sum -= c[fines[zone].N][n_z][i] * samp_txz_x(f, ix_global, iz_global);
        }
    }
    return sum / fines[zone].dz_fine;
}