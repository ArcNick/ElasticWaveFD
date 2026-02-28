#include "fd_stencil.cuh"
#include "coeffs.cuh"
#include "grid_manager.cuh"

// interpolation
__device__ float samp_rho_x(
    float *f, int ix, int iz
) {
    float sum = 0;
    int zone;
    int ix_fine;
    int iz_fine;

    zone = tex1Dfetch<int>(sx_mask, iz * nx + ix);
    if (zone == -1) {
        sum += f[iz * nx + ix];
    } else {
        ix_fine = (ix - fines[zone].x_start) * fines[zone].N;
        iz_fine = (iz - fines[zone].z_start) * fines[zone].N;
        sum += f[sum_offset_fine_sx[zone] + iz_fine * fines[zone].lenx + ix_fine];
    }

    zone = tex1Dfetch<int>(sx_mask, iz * nx + ix + 1);
    if (zone == -1) {
        sum += f[iz * nx + ix + 1];
    } else {
        ix_fine = (ix + 1 - fines[zone].x_start) * fines[zone].N;
        iz_fine = (iz - fines[zone].z_start) * fines[zone].N;
        sum += f[sum_offset_fine_sx[zone] + iz_fine * fines[zone].lenx + ix_fine];
    }

    return sum * 0.5;
}

__device__ float samp_rho_z(
    float *f, int ix, int iz
) {
    float sum = 0;
    int zone;
    int ix_fine;
    int iz_fine;

    zone = tex1Dfetch<int>(sx_mask, iz * nx + ix);
    if (zone == -1) {
        sum += f[iz * nx + ix];
    } else {
        ix_fine = (ix - fines[zone].x_start) * fines[zone].N;
        iz_fine = (iz - fines[zone].z_start) * fines[zone].N;
        sum += f[sum_offset_fine_sx[zone] + iz_fine * fines[zone].lenx + ix_fine];
    }

    zone = tex1Dfetch<int>(sx_mask, (iz + 1) * nx + ix);
    if (zone == -1) {
        sum += f[(iz + 1) * nx + ix];
    } else {
        ix_fine = (ix - fines[zone].x_start) * fines[zone].N;
        iz_fine = (iz + 1 - fines[zone].z_start) * fines[zone].N;
        sum += f[sum_offset_fine_sx[zone] + iz_fine * fines[zone].lenx + ix_fine];
    }

    return sum * 0.5;
}

__device__ float samp_C55(
    float *f, int ix, int iz
) {
    float sum = 0;
    int zone;
    int ix_fine;
    int iz_fine;

    zone = tex1Dfetch<int>(sx_mask, iz * nx + ix);
    if (zone == -1) {
        sum += f[iz * nx + ix];
    } else {
        ix_fine = (ix - fines[zone].x_start) * fines[zone].N;
        iz_fine = (iz - fines[zone].z_start) * fines[zone].N;
        sum += f[sum_offset_fine_sx[zone] + iz_fine * fines[zone].lenx + ix_fine];
    }

    zone = tex1Dfetch<int>(sx_mask, (iz + 1) * nx + ix);
    if (zone == -1) {
        sum += f[(iz + 1) * nx + ix];
    } else {
        ix_fine = (ix - fines[zone].x_start) * fines[zone].N;
        iz_fine = (iz + 1 - fines[zone].z_start) * fines[zone].N;
        sum += f[sum_offset_fine_sx[zone] + iz_fine * fines[zone].lenx + ix_fine];
    }

    zone = tex1Dfetch<int>(sx_mask, iz * nx + ix + 1);
    if (zone == -1) {
        sum += f[iz * nx + ix + 1];
    } else {
        ix_fine = (ix + 1 - fines[zone].x_start) * fines[zone].N;
        iz_fine = (iz - fines[zone].z_start) * fines[zone].N;
        sum += f[sum_offset_fine_sx[zone] + iz_fine * fines[zone].lenx + ix_fine];
    }

    zone = tex1Dfetch<int>(sx_mask, (iz + 1) * nx + ix + 1);
    if (zone == -1) {
        sum += f[(iz + 1) * nx + ix + 1];
    } else {
        ix_fine = (ix + 1 - fines[zone].x_start) * fines[zone].N;
        iz_fine = (iz + 1 - fines[zone].z_start) * fines[zone].N;
        sum += f[sum_offset_fine_sx[zone] + iz_fine * fines[zone].lenx + ix_fine];
    }

    return sum * 0.25;
}

__device__ float samp_vx_z(
    float *f, float ix_global, float iz_global
) {
    int ix = int(ix_global + 1e-5f);
    ix = max(0, min(ix, nx - 2));

    int iz0 = int(iz_global);
    float tz = iz_global - iz0;
    int tz_idx = int(tz * LUT_SIZE + 0.5f);
    tz_idx = (tz_idx < 0) ? 0 : (tz_idx >= LUT_SIZE ? LUT_SIZE - 1 : tz_idx);
    const float *coeff = &lagrange_coeff[tz_idx * LAGRANGE_ORDER];

    float sum = 0.0f;
    // sum += coeff[0] * __ldg(f + max(0, min(iz0 - 2, nz - 1)) * (nx - 1) + ix);
    sum += coeff[1] * __ldg(f + max(0, min(iz0 - 1, nz - 1)) * (nx - 1) + ix);
    sum += coeff[2] * __ldg(f + max(0, min(iz0, nz - 1))     * (nx - 1) + ix);
    sum += coeff[3] * __ldg(f + max(0, min(iz0 + 1, nz - 1)) * (nx - 1) + ix);
    sum += coeff[4] * __ldg(f + max(0, min(iz0 + 2, nz - 1)) * (nx - 1) + ix);
    // sum += coeff[5] * __ldg(f + max(0, min(iz0 + 3, nz - 1)) * (nx - 1) + ix);

    return sum;
}

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

    float sum = 0.0f;
    sum += coeff[0] * __ldg(f + iz * (nx - 1) + max(0, min(ix0 - 2, nx - 2)));
    sum += coeff[1] * __ldg(f + iz * (nx - 1) + max(0, min(ix0 - 1, nx - 2)));
    sum += coeff[2] * __ldg(f + iz * (nx - 1) + max(0, min(ix0, nx - 2)));
    sum += coeff[3] * __ldg(f + iz * (nx - 1) + max(0, min(ix0 + 1, nx - 2)));
    sum += coeff[4] * __ldg(f + iz * (nx - 1) + max(0, min(ix0 + 2, nx - 2)));
    sum += coeff[5] * __ldg(f + iz * (nx - 1) + max(0, min(ix0 + 3, nx - 2)));

    return sum;
}

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

    float sum = 0.0f;
    sum += coeff[0] * __ldg(f + iz * nx + max(0, min(ix0 - 2, nx - 1)));
    sum += coeff[1] * __ldg(f + iz * nx + max(0, min(ix0 - 1, nx - 1)));
    sum += coeff[2] * __ldg(f + iz * nx + max(0, min(ix0, nx - 1)));
    sum += coeff[3] * __ldg(f + iz * nx + max(0, min(ix0 + 1, nx - 1)));
    sum += coeff[4] * __ldg(f + iz * nx + max(0, min(ix0 + 2, nx - 1)));
    sum += coeff[5] * __ldg(f + iz * nx + max(0, min(ix0 + 3, nx - 1)));

    return sum;
}

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

    float sum = 0.0f;
    sum += coeff[0] * __ldg(f + max(0, min(iz0 - 2, nz - 2)) * nx + ix);
    sum += coeff[1] * __ldg(f + max(0, min(iz0 - 1, nz - 2)) * nx + ix);
    sum += coeff[2] * __ldg(f + max(0, min(iz0, nz - 2))     * nx + ix);
    sum += coeff[3] * __ldg(f + max(0, min(iz0 + 1, nz - 2)) * nx + ix);
    sum += coeff[4] * __ldg(f + max(0, min(iz0 + 2, nz - 2)) * nx + ix);
    sum += coeff[5] * __ldg(f + max(0, min(iz0 + 3, nz - 2)) * nx + ix);

    return sum;
}

__device__ float samp_sx_z(
    float *f, float ix_global, float iz_global
) {
    int ix = int(ix_global);

    int iz0 = int(iz_global);
    float tz = iz_global - iz0;

    int tz_idx = int(tz * LUT_SIZE + 0.5f);
    tz_idx = (tz_idx < 0) ? 0 : (tz_idx >= LUT_SIZE ? LUT_SIZE - 1 : tz_idx);
    const float* coeff = &lagrange_coeff[tz_idx * LAGRANGE_ORDER];

    float sum = 0.0f;
    sum += coeff[0] * __ldg(f + max(0, min(iz0 - 2, nz - 1)) * nx + ix);
    sum += coeff[1] * __ldg(f + max(0, min(iz0 - 1, nz - 1)) * nx + ix);
    sum += coeff[2] * __ldg(f + max(0, min(iz0, nz - 1))     * nx + ix);
    sum += coeff[3] * __ldg(f + max(0, min(iz0 + 1, nz - 1)) * nx + ix);
    sum += coeff[4] * __ldg(f + max(0, min(iz0 + 2, nz - 1)) * nx + ix);
    sum += coeff[5] * __ldg(f + max(0, min(iz0 + 3, nz - 1)) * nx + ix);

    return sum;
}

__device__ float samp_sz_x(
    float *f, float ix_global, float iz_global
) {
    int ix0 = int(ix_global);
    int iz = int(iz_global);

    float tx = ix_global - ix0;
    int tx_idx = int(tx * LUT_SIZE + 0.5f);
    tx_idx = (tx_idx < 0) ? 0 : (tx_idx >= LUT_SIZE ? LUT_SIZE - 1 : tx_idx);
    const float* coeff = &lagrange_coeff[tx_idx * LAGRANGE_ORDER];

    float sum = 0.0f;
    sum += coeff[0] * __ldg(f + iz * nx + max(0, min(ix0 - 2, nx - 1)));
    sum += coeff[1] * __ldg(f + iz * nx + max(0, min(ix0 - 1, nx - 1)));
    sum += coeff[2] * __ldg(f + iz * nx + max(0, min(ix0, nx - 1)));
    sum += coeff[3] * __ldg(f + iz * nx + max(0, min(ix0 + 1, nx - 1)));
    sum += coeff[4] * __ldg(f + iz * nx + max(0, min(ix0 + 2, nx - 1)));
    sum += coeff[5] * __ldg(f + iz * nx + max(0, min(ix0 + 3, nx - 1)));

    return sum;
}

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

    float sum = 0.0f;
    sum += coeff[0] * __ldg(f + iz0 * (nx - 1) + max(0, min(ix0 - 2, nx - 2)));
    sum += coeff[1] * __ldg(f + iz0 * (nx - 1) + max(0, min(ix0 - 1, nx - 2)));
    sum += coeff[2] * __ldg(f + iz0 * (nx - 1) + max(0, min(ix0, nx - 2)));
    sum += coeff[3] * __ldg(f + iz0 * (nx - 1) + max(0, min(ix0 + 1, nx - 2)));
    sum += coeff[4] * __ldg(f + iz0 * (nx - 1) + max(0, min(ix0 + 2, nx - 2)));
    sum += coeff[5] * __ldg(f + iz0 * (nx - 1) + max(0, min(ix0 + 3, nx - 2)));

    return sum;
}

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

    float sum = 0.0f;
    sum += coeff[0] * __ldg(f + max(0, min(iz0 - 2, nz - 2)) * (nx - 1) + ix);
    sum += coeff[1] * __ldg(f + max(0, min(iz0 - 1, nz - 2)) * (nx - 1) + ix);
    sum += coeff[2] * __ldg(f + max(0, min(iz0, nz - 2))     * (nx - 1) + ix);
    sum += coeff[3] * __ldg(f + max(0, min(iz0 + 1, nz - 2)) * (nx - 1) + ix);
    sum += coeff[4] * __ldg(f + max(0, min(iz0 + 2, nz - 2)) * (nx - 1) + ix);
    sum += coeff[5] * __ldg(f + max(0, min(iz0 + 3, nz - 2)) * (nx - 1) + ix);

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
    int lenx;
    for (int i = 1; i <= 4; i++) {
        // samp1
        mask = tex1Dfetch<int>(vx_mask, (iz * (nx - 1) + ix + i - 1));
        if (mask != -1) {
            local_ix = (
                ix + i - 1 + 0.5 - fines[mask].x_start - 1.0 / (2 * fines[mask].N)
            ) * fines[mask].N;
            local_iz = (iz - fines[mask].z_start) * fines[mask].N;
            lenx = fines[mask].lenx;
            sum += c_normal[i] * f[sum_offset_fine_vx[mask] + local_iz * (lenx - 1) + local_ix];
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
            lenx = fines[mask].lenx;
            sum -= c_normal[i] * f[sum_offset_fine_vx[mask] + local_iz * (lenx - 1) + local_ix];
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
    int lenx;
    for (int i = 1; i <= 4; i++) {
        mask = tex1Dfetch<int>(vx_mask, (iz + i) * (nx - 1) + ix);
        if (mask != -1) {
            local_ix = (
                ix + 0.5 - fines[mask].x_start - 1.0 / (2 * fines[mask].N)
            ) * fines[mask].N;
            local_iz = (iz + i - fines[mask].z_start) * fines[mask].N;
            lenx = fines[mask].lenx;
            sum += d_normal[i] * f[sum_offset_fine_vx[mask] + local_iz * (lenx - 1) + local_ix];
        } else {
            sum += d_normal[i] * f[(iz + i) * (nx - 1) + ix];
        }

        mask = tex1Dfetch<int>(vx_mask, (iz - i + 1) * (nx - 1) + ix);
        if (mask != -1) {
            local_ix = (
                ix + 0.5 - fines[mask].x_start - 1.0 / (2 * fines[mask].N)
            ) * fines[mask].N;
            local_iz = (iz - i + 1 - fines[mask].z_start) * fines[mask].N;
            lenx = fines[mask].lenx;
            sum -= d_normal[i] * f[sum_offset_fine_vx[mask] + local_iz * (lenx - 1) + local_ix];
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
    int lenx;
    for (int i = 1; i <= 4; i++) {
        mask = tex1Dfetch<int>(vz_mask, (iz * nx + ix + i));
        if (mask != -1) {
            local_ix = (ix + i - fines[mask].x_start) * fines[mask].N;
            local_iz = (iz + 0.5 - fines[mask].z_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            lenx = fines[mask].lenx;
            sum += d_normal[i] * f[sum_offset_fine_vz[mask] + local_iz * lenx + local_ix];
        } else {
            sum += d_normal[i] * f[iz * nx + ix + i];
        }

        mask = tex1Dfetch<int>(vz_mask, (iz * nx + ix - i + 1));
        if (mask != -1) {
            local_ix = (ix - i + 1 - fines[mask].x_start) * fines[mask].N;
            local_iz = (iz + 0.5 - fines[mask].z_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            lenx = fines[mask].lenx;
            sum -= d_normal[i] * f[sum_offset_fine_vz[mask] + local_iz * lenx + local_ix];
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
    int lenx;
    for (int i = 1; i <= 4; i++) {
        mask = tex1Dfetch<int>(vz_mask, (iz + i - 1) * nx + ix);
        if (mask != -1) {
            local_iz = (iz + i - 1 + 0.5 - fines[mask].z_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            local_ix = (ix - fines[mask].x_start) * fines[mask].N;
            lenx = fines[mask].lenx;
            sum += c_normal[i] * f[sum_offset_fine_vz[mask] + local_iz * lenx + local_ix];
        } else {
            sum += c_normal[i] * f[(iz + i - 1) * nx + ix];
        }

        mask = tex1Dfetch<int>(vz_mask, (iz - i) * nx + ix);
        if (mask != -1) {
            local_iz = (iz - i + 0.5 - fines[mask].z_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            local_ix = (ix - fines[mask].x_start) * fines[mask].N;
            lenx = fines[mask].lenx;
            sum -= c_normal[i] * f[sum_offset_fine_vz[mask] + local_iz * lenx + local_ix];
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
    f += time * offset_sx_all;
    int mask;
    int local_ix;
    int local_iz;
    int lenx;
    for (int i = 1; i <= 4; i++) {
        mask = tex1Dfetch<int>(sx_mask, (iz * nx + ix + i));
        if (mask != -1) {
            local_ix = (ix + i - fines[mask].x_start) * fines[mask].N;
            local_iz = (iz - fines[mask].z_start) * fines[mask].N;
            lenx = fines[mask].lenx;
            sum += d_normal[i] * f[sum_offset_fine_sx[mask] + local_iz * lenx + local_ix];
        } else {
            sum += d_normal[i] * f[iz * nx + ix + i];
        }

        mask = tex1Dfetch<int>(sx_mask, (iz * nx + ix - i + 1));
        if (mask != -1) {
            local_ix = (ix - i + 1 - fines[mask].x_start) * fines[mask].N;
            local_iz = (iz - fines[mask].z_start) * fines[mask].N;
            lenx = fines[mask].lenx;
            sum -= d_normal[i] * f[sum_offset_fine_sx[mask] + local_iz * lenx + local_ix];
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
    f += time * offset_sz_all;
    int mask;
    int local_ix;
    int local_iz;
    int lenx;
    for (int i = 1; i <= 4; i++) {
        mask = tex1Dfetch<int>(sz_mask, (iz + i) * nx + ix);
        if (mask != -1) {
            local_ix = (ix - fines[mask].x_start) * fines[mask].N;
            local_iz = (iz + i - fines[mask].z_start) * fines[mask].N;
            lenx = fines[mask].lenx;
            sum += d_normal[i] * f[sum_offset_fine_sz[mask] + local_iz * lenx + local_ix];
        } else {
            sum += d_normal[i] * f[(iz + i) * nx + ix];
        }

        mask = tex1Dfetch<int>(sz_mask, (iz - i + 1) * nx + ix);
        if (mask != -1) {
            local_ix = (ix - fines[mask].x_start) * fines[mask].N;
            local_iz = (iz - i + 1 - fines[mask].z_start) * fines[mask].N;
            lenx = fines[mask].lenx;
            sum -= d_normal[i] * f[sum_offset_fine_sz[mask] + local_iz * lenx + local_ix];
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
    int lenx;
    for (int i = 1; i <= 4; i++) {
        mask = tex1Dfetch<int>(txz_mask, (iz * (nx - 1) + ix + i - 1));
        if (mask != -1) {
            local_ix = (ix + i - 1 + 0.5 - fines[mask].x_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            local_iz = (iz + 0.5 - fines[mask].z_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            lenx = fines[mask].lenx;
            sum += c_normal[i] * f[sum_offset_fine_txz[mask] + local_iz * (lenx - 1) + local_ix];
        } else {
            sum += c_normal[i] * f[iz * (nx - 1) + ix + i - 1];
        }

        mask = tex1Dfetch<int>(txz_mask, (iz * (nx - 1) + ix - i));
        if (mask != -1) {
            local_ix = (ix - i + 0.5 - fines[mask].x_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            local_iz = (iz + 0.5 - fines[mask].z_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            lenx = fines[mask].lenx;
            sum -= c_normal[i] * f[sum_offset_fine_txz[mask] + local_iz * (lenx - 1) + local_ix];
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
    int lenx;
    for (int i = 1; i <= 4; i++) {
        mask = tex1Dfetch<int>(txz_mask, (iz + i - 1) * (nx - 1) + ix);
        if (mask != -1) {
            local_ix = (ix + 0.5 - fines[mask].x_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            local_iz = (iz + i - 1 + 0.5 - fines[mask].z_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            lenx = fines[mask].lenx;
            sum += c_normal[i] * f[sum_offset_fine_txz[mask] + local_iz * (lenx - 1) + local_ix];
        } else {
            sum += c_normal[i] * f[(iz + i - 1) * (nx - 1) + ix];
        }

        mask = tex1Dfetch<int>(txz_mask, (iz - i) * (nx - 1) + ix);
        if (mask != -1) {
            local_ix = (ix + 0.5 - fines[mask].x_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            local_iz = (iz - i + 0.5 - fines[mask].z_start - 1.0 / (2 * fines[mask].N)) * fines[mask].N;
            lenx = fines[mask].lenx;
            if (sum_offset_fine_txz[mask] + local_iz * (lenx - 1) + local_ix >= offset_txz_all) {
                printf("local_ix=%d, local_iz=%d, ix=%d, iz=%d, i=%d\n", local_ix, local_iz, ix, iz, i);
                // printf("ix=%d, iz=%d, i=%d\n", ix, iz, i);
            }
            sum -= c_normal[i] * f[sum_offset_fine_txz[mask] + local_iz * (lenx - 1) + local_ix];
        } else {
            sum -= c_normal[i] * f[(iz - i) * (nx - 1) + ix];
        }
    }
    return sum / dz;
}

// fine
__device__ float dvx_dx_8th(
    float *f, int ix, int iz, int time, int zone
) {
    float sum = 0;
    f += time * offset_vx_all;
    int lenx = fines[zone].lenx;
    int n_x = tex1Dfetch<int2>(
        vx_n_tex, sum_offset_fine_vx[zone] - (nx - 1) * nz + iz * (lenx - 1) + ix
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
            + 1.0 * n_x / fines[zone].N + (i - n_x - 0.5)
        );
        iz_global = 1.0 * iz / fines[zone].N + fines[zone].z_start;

        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end) {
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
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end) {
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
    float *f, int ix, int iz, int time, int zone
) {
    float sum = 0;
    f += time * offset_vx_all;
    int lenx = (fines[zone].x_end - fines[zone].x_start) * fines[zone].N + 1;
    int n_z = tex1Dfetch<int2>(
        vx_n_tex, sum_offset_fine_vx[zone] - (nx - 1) * nz + iz * (lenx - 1) + ix
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
        if (fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
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
        if (fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
            iz_local = (iz_global - fines[zone].z_start) * fines[zone].N;
            sum -= d[fines[zone].N][n_z][i] * f[sum_offset_fine_vx[zone] + iz_local * (lenx - 1) + ix];
        } else {
            sum -= d[fines[zone].N][n_z][i] * samp_vx_x(f, ix_global, iz_global);
        }
    }
    return sum / fines[zone].dz_fine;
}

__device__ float dvz_dx_8th(
    float *f, int ix, int iz, int time, int zone
) {
    float sum = 0;
    f += time * offset_vz_all;
    int lenx = (fines[zone].x_end - fines[zone].x_start) * fines[zone].N + 1;
    int n_x = tex1Dfetch<int2>(
        vz_n_tex, sum_offset_fine_vz[zone] - nx * (nz - 1) + iz * lenx + ix
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
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end) {
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
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end) {
            ix_local = (ix_global - fines[zone].x_start) * fines[zone].N;
            sum -= d[fines[zone].N][n_x][i] * f[sum_offset_fine_vz[zone] + iz * lenx + ix_local];
        } else {
            sum -= d[fines[zone].N][n_x][i] * samp_vz_z(f, ix_global, iz_global);
        }
    }
    return sum / fines[zone].dx_fine;
}

__device__ float dvz_dz_8th(
    float *f, int ix, int iz, int time, int zone
) {
    float sum = 0;
    f += time * offset_vz_all;
    int lenx = (fines[zone].x_end - fines[zone].x_start) * fines[zone].N + 1;
    int n_z = tex1Dfetch<int2>(
        vz_n_tex, sum_offset_fine_vz[zone] - nx * (nz - 1) + iz * lenx + ix
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
        if (fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
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
        if (fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
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
    float *f, int ix, int iz, int time, int zone
) {
    float sum = 0;
    f += time * offset_sx_all;
    int lenx = (fines[zone].x_end - fines[zone].x_start) * fines[zone].N + 1;
    int n_x = tex1Dfetch<int2>(
        sx_n_tex, sum_offset_fine_sx[zone] - nx * nz + iz * lenx + ix
    ).x;
    float ix_global, iz_global;
    int ix_local;
    
    for (int i = 1; i <= n_x; i++) {
        sum += d[fines[zone].N][n_x][i] * (
            + f[sum_offset_fine_sx[zone] + iz * lenx + ix + i]
            - f[sum_offset_fine_sx[zone] + iz * lenx + ix - i + 1]
        );
    }

    for (int i = n_x + 1; i <= 4; i++) {
        // samp1
        ix_global = (
            + (ix + 0.5) / fines[zone].N + fines[zone].x_start
            + (n_x - 0.5) / fines[zone].N + (i - n_x)
        );
        iz_global = 1.0 * iz / fines[zone].N + fines[zone].z_start;
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end) {
            ix_local = (ix_global - fines[zone].x_start) * fines[zone].N;
            sum += d[fines[zone].N][n_x][i] * f[sum_offset_fine_sx[zone] + iz * lenx + ix_local];
        } else {
            sum += d[fines[zone].N][n_x][i] * samp_sx_z(f, ix_global, iz_global);
        }

        // samp2
        ix_global = (
            + (ix + 0.5) / fines[zone].N + fines[zone].x_start
            - (n_x - 0.5) / fines[zone].N - (i - n_x)
        );
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end) {
            ix_local = (ix_global - fines[zone].x_start) * fines[zone].N;
            sum -= d[fines[zone].N][n_x][i] * f[sum_offset_fine_sx[zone] + iz * lenx + ix_local];
        } else {
            sum -= d[fines[zone].N][n_x][i] * samp_sx_z(f, ix_global, iz_global);
        }   
    }
    return sum / fines[zone].dx_fine;
}

__device__ float dsz_dz_8th(
    float *f, int ix, int iz, int time, int zone
) {
    float sum = 0;
    f += time * offset_sz_all;
    int lenx = (fines[zone].x_end - fines[zone].x_start) * fines[zone].N + 1;
    int n_z = tex1Dfetch<int2>(
        sz_n_tex, sum_offset_fine_sz[zone] - nx * nz + iz * lenx + ix
    ).y;
    float ix_global, iz_global;
    int iz_local;

    for (int i = 1; i <= n_z; i++) {
        sum += d[fines[zone].N][n_z][i] * (
            + f[sum_offset_fine_sz[zone] + (iz + i) * lenx + ix]
            - f[sum_offset_fine_sz[zone] + (iz - i + 1) * lenx + ix]
        );
    }

    for (int i = n_z + 1; i <= 4; i++) {
        // samp1
        ix_global = 1.0 * ix / fines[zone].N + fines[zone].x_start;
        iz_global = (
            + (iz + 0.5) / fines[zone].N + fines[zone].z_start
            + (n_z - 0.5) / fines[zone].N + (i - n_z)
        );
        if (fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
            iz_local = (iz_global - fines[zone].z_start) * fines[zone].N;
            sum += d[fines[zone].N][n_z][i] * f[sum_offset_fine_sz[zone] + iz_local * lenx + ix];
        } else {
            sum += d[fines[zone].N][n_z][i] * samp_sz_x(f, ix_global, iz_global);
        }

        // samp2
        iz_global = (
            + (iz + 0.5) / fines[zone].N + fines[zone].z_start
            - (n_z - 0.5) / fines[zone].N - (i - n_z)
        );
        if (fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
            iz_local = (iz_global - fines[zone].z_start) * fines[zone].N;
            sum -= d[fines[zone].N][n_z][i] * f[sum_offset_fine_sz[zone] + iz_local * lenx + ix];
        } else {
            sum -= d[fines[zone].N][n_z][i] * samp_sz_x(f, ix_global, iz_global);
        }
    }
    return sum / fines[zone].dz_fine;
}

__device__ float dtxz_dx_8th(
    float *f, int ix, int iz, int time, int zone
) {
    float sum = 0;
    f += time * offset_txz_all;
    int lenx = (fines[zone].x_end - fines[zone].x_start) * fines[zone].N + 1;
    int n_x = tex1Dfetch<int2>(
        txz_n_tex, sum_offset_fine_txz[zone] - (nx - 1) * (nz - 1) + iz * (lenx - 1) + ix
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
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end) {
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
        if (fines[zone].x_start <= ix_global && ix_global <= fines[zone].x_end) {
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
    float *f, int ix, int iz, int time, int zone
) {
    float sum = 0;
    f += time * offset_txz_all;
    int lenx = (fines[zone].x_end - fines[zone].x_start) * fines[zone].N + 1;
    int n_z = tex1Dfetch<int2>(
        txz_n_tex, sum_offset_fine_txz[zone] - (nx - 1) * (nz - 1) + iz * (lenx - 1) + ix
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
        if (fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
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
        if (fines[zone].z_start <= iz_global && iz_global <= fines[zone].z_end) {
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