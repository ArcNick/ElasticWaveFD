#ifndef DIFFERENTIATE_CUH
#define DIFFERENTIATE_CUH

// 半网格点系数
static __constant__ float c[5] = {
    0, 1.196289f, -0.0797526f, 0.00957031f, -0.000697545f
};

// 整网格点系数
static __constant__ float d[5] = {
    0, 0.8f, -0.2f, 0.038095f, -0.00357143f
};

// 整网格点，向前差分

static __device__ __forceinline__ float Dx_int_8th(
    float *f, int ix, int iz, int ldx, float dx
) {
    float sum = 0.0f;
    for (int n = 1; n <= 4; n++) {
        sum += d[n] * (
            f[iz * ldx + (ix + n)] - f[iz * ldx + (ix - n + 1)]
        );
    }
    return sum / dx;
}

static __device__ __forceinline__ float Dz_int_8th(
    float *f, int ix, int iz, int ldx, float dz
) {
    float sum = 0.0f;
    for (int n = 1; n <= 4; n++) {
        sum += d[n] * (
            f[(iz + n) * ldx + ix] - f[(iz - n + 1) * ldx + ix]
        );
    }
    return sum / dz;
}

// 半网格点，向后差分

static __device__ __forceinline__ float Dx_half_8th(
    float *f, int ix, int iz, int ldx, float dx
) {
    float sum = 0.0f;
    for (int n = 1; n <= 4; n++) {
        sum += c[n] * (
            f[iz * ldx + (ix + n - 1)] - f[iz * ldx + (ix - n)]
        );
    }
    return sum / dx;
}

static __device__ __forceinline__ float Dz_half_8th(
    float *f, int ix, int iz, int ldx, float dz
) {
    float sum = 0.0f;
    for (int n = 1; n <= 4; n++) {
        sum += c[n] * (
            f[(iz + n - 1) * ldx + ix] - f[(iz - n) * ldx + ix]
        );
    }
    return sum / dz;
}

#endif