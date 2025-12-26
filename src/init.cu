#include "common.cuh"
#include <memory>
#include <cmath>
#include <cstdio>
#include <array>
#include <cstdlib>
#include <cuda_runtime.h>

void Params::read(const char *file) {
    FILE *fp = fopen(file, "r");
    if (fp == nullptr) {
        printf("Failed to open parameter file %s\n", file);
        exit(1);
    }

    fscanf(fp, "fpeak = %f\n", &fpeak);
    fscanf(fp, "nx = %d\n", &nx);
    fscanf(fp, "nz = %d\n", &nz);
    fscanf(fp, "dx = %f\n", &dx);
    fscanf(fp, "dz = %f\n", &dz);
    fscanf(fp, "nt = %d\n", &nt);
    fscanf(fp, "dt = %f\n", &dt);
    fscanf(fp, "posx = %d\n", &posx);
    fscanf(fp, "posz = %d\n", &posz);
    fscanf(fp, "snapshot = %d\n", &snapshot);

    printf("Parameters loaded.\n");
    fclose(fp);
}

// begin ==========Thomsen 参数========== begin

Grid_Model_Thomsen::Grid_Model_Thomsen(int nx, int nz, bool mem_location) 
    : nx(nx), nz(nz), mem_location(mem_location) {
    if (mem_location == HOST_MEM) {
        vp0 = new float[nx * nz]();
        vs0 = new float[nx * nz]();
        rho = new float[nx * nz]();
        epsilon = new float[nx * nz]();
        delta = new float[nx * nz]();
        gamma = new float[nx * nz]();
        C11 = new float[nx * nz]();
        C13 = new float[nx * nz]();
        C33 = new float[nx * nz]();
        C44 = new float[nx * nz]();
        C66 = new float[nx * nz]();
    } else {
        cudaMalloc((void**)&vp0, nx * nz * sizeof(float));
        cudaMalloc((void**)&vs0, nx * nz * sizeof(float));
        cudaMalloc((void**)&rho, nx * nz * sizeof(float));
        cudaMalloc((void**)&epsilon, nx * nz * sizeof(float));
        cudaMalloc((void**)&delta, nx * nz * sizeof(float));
        cudaMalloc((void**)&gamma, nx * nz * sizeof(float));
        cudaMalloc((void**)&C11, nx * nz * sizeof(float));
        cudaMalloc((void**)&C13, nx * nz * sizeof(float));
        cudaMalloc((void**)&C33, nx * nz * sizeof(float));
        cudaMalloc((void**)&C44, nx * nz * sizeof(float));
        cudaMalloc((void**)&C66, nx * nz * sizeof(float));

        cudaMemset(vp0, 0, nx * nz * sizeof(float));
        cudaMemset(vs0, 0, nx * nz * sizeof(float));
        cudaMemset(rho, 0, nx * nz * sizeof(float));
        cudaMemset(epsilon, 0, nx * nz * sizeof(float));
        cudaMemset(delta, 0, nx * nz * sizeof(float));
        cudaMemset(gamma, 0, nx * nz * sizeof(float));
        cudaMemset(C11, 0, nx * nz * sizeof(float));
        cudaMemset(C13, 0, nx * nz * sizeof(float));
        cudaMemset(C33, 0, nx * nz * sizeof(float));
        cudaMemset(C44, 0, nx * nz * sizeof(float));
        cudaMemset(C66, 0, nx * nz * sizeof(float));
    }
}

Grid_Model_Thomsen::~Grid_Model_Thomsen() {
    if (mem_location == HOST_MEM) {
        if (vp0) {
            delete[] vp0;
            vp0 = nullptr;
        }
        if (vs0) {
            delete[] vs0;
            vs0 = nullptr;
        }
        if (rho) {
            delete[] rho;
            rho = nullptr;
        }
        if (epsilon) {
            delete[] epsilon;
            epsilon = nullptr;
        }
        if (delta) {
            delete[] delta;
            delta = nullptr;
        }
        if (gamma) {
            delete[] gamma;
            gamma = nullptr;
        }
        if (C11) {
            delete[] C11;
            C11 = nullptr;
        }
        if (C13) {
            delete[] C13;
            C13 = nullptr;
        }
        if (C33) {
            delete[] C33;
            C33 = nullptr;
        }
        if (C44) {
            delete[] C44;
            C44 = nullptr;
        }
        if (C66) {
            delete[] C66;
            C66 = nullptr;
        }
    } else {
        if (vp0) {
            cudaFree(vp0);
            vp0 = nullptr;
        }
        if (vs0) {
            cudaFree(vs0);
            vs0 = nullptr;
        }
        if (rho) {
            cudaFree(rho);
            rho = nullptr;
        }
        if (epsilon) {
            cudaFree(epsilon);
            epsilon = nullptr;
        }
        if (delta) {
            cudaFree(delta);
            delta = nullptr;
        }
        if (gamma) {
            cudaFree(gamma);
            gamma = nullptr;
        }
        if (C11) {
            cudaFree(C11);
            C11 = nullptr;
        }
        if (C13) {
            cudaFree(C13);
            C13 = nullptr;
        }
        if (C33) {
            cudaFree(C33);
            C33 = nullptr;
        }
        if (C44) {
            cudaFree(C44);
            C44 = nullptr;
        }
        if (C66) {
            cudaFree(C66);
            C66 = nullptr;
        }
    }
}

void Grid_Model_Thomsen::read(const std::array<const char *, 6> &files) {
    if (mem_location == DEVICE_MEM) {
        printf("RE in \"Grid_Model_Thomsen::read\"!\n");
        exit(1);
    }

    std::array<float *, 6> dst = {
        vp0, vs0, rho, epsilon, delta, gamma
    };
    FILE *fp = nullptr;
    for (int i = 0; i < 6; i++) {
        fp = fopen(files[i], "rb");
        for (int iz = 0; iz < nz; iz++) {
            fread(dst[i] + iz * nx, sizeof(float), nx, fp);
        }
        printf("Finished reading %s\n", files[i]);
        fclose(fp);
    }
}

void Grid_Model_Thomsen::memcpy_to_device_from(const Grid_Model_Thomsen &rhs) {
    if (rhs.mem_location == mem_location || mem_location == HOST_MEM) {
        printf("RE in \"Grid_Model_Thomsen::memcpy_to_device_from\"!\n");
        exit(1);
    }

    int total_bytes = nx * nz * sizeof(float);
    cudaMemcpy(vp0, rhs.vp0, total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(vs0, rhs.vs0, total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(rho, rhs.rho, total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(epsilon, rhs.epsilon, total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(delta, rhs.delta, total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gamma, rhs.gamma, total_bytes, cudaMemcpyHostToDevice);
}

void Grid_Model_Thomsen::calc_stiffness() {
    dim3 gridSize((nx + 15) / 16, (nz + 15) / 16);
    dim3 blockSize(16, 16);
    thomsen_to_stiffness<<<gridSize, blockSize>>>(view());
}

__global__ void thomsen_to_stiffness(Grid_Model_Thomsen::View gm) {
    if (gm.mem_location != DEVICE_MEM) {
        printf("RE in \"thomsen_to_stiffness\"!\n");
        return;
    }
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= gm.nx || iz >= gm.nz) {
        return;
    }

    int idx = iz * gm.nx + ix;
    float vp0 = gm.vp0[idx];
    float vs0 = gm.vs0[idx];
    float rho = gm.rho[idx];
    float eps = gm.epsilon[idx];
    float del = gm.delta[idx];
    float gam = gm.gamma[idx];

    float vs0_sq = vs0 * vs0;
    float vp0_sq = vp0 * vp0;

    gm.C11[idx] = rho * vp0_sq * (1 + 2 * eps);
    gm.C33[idx] = rho * vp0_sq;
    gm.C44[idx] = rho * vs0_sq;
    gm.C66[idx] = rho * vs0_sq * (1 + 2 * gam);

    float temp1 = gm.C33[idx] - gm.C44[idx];
    float temp2 = 2 * gm.C33[idx] * temp1 * del;
    gm.C13[idx] = sqrtf(temp1 * temp1 + temp2) - gm.C44[idx];
}

// end ========== Thomsen 参数 ========== end 

// begin ========== 直接读入的刚度参数 ========== begin

Grid_Model::Grid_Model(int nx, int nz)
    : nx(nx), nz(nz), mem_location(DEVICE_MEM) 
{
    cudaMalloc((void**)&vp0, nx * nz * sizeof(float));
    cudaMalloc((void**)&vs0, nx * nz * sizeof(float));
    cudaMalloc((void**)&rho, nx * nz * sizeof(float));
    cudaMalloc((void**)&C11, nx * nz * sizeof(float));
    cudaMalloc((void**)&C13, nx * nz * sizeof(float));
    cudaMalloc((void**)&C33, nx * nz * sizeof(float));
    cudaMalloc((void**)&C55, nx * nz * sizeof(float));

    cudaMemset(vp0, 0, nx * nz * sizeof(float));
    cudaMemset(vs0, 0, nx * nz * sizeof(float));
    cudaMemset(rho, 0, nx * nz * sizeof(float));
    cudaMemset(C11, 0, nx * nz * sizeof(float));
    cudaMemset(C13, 0, nx * nz * sizeof(float));
    cudaMemset(C33, 0, nx * nz * sizeof(float));
    cudaMemset(C55, 0, nx * nz * sizeof(float));
}

Grid_Model::~Grid_Model() {
    if (vp0) {
        cudaFree(vp0);
        vp0 = nullptr;
    }
    if (vs0) {
        cudaFree(vs0);
        vs0 = nullptr;
    }
    if (rho) {
        cudaFree(rho);
        rho = nullptr;
    }
    if (C11) {
        cudaFree(C11);
        C11 = nullptr;
    }
    if (C13) {
        cudaFree(C13);
        C13 = nullptr;
    }
    if (C33) {
        cudaFree(C33);
        C33 = nullptr;
    }
    if (C55) {
        cudaFree(C55);
        C55 = nullptr;
    }
}

void Grid_Model::read(const std::array<const char *, 7> &files) {
    std::array<float *, 7> dst = {
        vp0, vs0, rho, C11, C13, C33, C55  
    };

    FILE *fp = nullptr;
    float *temp = new float[nx * nz]();
    for (int i = 0; i < 7; i++) {
        fp = fopen(files[i], "rb");
        for (int iz = 0; iz < nz; iz++) {
            fread(temp + iz * nx, sizeof(float), nx, fp);
        }
        printf("Finished reading %s\n", files[i]);
        fclose(fp);
        cudaMemcpy(dst[i], temp, nx * nz * sizeof(float), cudaMemcpyHostToDevice);
    }
    delete[] temp;
    temp = nullptr;
}

// end ========== 直接读入的刚度参数 ========== end

Grid_Core::Grid_Core(int nx, int nz, bool mem_location) 
    : nx(nx), nz(nz), mem_location(mem_location) {
    if (mem_location == HOST_MEM) {
        vx[0] = new float[(nx - 1) * nz]();
        vx[1] = new float[(nx - 1) * nz]();

        vz[0] = new float[nx * (nz - 1)]();
        vz[1] = new float[nx * (nz - 1)]();

        sx[0] = new float[nx * nz]();
        sx[1] = new float[nx * nz]();

        sz[0] = new float[nx * nz]();
        sz[1] = new float[nx * nz]();

        txz[0] = new float[(nx - 1) * (nz - 1)]();
        txz[1] = new float[(nx - 1) * (nz - 1)]();
    }  else {
        cudaMalloc((void**)&vx[0], (nx - 1) * nz * sizeof(float));
        cudaMalloc((void**)&vx[1], (nx - 1) * nz * sizeof(float));
        cudaMalloc((void**)&vz[0], nx * (nz - 1) * sizeof(float));
        cudaMalloc((void**)&vz[1], nx * (nz - 1) * sizeof(float));
        cudaMalloc((void**)&sx[0], nx * nz * sizeof(float));
        cudaMalloc((void**)&sx[1], nx * nz * sizeof(float));
        cudaMalloc((void**)&sz[0], nx * nz * sizeof(float));
        cudaMalloc((void**)&sz[1], nx * nz * sizeof(float));
        cudaMalloc((void**)&txz[0], (nx - 1) * (nz - 1) * sizeof(float));
        cudaMalloc((void**)&txz[1], (nx - 1) * (nz - 1) * sizeof(float));

        cudaMemset(vx[0], 0, (nx - 1) * nz * sizeof(float));
        cudaMemset(vx[1], 0, (nx - 1) * nz * sizeof(float));
        cudaMemset(vz[0], 0, nx * (nz - 1) * sizeof(float));
        cudaMemset(vz[1], 0, nx * (nz - 1) * sizeof(float));
        cudaMemset(sx[0], 0, nx * nz * sizeof(float));
        cudaMemset(sx[1], 0, nx * nz * sizeof(float));
        cudaMemset(sz[0], 0, nx * nz * sizeof(float));
        cudaMemset(sz[1], 0, nx * nz * sizeof(float));
        cudaMemset(txz[0], 0, (nx - 1) * (nz - 1) * sizeof(float));
        cudaMemset(txz[1], 0, (nx - 1) * (nz - 1) * sizeof(float));
    }
}

Grid_Core::~Grid_Core() {
    if (mem_location == HOST_MEM) {
        if (vx[0]) {
            delete[] vx[0];
            vx[0] = nullptr;
        }
        if (vx[1]) {
            delete[] vx[1];
            vx[1] = nullptr;
        }
        if (vz[0]) {
            delete[] vz[0];
            vz[0] = nullptr;
        }
        if (vz[1]) {
            delete[] vz[1];
            vz[1] = nullptr;
        }
        if (sx[0]) {
            delete[] sx[0];
            sx[0] = nullptr;
        }
        if (sx[1]) {
            delete[] sx[1];
            sx[1] = nullptr;
        }
        if (sz[0]) {
            delete[] sz[0];
            sz[0] = nullptr;
        }
        if (sz[1]) {
            delete[] sz[1];
            sz[1] = nullptr;
        }
        if (txz[0]) {
            delete[] txz[0];
            txz[0] = nullptr;
        }
        if (txz[1]) {
            delete[] txz[1];
            txz[1] = nullptr;
        }
    } else {
        if (vx[0]) {
            cudaFree(vx[0]);
            vx[0] = nullptr;
        }
        if (vx[1]) {
            cudaFree(vx[1]);
            vx[1] = nullptr;
        }
        if (vz[0]) {
            cudaFree(vz[0]);
            vz[0] = nullptr;
        }
        if (vz[1]) {
            cudaFree(vz[1]);
            vz[1] = nullptr;
        }
        if (sx[0]) {
            cudaFree(sx[0]);
            sx[0] = nullptr;
        }
        if (sx[1]) {
            cudaFree(sx[1]);
            sx[1] = nullptr;
        }
        if (sz[0]) {
            cudaFree(sz[0]);
            sz[0] = nullptr;
        }
        if (sz[1]) {
            cudaFree(sz[1]);
            sz[1] = nullptr;
        }
        if (txz[0]) {
            cudaFree(txz[0]);
            txz[0] = nullptr;
        }
        if (txz[1]) {
            cudaFree(txz[1]);
            txz[1] = nullptr;
        }
    }
}

void Grid_Core::memcpy_to_host_from(const Grid_Core &rhs) {
    if (rhs.mem_location == mem_location || rhs.mem_location == HOST_MEM) {
        printf("RE in \"Grid_Core::memcpy_to_host_from\"!\n");
        exit(1);
    }
    cudaMemcpy(vx[0], rhs.vx[0], (nx - 1) * nz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vx[1], rhs.vx[1], (nx - 1) * nz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vz[0], rhs.vz[0], nx * (nz - 1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vz[1], rhs.vz[1], nx * (nz - 1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sx[0], rhs.sx[0], nx * nz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sx[1], rhs.sx[1], nx * nz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sz[0], rhs.sz[0], nx * nz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sz[1], rhs.sz[1], nx * nz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(txz[0], rhs.txz[0], (nx - 1) * (nz - 1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(txz[1], rhs.txz[1], (nx - 1) * (nz - 1) * sizeof(float), cudaMemcpyDeviceToHost);
}

std::unique_ptr<float[]> ricker_wavelet(int nt, float dt, float fpeak) {
    std::unique_ptr<float[]> wavelet = std::make_unique<float[]>(nt);
    float T = 1.3 / fpeak;
    for (int it = 0; it < nt; it++) {
        float t = it * dt - T;
        float temp = M_PI * fpeak * t;
        temp *= temp;
        wavelet[it] = (1 - 2 * temp) * exp(-temp);
    }
    return wavelet;
}