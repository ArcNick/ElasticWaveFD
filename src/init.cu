#include "common.cuh"
#include "cJSON.h"
#include <memory>
#include <cmath>
#include <cstdio>
#include <array>
#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <cuda_runtime.h>

void Params::read(const std::string &file) {
    const std::string json_content = readJsonFile(file);
    cJSON *root = cJSON_Parse(json_content.c_str());
    if (root == nullptr) {
        std::cout << "无法解析: " << std::string(cJSON_GetErrorPtr()) << '\n';
    }

    fpeak = cJSON_GetObjectItem(root, "fpeak")->valuedouble;
    nx = cJSON_GetObjectItem(root, "nx")->valueint;
    nz = cJSON_GetObjectItem(root, "nz")->valueint;
    dx = cJSON_GetObjectItem(root, "dx")->valuedouble;
    dz = cJSON_GetObjectItem(root, "dz")->valuedouble;
    nt = cJSON_GetObjectItem(root, "nt")->valueint;
    dt = cJSON_GetObjectItem(root, "dt")->valuedouble;
    posx = cJSON_GetObjectItem(root, "posx")->valueint;
    posz = cJSON_GetObjectItem(root, "posz")->valueint;
    snapshot = cJSON_GetObjectItem(root, "snapshot")->valueint;

    cJSON_Delete(root);
    std::cout << "Parameters loaded.\n";
}

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

void Grid_Model::read(const std::string &file) {
    // 解析 JSON 文件
    const std::string json_content = readJsonFile(file);
    cJSON *root = cJSON_Parse(json_content.c_str());
    if (root == nullptr) {
        std::cout << "无法解析: " << std::string(cJSON_GetErrorPtr()) << '\n';
    }

    // 做表方便读取
    struct Pair {float *ptr; std::string name;} dst[7] = {
        {vp0, "vp"}, {vs0, "vs"}, {rho, "rho"}, 
        {C11, "C11"}, {C13, "C13"}, {C33, "C33"}, {C55, "C55"}
    };
    std::string files[7];

    // 读取文件名
    for (int i = 0; i < 7; i++) {
        cJSON *item = cJSON_GetObjectItem(root, "coarse_model");
        item = cJSON_GetObjectItem(item, dst[i].name.c_str());
        files[i] = item->valuestring;
        
    }
    cJSON_Delete(root);

    // 逐个读入对应文件的数据
    std::unique_ptr<float[]> temp = std::make_unique<float[]>(nx * nz);
    for (int i = 0; i < 7; i++) {
        std::ifstream fp(files[i], std::ios::binary | std::ios::in); // equal : FILE *fp = fopen(files[i], "rb");
        for (int iz = 0; iz < nz; iz++) {
            fp.read(reinterpret_cast<char*>(temp.get() + iz * nx), nx * sizeof(float));
        }
        std::cout << "Finished reading " << files[i] << "\n";
        fp.close();
        cudaMemcpy(dst[i].ptr, temp.get(), nx * nz * sizeof(float), cudaMemcpyHostToDevice);
    }
}

// end ========== 直接读入的刚度参数 ========== end

Grid_Core::Grid_Core(int nx, int nz, bool mem_location) 
    : nx(nx), nz(nz), mem_location(mem_location) {
    if (mem_location == HOST_MEM) {
        vx = new float[(nx - 1) * nz * 2]();
        vz = new float[nx * (nz - 1) * 2]();
        sx = new float[nx * nz * 2]();
        sz = new float[nx * nz * 2]();
        txz = new float[(nx - 1) * (nz - 1) * 2]();
    }  else {
        cudaMalloc((void**)&vx, 2 * (nx - 1) * nz * sizeof(float));
        cudaMalloc((void**)&vz, 2 * nx * (nz - 1) * sizeof(float));
        cudaMalloc((void**)&sx, 2 * nx * nz * sizeof(float));
        cudaMalloc((void**)&sz, 2 * nx * nz * sizeof(float));
        cudaMalloc((void**)&txz, 2 * (nx - 1) * (nz - 1) * sizeof(float));
    }
}

Grid_Core::~Grid_Core() {
    if (mem_location == HOST_MEM) {
        if (vx) {
            delete[] vx;
            vx = nullptr;
        }
        if (vz) {
            delete[] vz;
            vz = nullptr;
        }
        if (sx) {
            delete[] sx;
            sx = nullptr;
        }
        if (sz) {
            delete[] sz;
            sz = nullptr;
        }
        if (txz) {
            delete[] txz;
            txz = nullptr;
        }
    } else {
        if (vx) {
            cudaFree(vx);
            vx = nullptr;
        }
        if (vz) {
            cudaFree(vz);
            vz = nullptr;
        }
        if (sx) {
            cudaFree(sx);
            sx = nullptr;
        }
        if (sz) {
            cudaFree(sz);
            sz = nullptr;
        }
        if (txz) {
            cudaFree(txz);
            txz = nullptr;
        }
    }
}

void Grid_Core::memcpy_to_host_from(const Grid_Core &rhs) {
    if (rhs.mem_location == mem_location || rhs.mem_location == HOST_MEM) {
        printf("RE in \"Grid_Core::memcpy_to_host_from\"!\n");
        exit(1);
    }
    cudaMemcpy(vx, rhs.vx, 2 * (nx - 1) * nz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vz, rhs.vz, 2 * nx * (nz - 1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sx, rhs.sx, 2 * nx * nz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sz, rhs.sz, 2 * nx * nz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(txz, rhs.txz, 2 * (nx - 1) * (nz - 1) * sizeof(float), cudaMemcpyDeviceToHost);
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