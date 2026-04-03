#include "params.cuh"
#include "cJSON.h"
#include "json_func.cuh"
#include <cuda_runtime.h>
#include <iostream>

__constant__ float freq;
__constant__ float dt_d;
__constant__ int nt_d;
__constant__ int posx_d;
__constant__ int posz_d;

Params::Params(const std::string &file) {
    read(file);
    build_constant();
}

void Params::read(const std::string &file) {
    const std::string json_content = readJsonFile(file);
    cJSON *root = cJSON_Parse(json_content.c_str());
    if (root == nullptr) {
        std::cout << "无法解析: " << std::string(cJSON_GetErrorPtr()) << '\n';
    }

    cJSON *base = cJSON_GetObjectItem(root, "base");

    fpeak = cJSON_GetObjectItem(base, "fpeak")->valuedouble;
    dt = cJSON_GetObjectItem(base, "dt")->valuedouble;
    nt = cJSON_GetObjectItem(base, "nt")->valueint;
    posx = cJSON_GetObjectItem(base, "posx")->valueint;
    posz = cJSON_GetObjectItem(base, "posz")->valueint;
    snapshot = cJSON_GetObjectItem(base, "snapshot")->valueint;

    cJSON_Delete(root);
    std::cout << "Parameters loaded.\n";
}

void Params::build_constant() {
    cudaMemcpyToSymbol(freq, &fpeak, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dt_d, &dt, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nt_d, &nt, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(posx_d, &posx, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(posz_d, &posz, sizeof(int), 0, cudaMemcpyHostToDevice);
}

std::unique_ptr<float[]> Params::ricker_wavelet() {
    std::unique_ptr<float[]> wavelet = std::make_unique<float[]>(nt);
    float T = 1.2 / fpeak;
    double PI = 3.14159265358979323846;
    for (int it = 0; it < nt; it++) {
        float t = it * dt - T;
        float temp = PI * fpeak * t;
        temp *= temp;
        wavelet[it] = (1 - 2 * temp) * exp(-temp) * GAIN;
    }
    return wavelet;
}

