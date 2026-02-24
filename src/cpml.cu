#include "cpml.cuh"
#include "cJSON.h"
#include "params.cuh"
#include "json_func.cuh"
#include <iostream>
#include <cmath>

extern float dx_host_coarse;
extern float dz_host_coarse;
extern int nx_host_coarse;
extern int nz_host_coarse;

__constant__ int thickness_d;
__constant__ float a_int_d[CPMLMAX];
__constant__ float a_half_d[CPMLMAX];
__constant__ float b_int_d[CPMLMAX];
__constant__ float b_half_d[CPMLMAX];
__constant__ float kappa_int_d[CPMLMAX];
__constant__ float kappa_half_d[CPMLMAX];

Cpml::Cpml(const std::string &file) {
    load(file);
    build_constant();
}

Cpml::~Cpml() {
    mem_release();
}

void Cpml::load(const std::string &file) {
    const std::string json_content = readJsonFile(file);
    cJSON *root = cJSON_Parse(json_content.c_str());
    if (root == nullptr) {
        std::cout << "Failed to parse CPML JSON file " << file << '\n';
        exit(1);
    }

    cJSON *base = cJSON_GetObjectItem(root, "base");
    cJSON *pml = cJSON_GetObjectItem(root, "cpml");

    thickness = cJSON_GetObjectItem(pml, "thickness")->valueint;
    NPOW = cJSON_GetObjectItem(pml, "N")->valuedouble;
    cp_max = cJSON_GetObjectItem(pml, "cp_max")->valuedouble;
    Rc = cJSON_GetObjectItem(pml, "Rc")->valuedouble;
    kappa0 = cJSON_GetObjectItem(pml, "kappa0")->valuedouble;
    
    std::cout << "CPML parameters loaded.\n";

    L = thickness * dx_host_coarse;
    damp0 = -(NPOW + 1) * cp_max * log(Rc) / (2 * L);
    alpha0 = M_PI * cJSON_GetObjectItem(base, "fpeak")->valuedouble;

    mem_allocate(nx_host_coarse, nz_host_coarse);
    
    float dt = cJSON_GetObjectItem(base, "dt")->valuedouble;
    for (int i = 0; i < thickness; i++) {
        // 整网格点
        float x_int = 1.0 * i / thickness;
        damp_int[i] = damp0 * powf(x_int, NPOW);
        alpha_int[i] = alpha0 * (1.0f - x_int);
        kappa_int[i] = 1.0f + (kappa0 - 1.0f) * powf(x_int, NPOW);

        b_int[i] = exp(
            -(damp_int[i] / kappa_int[i] + alpha_int[i]) * dt
        );
        a_int[i] = damp_int[i] * (b_int[i] - 1) / (
            kappa_int[i] * (damp_int[i] + kappa_int[i] * alpha_int[i])
        );
        
        // 半网格点
        if (i == thickness - 1) break;
        float x_half = (i + 0.5f) / (thickness);
        damp_half[i] = damp0 * powf(x_half, NPOW);
        alpha_half[i] = alpha0 * (1.0f - x_half);
        kappa_half[i] = 1.0f + (kappa0 - 1.0f) * powf(x_half, NPOW);

        b_half[i] = exp(
            -(damp_half[i] / kappa_half[i] + alpha_half[i]) * dt
        );
        a_half[i] = damp_half[i] * (b_half[i] - 1) / (
            kappa_half[i] * (damp_half[i] + kappa_half[i] * alpha_half[i])
        );
    }
    cJSON_Delete(root);
}

void Cpml::mem_allocate(int lx, int lz) {
    alpha_int = new float[thickness]();
    alpha_half = new float[thickness - 1]();
    damp_int = new float[thickness]();
    damp_half = new float[thickness - 1]();
    a_int = new float[thickness]();
    a_half = new float[thickness - 1]();
    b_int = new float[thickness]();
    b_half = new float[thickness - 1]();
    kappa_int = new float[thickness]();
    kappa_half = new float[thickness - 1]();

    cudaMalloc(&psi_vel.psi_vx_x, (lx - 1) * lz * sizeof(float));
    cudaMalloc(&psi_vel.psi_vx_z, (lx - 1) * lz * sizeof(float));
    cudaMalloc(&psi_vel.psi_vz_x, lx * (lz - 1) * sizeof(float));
    cudaMalloc(&psi_vel.psi_vz_z, lx * (lz - 1) * sizeof(float));
    cudaMalloc(&psi_str.psi_sx_x, lx * lz * sizeof(float));
    cudaMalloc(&psi_str.psi_sx_z, lx * lz * sizeof(float));
    cudaMalloc(&psi_str.psi_sz_x, lx * lz * sizeof(float));
    cudaMalloc(&psi_str.psi_sz_z, lx * lz * sizeof(float));
    cudaMalloc(&psi_str.psi_txz_x, (lx - 1) * (lz - 1) * sizeof(float));
    cudaMalloc(&psi_str.psi_txz_z, (lx - 1) * (lz - 1) * sizeof(float));

    cudaMemset(psi_vel.psi_vx_x, 0, (lx - 1) * lz * sizeof(float));
    cudaMemset(psi_vel.psi_vx_z, 0, (lx - 1) * lz * sizeof(float));
    cudaMemset(psi_vel.psi_vz_x, 0, lx * (lz - 1) * sizeof(float));
    cudaMemset(psi_vel.psi_vz_z, 0, lx * (lz - 1) * sizeof(float));
    cudaMemset(psi_str.psi_sx_x, 0, lx * lz * sizeof(float));
    cudaMemset(psi_str.psi_sx_z, 0, lx * lz * sizeof(float));
    cudaMemset(psi_str.psi_sz_x, 0, lx * lz * sizeof(float));
    cudaMemset(psi_str.psi_sz_z, 0, lx * lz * sizeof(float));
    cudaMemset(psi_str.psi_txz_x, 0, (lx - 1) * (lz - 1) * sizeof(float));
    cudaMemset(psi_str.psi_txz_z, 0, (lx - 1) * (lz - 1) * sizeof(float));
}

void Cpml::build_constant() {
    cudaMemcpyToSymbol(thickness_d, &thickness, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(a_int_d, a_int, thickness * sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(a_half_d, a_half, (thickness - 1) * sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(b_int_d, b_int, thickness * sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(b_half_d, b_half, (thickness - 1) * sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kappa_int_d, kappa_int, thickness * sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kappa_half_d, kappa_half, (thickness - 1) * sizeof(float), 0, cudaMemcpyHostToDevice);
}

void Cpml::mem_release() {
    delete[] alpha_int;
    delete[] alpha_half;
    delete[] damp_int;
    delete[] damp_half;
    delete[] a_int;
    delete[] a_half;
    delete[] b_int;
    delete[] b_half;
    delete[] kappa_int;
    delete[] kappa_half;
    cudaFree(psi_vel.psi_vx_x); 
    cudaFree(psi_vel.psi_vx_z);
    cudaFree(psi_vel.psi_vz_x);
    cudaFree(psi_vel.psi_vz_z);
    cudaFree(psi_str.psi_sx_x);
    cudaFree(psi_str.psi_sx_z);
    cudaFree(psi_str.psi_sz_x);
    cudaFree(psi_str.psi_sz_z);
    cudaFree(psi_str.psi_txz_x);
    cudaFree(psi_str.psi_txz_z);
}

// __device__ int get_cpml_idx_x_half(int lx, int ix, int thickness) {
//     int res = -1;
//     int arr[] = {ix - 4, lx - 1 - ix - 3};
//     for (int i = 0; i < 2; i++) {
//         if (0 <= arr[i] && arr[i] < thickness) {
//             res = arr[i];
//             break;
//         }
//     }
//     return thickness - 1 - res;
// }
// __device__ int get_cpml_idx_z_half(int lz, int iz, int thickness) {
//     int res = -1;
//     int arr[] = {iz - 4, lz - 1 - iz - 3};
//     for (int i = 0; i < 2; i++) {
//         if (0 <= arr[i] && arr[i] < thickness) {
//             res = arr[i];
//             break;
//         }
//     }
//     return thickness - 1 - res;    
// }

// __device__ int get_cpml_idx_x_int(int lx, int ix, int thickness) {
//     int res = -1;
//     if ()
//     int arr[] = {ix - 3, lx - 1 - ix - 4};
//     for (int i = 0; i < 2; i++) {
//         if (0 <= arr[i] && arr[i] < thickness) {
//             res = arr[i];
//             break;
//         }
//     }
//     return thickness - 1 - res;
// }

// __device__ int get_cpml_idx_z_int(int lz, int iz, int thickness) {
//     int res = -1;
//     int arr[] = {iz - 3, lz - 1 - iz - 4};
//     for (int i = 0; i < 2; i++) {
//         if (0 <= arr[i] && arr[i] < thickness) {
//             res = arr[i];
//             break;
//         }
//     }
//     return thickness - 1 - res;
// }

