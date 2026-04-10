#include "grid_manager.cuh"
#include "cJSON.h"
#include "json_func.cuh"
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>

// 设备端全局符号定义
__device__ cudaTextureObject_t vx_mask;
__device__ cudaTextureObject_t vz_mask;
__device__ cudaTextureObject_t sig_mask;
__device__ cudaTextureObject_t txz_mask;
__device__ cudaTextureObject_t mat_tex;

__device__ cudaTextureObject_t vx_n_tex;
__device__ cudaTextureObject_t vz_n_tex;
__device__ cudaTextureObject_t sig_n_tex;
__device__ cudaTextureObject_t txz_n_tex;

__constant__ float dx, dz;
__constant__ int nx, nz;
__constant__ int offset_vx_all;
__constant__ int offset_vz_all;
__constant__ int offset_sig_all;
__constant__ int offset_txz_all;

__constant__ int num_fine;
__constant__ FineInfo fines[8];
__constant__ int sum_offset_fine_vx[8];
__constant__ int sum_offset_fine_vz[8];
__constant__ int sum_offset_fine_sig[8];
__constant__ int sum_offset_fine_txz[8];

__constant__ float lagrange_coeff[LUT_SIZE * LAGRANGE_ORDER];

float dx_host_coarse, dz_host_coarse;
int nx_host_coarse, nz_host_coarse;

GridManager::GridManager(const std::string &file) {
    load_from_file(file);
}

void GridManager::load_from_file(const std::string &file) {
    const std::string json_content = readJsonFile(file);
    cJSON *root = cJSON_Parse(json_content.c_str());
    if (root == nullptr) {
        std::cout << "无法解析: " << std::string(cJSON_GetErrorPtr()) << '\n';
        exit(1);
    }

    // 粗网格
    cJSON *coarse_ptr = cJSON_GetObjectItem(root, "coarse");
    nx_coarse = cJSON_GetObjectItem(coarse_ptr, "nx")->valueint;
    nz_coarse = cJSON_GetObjectItem(coarse_ptr, "nz")->valueint;
    dx_coarse = cJSON_GetObjectItem(coarse_ptr, "dx")->valuedouble;
    dz_coarse = cJSON_GetObjectItem(coarse_ptr, "dz")->valuedouble;

    dx_host_coarse = dx_coarse;
    dz_host_coarse = dz_coarse;
    nx_host_coarse = nx_coarse;
    nz_host_coarse = nz_coarse;

    offset_time_vx = (nx_coarse - 1) * nz_coarse;
    offset_time_vz = nx_coarse * (nz_coarse - 1);
    offset_time_sig = nx_coarse * nz_coarse;
    offset_time_txz = (nx_coarse - 1) * (nz_coarse - 1);

    offset_coarse_vx = (nx_coarse - 1) * nz_coarse;
    offset_coarse_vz = nx_coarse * (nz_coarse - 1);
    offset_coarse_sig = nx_coarse * nz_coarse;
    offset_coarse_txz = (nx_coarse - 1) * (nz_coarse - 1);

    // 细网格
    cJSON *fine_ptr = cJSON_GetObjectItem(root, "fine");
    if (fine_ptr == nullptr || cJSON_GetArraySize(fine_ptr) == 0) {
        FINE = FINE_OFF;
    } else {
        FINE = FINE_ON;
    }
    int num = 0;
    if (FINE == FINE_ON) num = cJSON_GetArraySize(fine_ptr);

    for (int i = 0; i < num; i++) {
        FineInfo info;
        cJSON *item = cJSON_GetArrayItem(fine_ptr, i);
        info.x_start = cJSON_GetObjectItem(item, "x_start")->valueint;
        info.x_end = cJSON_GetObjectItem(item, "x_end")->valueint;
        info.z_start = cJSON_GetObjectItem(item, "z_start")->valueint;
        info.z_end = cJSON_GetObjectItem(item, "z_end")->valueint;
        info.N = cJSON_GetObjectItem(item, "N")->valueint;
        info.dx_fine = dx_coarse / info.N;
        info.dz_fine = dz_coarse / info.N;
        info.lenx = (info.x_end - info.x_start) * info.N + 1;
        info.lenz = (info.z_end - info.z_start) * info.N + 1;
        fine_info.push_back(info);
    }
    for (int i = 0; i < fine_info.size(); i++) {
        offset_time_vx += (fine_info[i].lenx - 1) * fine_info[i].lenz;
        offset_time_vz += fine_info[i].lenx * (fine_info[i].lenz - 1);
        offset_time_sig += fine_info[i].lenx * fine_info[i].lenz;
        offset_time_txz += (fine_info[i].lenx - 1) * (fine_info[i].lenz - 1);
    }

    memory_allocate();

    // 读取模型参数文件
    struct Pair { void *ptr; std::string name; } dst[9] = {
        { model_h.rho, "rho" }, { model_h.C11, "C11" }, { model_h.C13, "C13" },
        { model_h.C33, "C33" }, { model_h.C55, "C55" }, { model_h.inv_tsig, "inv_tsig"},
        { model_h.taup, "taup"}, {model_h.taus, "taus"}, { model_h.mat, "material"}
    };
    std::string filename;

    cJSON *item = nullptr;
    // coarse
    for (int i = 0; i < 9; i++) {
        item = coarse_ptr;
        item = cJSON_GetObjectItem(item, dst[i].name.c_str());
        filename = item->valuestring;
        FILE *fp = fopen(filename.c_str(), "rb");
        if (!fp) {
            std::cout << "无法打开文件: " << filename << '\n';
            exit(1);
        }
        if (dst[i].name == "material") {
            fread(static_cast<MAT_FLAG*>(dst[i].ptr), sizeof(MAT_FLAG), nx_coarse * nz_coarse, fp);
        } else {
            fread(static_cast<float*>(dst[i].ptr), sizeof(float), nx_coarse * nz_coarse, fp);
        }
        fclose(fp);
    }
    
    // fine
    for (int i = 0, offset = nx_coarse * nz_coarse; i < num; i++, offset += fine_info[i].lenx * fine_info[i].lenz) {
        for (int j = 0; j < 9; j++) {
            item = fine_ptr;
            item = cJSON_GetArrayItem(item, i);
            item = cJSON_GetObjectItem(item, dst[j].name.c_str());
            filename = item->valuestring;
            FILE *fp = fopen(filename.c_str(), "rb");
            if (!fp) {
                std::cout << "无法打开文件: " << filename << '\n';
                exit(1);
            }
            if (dst[j].name == "material") {
                fread(static_cast<MAT_FLAG*>(dst[j].ptr) + offset, sizeof(MAT_FLAG), fine_info[i].lenx * fine_info[i].lenz, fp);
            } else {
                fread(static_cast<float*>(dst[j].ptr) + offset, sizeof(float), fine_info[i].lenx * fine_info[i].lenz, fp);
            }
            fclose(fp);
        }
    }

    cJSON_Delete(root);

    build_texture();
    build_n();
    build_constant();
    build_insterp_LUT();
}

GridManager::~GridManager() {
    memory_release();
}

void GridManager::memcpy_core_d2h(int time) {
    int bytes_vx = offset_time_vx * sizeof(float);
    int bytes_vz = offset_time_vz * sizeof(float);
    int bytes_sig = offset_time_sig * sizeof(float);
    int bytes_txz = offset_time_txz * sizeof(float);
    cudaMemcpy(core_h.vx, core_d.vx + time * bytes_vx, bytes_vx, cudaMemcpyDeviceToHost);
    cudaMemcpy(core_h.vz, core_d.vz + time * bytes_vz, bytes_vz, cudaMemcpyDeviceToHost);
    cudaMemcpy(core_h.sx, core_d.sx + time * bytes_sig, bytes_sig, cudaMemcpyDeviceToHost);
    cudaMemcpy(core_h.sz, core_d.sz + time * bytes_sig, bytes_sig, cudaMemcpyDeviceToHost);
    cudaMemcpy(core_h.txz, core_d.txz + time * bytes_txz, bytes_txz, cudaMemcpyDeviceToHost);
}

void GridManager::memcpy_model_h2d() {
    int bytes_sig = offset_time_sig * sizeof(float);
    int bytes_mat = offset_time_sig * sizeof(MAT_FLAG);
    cudaMemcpy(model_d.rho, model_h.rho, bytes_sig, cudaMemcpyHostToDevice);
    cudaMemcpy(model_d.C11, model_h.C11, bytes_sig, cudaMemcpyHostToDevice);
    cudaMemcpy(model_d.C13, model_h.C13, bytes_sig, cudaMemcpyHostToDevice);
    cudaMemcpy(model_d.C33, model_h.C33, bytes_sig, cudaMemcpyHostToDevice);
    cudaMemcpy(model_d.C55, model_h.C55, bytes_sig, cudaMemcpyHostToDevice);
    cudaMemcpy(model_d.taup, model_h.taup, bytes_sig, cudaMemcpyHostToDevice);
    cudaMemcpy(model_d.taus, model_h.taus, bytes_sig, cudaMemcpyHostToDevice);
    cudaMemcpy(model_d.inv_tsig, model_h.inv_tsig, bytes_sig, cudaMemcpyHostToDevice);
    cudaMemcpy(model_d.mat, model_h.mat, bytes_mat, cudaMemcpyHostToDevice);
}

void GridManager::build_texture() {
    int *vx_mask_h = new int[(nx_coarse - 1) * nz_coarse]();
    int *vz_mask_h = new int[nx_coarse * (nz_coarse - 1)]();
    int *sig_mask_h = new int[nx_coarse * nz_coarse]();
    int *txz_mask_h = new int[(nx_coarse - 1) * (nz_coarse - 1)]();

    memset(vx_mask_h, -1, (nx_coarse - 1) * nz_coarse * sizeof(int));
    memset(vz_mask_h, -1, nx_coarse * (nz_coarse - 1) * sizeof(int));
    memset(sig_mask_h, -1, nx_coarse * nz_coarse * sizeof(int));
    memset(txz_mask_h, -1, (nx_coarse - 1) * (nz_coarse - 1) * sizeof(int));

    for (int iz = 0; iz < nz_coarse; iz++) {
        for (int ix = 0; ix < nx_coarse; ix++) {
            for (int i = 0; i < fine_info.size(); i++) {
                int z_start = fine_info[i].z_start;
                int z_end = fine_info[i].z_end;
                int x_start = fine_info[i].x_start;
                int x_end = fine_info[i].x_end;
                // sx 和 sz
                if (x_start <= ix && ix <= x_end && z_start <= iz && iz <= z_end) {
                    sig_mask_h[iz * nx_coarse + ix] = i;
                }
                // vx
                if (x_start <= ix && ix < x_end && z_start <= iz && iz <= z_end) {
                    vx_mask_h[iz * (nx_coarse - 1) + ix] = i;
                }
                // vz
                if (x_start <= ix && ix <= x_end && z_start <= iz && iz < z_end) {
                    vz_mask_h[iz * nx_coarse + ix] = i;
                }
                // txz
                if (x_start <= ix && ix < x_end && z_start <= iz && iz < z_end) {
                    txz_mask_h[iz * (nx_coarse - 1) + ix] = i;
                }
            }
        }
    }

    int bytes_vx_mask = (nx_coarse - 1) * nz_coarse * sizeof(int);
    int bytes_vz_mask = nx_coarse * (nz_coarse - 1) * sizeof(int);
    int bytes_sig_mask = nx_coarse * nz_coarse * sizeof(int);
    int bytes_txz_mask = (nx_coarse - 1) * (nz_coarse - 1) * sizeof(int);
    int bytes_mat_mask = nx_coarse * nz_coarse * sizeof(MAT_FLAG);
    cudaMemcpy(core_mask.vx, vx_mask_h, bytes_vx_mask, cudaMemcpyHostToDevice);
    cudaMemcpy(core_mask.vz, vz_mask_h, bytes_vz_mask, cudaMemcpyHostToDevice);
    cudaMemcpy(core_mask.sig, sig_mask_h, bytes_sig_mask, cudaMemcpyHostToDevice);
    cudaMemcpy(core_mask.txz, txz_mask_h, bytes_txz_mask, cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.desc = cudaCreateChannelDesc<int>();

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.normalizedCoords = 0;

    // sig_mask
    resDesc.res.linear.devPtr = core_mask.sig;
    resDesc.res.linear.sizeInBytes = bytes_sig_mask;
    cudaCreateTextureObject(&tex_sig_mask, &resDesc, &texDesc, NULL);
    cudaMemcpyToSymbol(sig_mask, &tex_sig_mask, sizeof(cudaTextureObject_t));

    // vx_mask
    resDesc.res.linear.devPtr = core_mask.vx;
    resDesc.res.linear.sizeInBytes = bytes_vx_mask;
    cudaCreateTextureObject(&tex_vx_mask, &resDesc, &texDesc, NULL);
    cudaMemcpyToSymbol(vx_mask, &tex_vx_mask, sizeof(cudaTextureObject_t));

    // vz_mask
    resDesc.res.linear.devPtr = core_mask.vz;
    resDesc.res.linear.sizeInBytes = bytes_vz_mask;
    cudaCreateTextureObject(&tex_vz_mask, &resDesc, &texDesc, NULL);
    cudaMemcpyToSymbol(vz_mask, &tex_vz_mask, sizeof(cudaTextureObject_t));

    // txz_mask
    resDesc.res.linear.devPtr = core_mask.txz;
    resDesc.res.linear.sizeInBytes = bytes_txz_mask;
    cudaCreateTextureObject(&tex_txz_mask, &resDesc, &texDesc, NULL);
    cudaMemcpyToSymbol(txz_mask, &tex_txz_mask, sizeof(cudaTextureObject_t));

    // material
    resDesc.res.linear.devPtr = model_d.mat;
    resDesc.res.linear.sizeInBytes = bytes_mat_mask;
    cudaCreateTextureObject(&tex_mat, &resDesc, &texDesc, NULL);
    cudaMemcpyToSymbol(mat_tex, &tex_mat, sizeof(cudaTextureObject_t));

    delete[] vx_mask_h;
    delete[] vz_mask_h;
    delete[] sig_mask_h;
    delete[] txz_mask_h;
}

void GridManager::build_n() {
    if (FINE == FINE_OFF) return;

    int offset_n = 0;
    for (int i = 0; i < fine_info.size(); i++) {
        offset_n += fine_info[i].lenx * fine_info[i].lenz;
    }

    FD_n *vx_n_h = new FD_n[offset_n]();
    FD_n *vz_n_h = new FD_n[offset_n]();
    FD_n *sig_n_h = new FD_n[offset_n]();
    FD_n *txz_n_h = new FD_n[offset_n]();
    
    offset_n = 0;

    for (int i = 0; i < fine_info.size(); i++) {
        int lz = fine_info[i].lenz;
        int lx = fine_info[i].lenx;
        for (int iz = 0; iz < lz; iz++) {
            for (int ix = 0; ix < lx; ix++) {
                int n_x_int = 0, n_x_half = 0, n_z_int = 0, n_z_half = 0;

                if (0 <= iz && iz < lz) {
                    n_z_int = std::min({ iz + 1, lz - iz - 1, 4 });
                }
                if (0 <= ix && ix < lx) {
                    n_x_int = std::min({ ix + 1, lx - ix - 1, 4 });
                }
                if (0 <= iz && iz < lz - 1) {
                    n_z_half = std::min({ iz, lz - 1 - iz, 4 });
                }
                if (0 <= ix && ix < lx - 1) {
                    n_x_half = std::min({ ix, lx - 1 - ix, 4 });
                }
                // vx
                if (ix < lx - 1) {
                    vx_n_h[offset_n + iz * lx + ix].n_x = n_x_half;
                    vx_n_h[offset_n + iz * lx + ix].n_z = n_z_int;
                }
                // vz
                if (iz < lz - 1) {
                    vz_n_h[offset_n + iz * lx + ix].n_x = n_x_int;
                    vz_n_h[offset_n + iz * lx + ix].n_z = n_z_half;
                }
                // sx & sz
                sig_n_h[offset_n + iz * lx + ix].n_x = n_x_int;
                sig_n_h[offset_n + iz * lx + ix].n_z = n_z_int;
                // txz
                if (ix < lx - 1 && iz < lz - 1) {
                    txz_n_h[offset_n + iz * lx + ix].n_x = n_x_half;
                    txz_n_h[offset_n + iz * lx + ix].n_z = n_z_half;
                }
            }
        }
        offset_n += lx * lz;
    }

    cudaMemcpy(vx_n, vx_n_h, offset_n * sizeof(FD_n), cudaMemcpyHostToDevice);
    cudaMemcpy(vz_n, vz_n_h, offset_n * sizeof(FD_n), cudaMemcpyHostToDevice);
    cudaMemcpy(sig_n, sig_n_h, offset_n * sizeof(FD_n), cudaMemcpyHostToDevice);
    cudaMemcpy(txz_n, txz_n_h, offset_n * sizeof(FD_n), cudaMemcpyHostToDevice);

    // 创建 n 数组纹理对象并保存到成员变量
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;

    cudaChannelFormatDesc fd_n_desc = cudaCreateChannelDesc(
        32, 32, 0, 0, cudaChannelFormatKindSigned);
    resDesc.res.linear.desc = fd_n_desc;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.normalizedCoords = 0;

    // vx_n
    resDesc.res.linear.devPtr = vx_n;
    resDesc.res.linear.sizeInBytes = offset_n * sizeof(FD_n);
    cudaCreateTextureObject(&tex_vx_n, &resDesc, &texDesc, NULL);
    cudaMemcpyToSymbol(vx_n_tex, &tex_vx_n, sizeof(cudaTextureObject_t));

    // vz_n
    resDesc.res.linear.devPtr = vz_n;
    resDesc.res.linear.sizeInBytes = offset_n * sizeof(FD_n);
    cudaCreateTextureObject(&tex_vz_n, &resDesc, &texDesc, NULL);
    cudaMemcpyToSymbol(vz_n_tex, &tex_vz_n, sizeof(cudaTextureObject_t));

    // sig_n
    resDesc.res.linear.devPtr = sig_n;
    resDesc.res.linear.sizeInBytes = offset_n * sizeof(FD_n);
    cudaCreateTextureObject(&tex_sig_n, &resDesc, &texDesc, NULL);
    cudaMemcpyToSymbol(sig_n_tex, &tex_sig_n, sizeof(cudaTextureObject_t));

    // txz_n
    resDesc.res.linear.devPtr = txz_n;
    resDesc.res.linear.sizeInBytes = offset_n * sizeof(FD_n);
    cudaCreateTextureObject(&tex_txz_n, &resDesc, &texDesc, NULL);
    cudaMemcpyToSymbol(txz_n_tex, &tex_txz_n, sizeof(cudaTextureObject_t));

    delete[] vx_n_h;
    delete[] vz_n_h;
    delete[] sig_n_h;
    delete[] txz_n_h;
}

void GridManager::build_constant() {
    cudaMemcpyToSymbol(nx, &nx_coarse, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nz, &nz_coarse, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dx, &dx_coarse, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dz, &dz_coarse, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(offset_vx_all, &offset_time_vx, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(offset_vz_all, &offset_time_vz, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(offset_sig_all, &offset_time_sig, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(offset_txz_all, &offset_time_txz, sizeof(int), 0, cudaMemcpyHostToDevice);

    if (FINE == FINE_OFF) return;
    int f_size = fine_info.size();
    cudaMemcpyToSymbol(num_fine, &f_size, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(fines, fine_info.data(), f_size * sizeof(FineInfo), 0, cudaMemcpyHostToDevice);

    std::vector<int> off_vx(f_size, 0);
    std::vector<int> off_vz(f_size, 0);
    std::vector<int> off_sig(f_size, 0);
    std::vector<int> off_txz(f_size, 0);

    off_vx[0] = (nx_coarse - 1) * nz_coarse;
    off_vz[0] = nx_coarse * (nz_coarse - 1);
    off_sig[0] = nx_coarse * nz_coarse;
    off_txz[0] = (nx_coarse - 1) * (nz_coarse - 1);

    for (int i = 1; i < f_size; i++) {
        off_vx[i] = (fine_info[i - 1].lenx - 1) * fine_info[i - 1].lenz + off_vx[i - 1];
        off_vz[i] = fine_info[i - 1].lenx * (fine_info[i - 1].lenz - 1) + off_vz[i - 1];
        off_sig[i] = fine_info[i - 1].lenx * fine_info[i - 1].lenz + off_sig[i - 1];
        off_txz[i] = (fine_info[i - 1].lenx - 1) * (fine_info[i - 1].lenz - 1) + off_txz[i - 1];
    }

    cudaMemcpyToSymbol(sum_offset_fine_vx, off_vx.data(), f_size * sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(sum_offset_fine_vz, off_vz.data(), f_size * sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(sum_offset_fine_sig, off_sig.data(), f_size * sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(sum_offset_fine_txz, off_txz.data(), f_size * sizeof(int), 0, cudaMemcpyHostToDevice);
}

void GridManager::memory_allocate() {
    // host core
    core_h.vx = new float[offset_time_vx]();
    core_h.vz = new float[offset_time_vz]();
    core_h.sx = new float[offset_time_sig]();
    core_h.sz = new float[offset_time_sig]();
    core_h.txz = new float[offset_time_txz]();

    // device core (2 time layers)
    cudaMalloc((void**)&core_d.vx, offset_time_vx * sizeof(float) * 2);
    cudaMalloc((void**)&core_d.vz, offset_time_vz * sizeof(float) * 2);

    cudaMalloc((void**)&core_d.sx, offset_time_sig * sizeof(float) * 2);
    cudaMalloc((void**)&core_d.sz, offset_time_sig * sizeof(float) * 2);
    cudaMalloc((void**)&core_d.txz, offset_time_txz * sizeof(float) * 2);
    cudaMalloc((void**)&core_d.rx, offset_time_sig * sizeof(float) * 2);
    cudaMalloc((void**)&core_d.rz, offset_time_sig * sizeof(float) * 2);
    cudaMalloc((void**)&core_d.rxz, offset_time_txz * sizeof(float) * 2);

    cudaMalloc((void**)&core_d.p, offset_time_sig * sizeof(float) * 2);
    cudaMalloc((void**)&core_d.rp, offset_time_sig * sizeof(float) * 2);


    cudaMemset(core_d.vx, 0, offset_time_vx * sizeof(float) * 2);
    cudaMemset(core_d.vz, 0, offset_time_vz * sizeof(float) * 2);

    cudaMemset(core_d.sx, 0, offset_time_sig * sizeof(float) * 2);
    cudaMemset(core_d.sz, 0, offset_time_sig * sizeof(float) * 2);
    cudaMemset(core_d.txz, 0, offset_time_txz * sizeof(float) * 2);
    cudaMemset(core_d.rx, 0, offset_time_sig * sizeof(float) * 2);
    cudaMemset(core_d.rz, 0, offset_time_sig * sizeof(float) * 2);
    cudaMemset(core_d.rxz, 0, offset_time_txz * sizeof(float) * 2);

    cudaMemset(core_d.p, 0, offset_time_sig * sizeof(float) * 2);
    cudaMemset(core_d.rp, 0, offset_time_sig * sizeof(float) * 2);

    // device temp core (1 time layer)
    cudaMalloc((void**)&core_temp.vx, (offset_time_vx - offset_coarse_vx) * sizeof(float));
    cudaMalloc((void**)&core_temp.vz, (offset_time_vz - offset_coarse_vz) * sizeof(float));

    cudaMalloc((void**)&core_temp.sx, (offset_time_sig - offset_coarse_sig) * sizeof(float));
    cudaMalloc((void**)&core_temp.sz, (offset_time_sig - offset_coarse_sig) * sizeof(float));
    cudaMalloc((void**)&core_temp.txz, (offset_time_txz - offset_coarse_txz) * sizeof(float));
    cudaMalloc((void**)&core_temp.rx, (offset_time_sig - offset_coarse_sig) * sizeof(float));
    cudaMalloc((void**)&core_temp.rz, (offset_time_sig - offset_coarse_sig) * sizeof(float));
    cudaMalloc((void**)&core_temp.rxz, (offset_time_txz - offset_coarse_txz) * sizeof(float));

    cudaMalloc((void**)&core_temp.rp, (offset_time_sig - offset_coarse_sig) * sizeof(float));
    cudaMalloc((void**)&core_temp.p, (offset_time_sig - offset_coarse_sig) * sizeof(float));

    cudaMemset(core_temp.vx, 0, (offset_time_vx - offset_coarse_vx) * sizeof(float));
    cudaMemset(core_temp.vz, 0, (offset_time_vz - offset_coarse_vz) * sizeof(float));

    cudaMemset(core_temp.sx, 0, (offset_time_sig - offset_coarse_sig) * sizeof(float));
    cudaMemset(core_temp.sz, 0, (offset_time_sig - offset_coarse_sig) * sizeof(float));
    cudaMemset(core_temp.txz, 0, (offset_time_txz - offset_coarse_txz) * sizeof(float));
    cudaMemset(core_temp.rx, 0, (offset_time_sig - offset_coarse_sig) * sizeof(float));
    cudaMemset(core_temp.rz, 0, (offset_time_sig - offset_coarse_sig) * sizeof(float));
    cudaMemset(core_temp.rxz, 0, (offset_time_txz - offset_coarse_txz) * sizeof(float));

    cudaMemset(core_temp.p, 0, (offset_time_sig - offset_coarse_sig) * sizeof(float));
    cudaMemset(core_temp.rp, 0, (offset_time_sig - offset_coarse_sig) * sizeof(float));

    // mask arrays (device)
    cudaMalloc((void**)&core_mask.vx, (nx_coarse - 1) * nz_coarse * sizeof(int));
    cudaMalloc((void**)&core_mask.vz, nx_coarse * (nz_coarse - 1) * sizeof(int));
    cudaMalloc((void**)&core_mask.sig, nx_coarse * nz_coarse * sizeof(int));
    cudaMalloc((void**)&core_mask.txz, (nx_coarse - 1) * (nz_coarse - 1) * sizeof(int));

    // host model
    model_h.rho = new float[offset_time_sig]();
    model_h.C11 = new float[offset_time_sig]();
    model_h.C13 = new float[offset_time_sig]();
    model_h.C33 = new float[offset_time_sig]();
    model_h.C55 = new float[offset_time_sig]();
    model_h.taup = new float[offset_time_sig]();
    model_h.taus = new float[offset_time_sig]();
    model_h.inv_tsig = new float[offset_time_sig]();
    model_h.mat = new MAT_FLAG[offset_time_sig]();

    // device model
    cudaMalloc((void**)&model_d.rho, offset_time_sig * sizeof(float));
    cudaMalloc((void**)&model_d.C11, offset_time_sig * sizeof(float));
    cudaMalloc((void**)&model_d.C13, offset_time_sig * sizeof(float));
    cudaMalloc((void**)&model_d.C33, offset_time_sig * sizeof(float));
    cudaMalloc((void**)&model_d.C55, offset_time_sig * sizeof(float));
    cudaMalloc((void**)&model_d.taup, offset_time_sig * sizeof(float));
    cudaMalloc((void**)&model_d.taus, offset_time_sig * sizeof(float));
    cudaMalloc((void**)&model_d.inv_tsig, offset_time_sig * sizeof(float));
    cudaMalloc((void**)&model_d.mat, offset_time_sig * sizeof(MAT_FLAG));

    // compute n arrays total size
    // int vx_n_size = 0, vz_n_size = 0, sx_n_size = 0, sz_n_size = 0, txz_n_size = 0;
    int n_size = 0;
    for (int i = 0; i < fine_info.size(); i++) {
        int lz = (fine_info[i].z_end - fine_info[i].z_start) * fine_info[i].N + 1;
        int lx = (fine_info[i].x_end - fine_info[i].x_start) * fine_info[i].N + 1;
        n_size += lx * lz;
    }

    // device n arrays
    cudaMalloc((void**)&vx_n, n_size * sizeof(FD_n));
    cudaMalloc((void**)&vz_n, n_size * sizeof(FD_n));
    cudaMalloc((void**)&sig_n, n_size * sizeof(FD_n));
    cudaMalloc((void**)&txz_n, n_size * sizeof(FD_n));
}

void GridManager::build_insterp_LUT() {
    const int m = LAGRANGE_ORDER / 2;
    float h_coeff[LUT_SIZE * LAGRANGE_ORDER];

    for (int idx = 0; idx < LUT_SIZE; idx++) {
        double t = 1.0 * idx / LUT_SIZE;
        for (int k = 0; k < LAGRANGE_ORDER; ++k) {
            double xk = -m + k;
            double w = 1.0;
            for (int j = 0; j < LAGRANGE_ORDER; ++j) {
                if (j != k) {
                    double xj = -m + j;
                    w *= (t - xj) / (xk - xj);
                }
            }
            h_coeff[idx * LAGRANGE_ORDER + k] = (float)w;
        }
    }
    cudaMemcpyToSymbol(lagrange_coeff, h_coeff, sizeof(float) * LUT_SIZE * LAGRANGE_ORDER);
}

void GridManager::memory_release() {
    // host core
    if (core_h.vx) { delete[] core_h.vx; core_h.vx = nullptr; }
    if (core_h.vz) { delete[] core_h.vz; core_h.vz = nullptr; }
    if (core_h.sx) { delete[] core_h.sx; core_h.sx = nullptr; }
    if (core_h.sz) { delete[] core_h.sz; core_h.sz = nullptr; }
    if (core_h.txz) { delete[] core_h.txz; core_h.txz = nullptr; }
    if (core_h.rx) { delete[] core_h.rx; core_h.rx = nullptr; }
    if (core_h.rz) { delete[] core_h.rz; core_h.rz = nullptr; }
    if (core_h.rxz) { delete[] core_h.rxz; core_h.rxz = nullptr; }
    if (core_h.p) { delete[] core_h.p; core_h.p = nullptr; }
    if (core_h.rp) { delete[] core_h.rp; core_h.rp = nullptr; }

    // device core
    if (core_d.vx) { cudaFree(core_d.vx); core_d.vx = nullptr; }
    if (core_d.vz) { cudaFree(core_d.vz); core_d.vz = nullptr; }
    if (core_d.sx) { cudaFree(core_d.sx); core_d.sx = nullptr; }
    if (core_d.sz) { cudaFree(core_d.sz); core_d.sz = nullptr; }
    if (core_d.txz) { cudaFree(core_d.txz); core_d.txz = nullptr; }
    if (core_d.p) { cudaFree(core_d.p); core_d.p = nullptr; }
    if (core_d.rp) { cudaFree(core_d.rp); core_d.rp = nullptr; }
    if (core_d.rx) { cudaFree(core_d.rx); core_d.rx = nullptr; }
    if (core_d.rz) { cudaFree(core_d.rz); core_d.rz = nullptr; }
    if (core_d.rxz) { cudaFree(core_d.rxz); core_d.rxz = nullptr; }

    if (core_temp.vx) { cudaFree(core_temp.vx); core_temp.vx = nullptr; }
    if (core_temp.vz) { cudaFree(core_temp.vz); core_temp.vz = nullptr; }
    if (core_temp.sx) { cudaFree(core_temp.sx); core_temp.sx = nullptr; }
    if (core_temp.sz) { cudaFree(core_temp.sz); core_temp.sz = nullptr; }
    if (core_temp.txz) { cudaFree(core_temp.txz); core_temp.txz = nullptr; }
    if (core_temp.rp) { cudaFree(core_temp.rp); core_temp.rp = nullptr; }
    if (core_temp.p) { cudaFree(core_temp.p); core_temp.p = nullptr; }
    if (core_temp.rx) { cudaFree(core_temp.rx); core_temp.rx = nullptr; }
    if (core_temp.rz) { cudaFree(core_temp.rz); core_temp.rz = nullptr; }
    if (core_temp.rxz) { cudaFree(core_temp.rxz); core_temp.rxz = nullptr; }

    // mask arrays (device)
    if (core_mask.vx) { cudaFree(core_mask.vx); core_mask.vx = nullptr; }
    if (core_mask.vz) { cudaFree(core_mask.vz); core_mask.vz = nullptr; }
    if (core_mask.sig) { cudaFree(core_mask.sig); core_mask.sig = nullptr; }
    if (core_mask.txz) { cudaFree(core_mask.txz); core_mask.txz = nullptr; }

    // host model
    if (model_h.rho) { delete[] model_h.rho; model_h.rho = nullptr; }
    if (model_h.C11) { delete[] model_h.C11; model_h.C11 = nullptr; }
    if (model_h.C13) { delete[] model_h.C13; model_h.C13 = nullptr; }
    if (model_h.C33) { delete[] model_h.C33; model_h.C33 = nullptr; }
    if (model_h.C55) { delete[] model_h.C55; model_h.C55 = nullptr; }
    if (model_h.taup) { delete[] model_h.taup; model_h.taup = nullptr; }
    if (model_h.taus) { delete[] model_h.taus; model_h.taus = nullptr; }
    if (model_h.inv_tsig) { delete[] model_h.inv_tsig; model_h.inv_tsig = nullptr; }
    if (model_h.mat) { delete[] model_h.mat; model_h.mat = nullptr; }

    // device model
    if (model_d.rho) { cudaFree(model_d.rho); model_d.rho = nullptr; }
    if (model_d.C11) { cudaFree(model_d.C11); model_d.C11 = nullptr; }
    if (model_d.C13) { cudaFree(model_d.C13); model_d.C13 = nullptr; }
    if (model_d.C33) { cudaFree(model_d.C33); model_d.C33 = nullptr; }
    if (model_d.C55) { cudaFree(model_d.C55); model_d.C55 = nullptr; }
    if (model_d.taup) { cudaFree(model_d.taup); model_d.taup = nullptr; }
    if (model_d.taus) { cudaFree(model_d.taus); model_d.taus = nullptr; }
    if (model_d.inv_tsig) { cudaFree(model_d.inv_tsig); model_d.inv_tsig = nullptr; }
    if (model_d.mat) { cudaFree(model_d.mat); model_d.mat = nullptr; }

    // 先销毁纹理对象
    cudaDestroyTextureObject(tex_vx_mask);
    cudaDestroyTextureObject(tex_vz_mask);
    cudaDestroyTextureObject(tex_sig_mask);
    cudaDestroyTextureObject(tex_txz_mask);
    cudaDestroyTextureObject(tex_mat);

    cudaDestroyTextureObject(tex_vx_n);
    cudaDestroyTextureObject(tex_vz_n);
    cudaDestroyTextureObject(tex_sig_n);
    cudaDestroyTextureObject(tex_txz_n);

    // FD_n arrays (device)
    if (vx_n) { cudaFree(vx_n); vx_n = nullptr; }
    if (vz_n) { cudaFree(vz_n); vz_n = nullptr; }
    if (sig_n) { cudaFree(sig_n); sig_n = nullptr; }
    if (txz_n) { cudaFree(txz_n); txz_n = nullptr; }
}