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
__device__ cudaTextureObject_t sx_mask;
__device__ cudaTextureObject_t sz_mask;
__device__ cudaTextureObject_t txz_mask;

__device__ cudaTextureObject_t vx_n_tex;
__device__ cudaTextureObject_t vz_n_tex;
__device__ cudaTextureObject_t sx_n_tex;
__device__ cudaTextureObject_t sz_n_tex;
__device__ cudaTextureObject_t txz_n_tex;

__constant__ float dx, dz;
__constant__ int nx, nz;
__constant__ int stride_vx;
__constant__ int stride_vz;
__constant__ int stride_sx;
__constant__ int stride_sz;
__constant__ int stride_txz;
__constant__ int offset_vx_all;
__constant__ int offset_vz_all;
__constant__ int offset_sx_all;
__constant__ int offset_sz_all;
__constant__ int offset_txz_all;

__constant__ int num_fine;
__constant__ FineInfo fines[12];
__constant__ int sum_offset_fine_vx[12];
__constant__ int sum_offset_fine_vz[12];
__constant__ int sum_offset_fine_sx[12];
__constant__ int sum_offset_fine_sz[12];
__constant__ int sum_offset_fine_txz[12];

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
    offset_time_sx = nx_coarse * nz_coarse;
    offset_time_sz = nx_coarse * nz_coarse;
    offset_time_txz = (nx_coarse - 1) * (nz_coarse - 1);

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
        offset_time_sx += fine_info[i].lenx * fine_info[i].lenz;
        offset_time_sz += fine_info[i].lenx * fine_info[i].lenz;
        offset_time_txz += (fine_info[i].lenx - 1) * (fine_info[i].lenz - 1);
    }

    memory_allocate();

    // 读取模型参数文件
    struct Pair { float *ptr; std::string name; } dst[5] = {
        { model_h.rho, "rho" }, { model_h.C11, "C11" }, { model_h.C13, "C13" },
        { model_h.C33, "C33" }, { model_h.C55, "C55" }
    };
    std::string filename;

    cJSON *item = nullptr;
    // coarse
    for (int i = 0; i < 5; i++) {
        item = coarse_ptr;
        item = cJSON_GetObjectItem(item, dst[i].name.c_str());
        filename = item->valuestring;
        FILE *fp = fopen(filename.c_str(), "rb");
        if (!fp) {
            std::cout << "无法打开文件: " << filename << '\n';
            exit(1);
        }
        fread(dst[i].ptr, sizeof(float), nx_coarse * nz_coarse, fp);
        fclose(fp);
    }

    // fine
    for (int i = 0, offset = nx_coarse * nz_coarse; i < num; i++, offset += fine_info[i].lenx * fine_info[i].lenz) {
        for (int j = 0; j < 5; j++) {
            item = fine_ptr;
            item = cJSON_GetArrayItem(item, i);
            item = cJSON_GetObjectItem(item, dst[j].name.c_str());
            filename = item->valuestring;
            FILE *fp = fopen(filename.c_str(), "rb");
            if (!fp) {
                std::cout << "无法打开文件: " << filename << '\n';
                exit(1);
            }
            fread(dst[j].ptr + offset, sizeof(float), fine_info[i].lenx * fine_info[i].lenz, fp);
            fclose(fp);
        }
    }

    cJSON_Delete(root);

    build_mask();
    build_n();
    build_constant();
    build_insterp_LUT();
}

GridManager::~GridManager() {
    memory_release();
}

Core GridManager::get_core() { return core_d; }
Model GridManager::get_model() { return model_d; }

void GridManager::memcpy_core_d2h(int time) {
    int bytes_vx = offset_time_vx * sizeof(float);
    int bytes_vz = offset_time_vz * sizeof(float);
    int bytes_sx = offset_time_sx * sizeof(float);
    int bytes_sz = offset_time_sz * sizeof(float);
    int bytes_txz = offset_time_txz * sizeof(float);
    cudaMemcpy(core_h.vx, core_d.vx + time * bytes_vx, bytes_vx, cudaMemcpyDeviceToHost);
    cudaMemcpy(core_h.vz, core_d.vz + time * bytes_vz, bytes_vz, cudaMemcpyDeviceToHost);
    cudaMemcpy(core_h.sx, core_d.sx + time * bytes_sx, bytes_sx, cudaMemcpyDeviceToHost);
    cudaMemcpy(core_h.sz, core_d.sz + time * bytes_sz, bytes_sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(core_h.txz, core_d.txz + time * bytes_txz, bytes_txz, cudaMemcpyDeviceToHost);
}

void GridManager::memcpy_model_h2d() {
    int bytes_sx = offset_time_sx * sizeof(float);
    cudaMemcpy(model_d.rho, model_h.rho, bytes_sx, cudaMemcpyHostToDevice);
    cudaMemcpy(model_d.C11, model_h.C11, bytes_sx, cudaMemcpyHostToDevice);
    cudaMemcpy(model_d.C13, model_h.C13, bytes_sx, cudaMemcpyHostToDevice);
    cudaMemcpy(model_d.C33, model_h.C33, bytes_sx, cudaMemcpyHostToDevice);
    cudaMemcpy(model_d.C55, model_h.C55, bytes_sx, cudaMemcpyHostToDevice);
}

void GridManager::build_mask() {
    int *vx_mask_h = new int[(nx_coarse - 1) * nz_coarse]();
    int *vz_mask_h = new int[nx_coarse * (nz_coarse - 1)]();
    int *sx_mask_h = new int[nx_coarse * nz_coarse]();
    int *sz_mask_h = new int[nx_coarse * nz_coarse]();
    int *txz_mask_h = new int[(nx_coarse - 1) * (nz_coarse - 1)]();

    memset(vx_mask_h, -1, (nx_coarse - 1) * nz_coarse * sizeof(int));
    memset(vz_mask_h, -1, nx_coarse * (nz_coarse - 1) * sizeof(int));
    memset(sx_mask_h, -1, nx_coarse * nz_coarse * sizeof(int));
    memset(sz_mask_h, -1, nx_coarse * nz_coarse * sizeof(int));
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
                    sx_mask_h[iz * nx_coarse + ix] = i;
                    sz_mask_h[iz * nx_coarse + ix] = i;
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
    int bytes_sx_mask = nx_coarse * nz_coarse * sizeof(int);
    int bytes_sz_mask = nx_coarse * nz_coarse * sizeof(int);
    int bytes_txz_mask = (nx_coarse - 1) * (nz_coarse - 1) * sizeof(int);

    cudaMemcpy(core_mask.vx, vx_mask_h, bytes_vx_mask, cudaMemcpyHostToDevice);
    cudaMemcpy(core_mask.vz, vz_mask_h, bytes_vz_mask, cudaMemcpyHostToDevice);
    cudaMemcpy(core_mask.sx, sx_mask_h, bytes_sx_mask, cudaMemcpyHostToDevice);
    cudaMemcpy(core_mask.sz, sz_mask_h, bytes_sz_mask, cudaMemcpyHostToDevice);
    cudaMemcpy(core_mask.txz, txz_mask_h, bytes_txz_mask, cudaMemcpyHostToDevice);
    // 创建纹理对象并保存到成员变量
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

    // sx_mask
    resDesc.res.linear.devPtr = core_mask.sx;
    resDesc.res.linear.sizeInBytes = bytes_sx_mask;
    cudaCreateTextureObject(&tex_sx_mask, &resDesc, &texDesc, NULL);
    cudaMemcpyToSymbol(sx_mask, &tex_sx_mask, sizeof(cudaTextureObject_t));

    // sz_mask
    resDesc.res.linear.devPtr = core_mask.sz;
    resDesc.res.linear.sizeInBytes = bytes_sz_mask;
    cudaCreateTextureObject(&tex_sz_mask, &resDesc, &texDesc, NULL);
    cudaMemcpyToSymbol(sz_mask, &tex_sz_mask, sizeof(cudaTextureObject_t));

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

    delete[] vx_mask_h;
    delete[] vz_mask_h;
    delete[] sx_mask_h;
    delete[] sz_mask_h;
    delete[] txz_mask_h;
}

void GridManager::build_n() {
    if (FINE == FINE_OFF) return;
    int offset_vx_n = 0;
    int offset_vz_n = 0;
    int offset_sx_n = 0;
    int offset_sz_n = 0;
    int offset_txz_n = 0;
    for (int i = 0; i < fine_info.size(); i++) {
        offset_vx_n += (fine_info[i].lenx - 1) * fine_info[i].lenz;
        offset_vz_n += fine_info[i].lenx * (fine_info[i].lenz - 1);
        offset_sx_n += fine_info[i].lenx * fine_info[i].lenz;
        offset_sz_n += fine_info[i].lenx * fine_info[i].lenz;
        offset_txz_n += (fine_info[i].lenx - 1) * (fine_info[i].lenz - 1);
    }

    FD_n *vx_n_h = new FD_n[offset_vx_n]();
    FD_n *vz_n_h = new FD_n[offset_vz_n]();
    FD_n *sx_n_h = new FD_n[offset_sx_n]();
    FD_n *sz_n_h = new FD_n[offset_sz_n]();
    FD_n *txz_n_h = new FD_n[offset_txz_n]();

    offset_vx_n = 0;
    offset_vz_n = 0;
    offset_sx_n = 0;
    offset_sz_n = 0;
    offset_txz_n = 0;

    for (int i = 0; i < fine_info.size(); i++) {
        int z_start = fine_info[i].z_start;
        int z_end = fine_info[i].z_end;
        int x_start = fine_info[i].x_start;
        int x_end = fine_info[i].x_end;
        int lz = (z_end - z_start) * fine_info[i].N + 1;
        int lx = (x_end - x_start) * fine_info[i].N + 1;
        fine_info[i].lenx = lx;
        fine_info[i].lenz = lz;

        for (int iz = 0; iz < lz; iz++) {
            for (int ix = 0; ix < lx; ix++) {
                int n_x_int = 0, n_x_half = 0, n_z_int = 0, n_z_half = 0;
                if (0 <= iz && iz < lz) {
                    n_z_int = std::min({ iz - 0 + 1, lz - 2 - iz + 1, 4 });
                }
                if (0 <= ix && ix < lx) {
                    n_x_int = std::min({ ix - 0 + 1, lx - 2 - ix + 1, 4 });
                }
                if (0 <= iz && iz < lz - 1) {
                    n_z_half = std::min({ iz - 1 + 1, lz - 1 - iz + 1, 4 });
                }
                if (0 <= ix && ix < lx - 1) {
                    n_x_half = std::min({ ix - 1 + 1, lx - 1 - ix + 1, 4 });
                }


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
                    vx_n_h[offset_vx_n + iz * (lx - 1) + ix].n_x = n_x_half;
                    vx_n_h[offset_vx_n + iz * (lx - 1) + ix].n_z = n_z_int;
                }
                // vz
                if (iz < lz - 1) {
                    vz_n_h[offset_vz_n + iz * lx + ix].n_x = n_x_int;
                    vz_n_h[offset_vz_n + iz * lx + ix].n_z = n_z_half;
                }
                // sx
                sx_n_h[offset_sx_n + iz * lx + ix].n_x = n_x_int;
                sx_n_h[offset_sx_n + iz * lx + ix].n_z = n_z_int;
                // sz
                sz_n_h[offset_sz_n + iz * lx + ix].n_x = n_x_int;
                sz_n_h[offset_sz_n + iz * lx + ix].n_z = n_z_int;
                // txz
                if (ix < lx - 1 && iz < lz - 1) {
                    txz_n_h[offset_txz_n + iz * (lx - 1) + ix].n_x = n_x_half;
                    txz_n_h[offset_txz_n + iz * (lx - 1) + ix].n_z = n_z_half;
                }
            }
        }

        offset_vx_n += (fine_info[i].lenx - 1) * fine_info[i].lenz;
        offset_vz_n += fine_info[i].lenx * (fine_info[i].lenz - 1);
        offset_sx_n += fine_info[i].lenx * fine_info[i].lenz;
        offset_sz_n += fine_info[i].lenx * fine_info[i].lenz;
        offset_txz_n += (fine_info[i].lenx - 1) * (fine_info[i].lenz - 1);
    }

    cudaMemcpy(vx_n, vx_n_h, offset_vx_n * sizeof(FD_n), cudaMemcpyHostToDevice);
    cudaMemcpy(vz_n, vz_n_h, offset_vz_n * sizeof(FD_n), cudaMemcpyHostToDevice);
    cudaMemcpy(sx_n, sx_n_h, offset_sx_n * sizeof(FD_n), cudaMemcpyHostToDevice);
    cudaMemcpy(sz_n, sz_n_h, offset_sz_n * sizeof(FD_n), cudaMemcpyHostToDevice);
    cudaMemcpy(txz_n, txz_n_h, offset_txz_n * sizeof(FD_n), cudaMemcpyHostToDevice);

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
    resDesc.res.linear.sizeInBytes = offset_vx_n * sizeof(FD_n);
    cudaCreateTextureObject(&tex_vx_n, &resDesc, &texDesc, NULL);
    cudaMemcpyToSymbol(vx_n_tex, &tex_vx_n, sizeof(cudaTextureObject_t));

    // vz_n
    resDesc.res.linear.devPtr = vz_n;
    resDesc.res.linear.sizeInBytes = offset_vz_n * sizeof(FD_n);
    cudaCreateTextureObject(&tex_vz_n, &resDesc, &texDesc, NULL);
    cudaMemcpyToSymbol(vz_n_tex, &tex_vz_n, sizeof(cudaTextureObject_t));

    // sx_n
    resDesc.res.linear.devPtr = sx_n;
    resDesc.res.linear.sizeInBytes = offset_sx_n * sizeof(FD_n);
    cudaCreateTextureObject(&tex_sx_n, &resDesc, &texDesc, NULL);
    cudaMemcpyToSymbol(sx_n_tex, &tex_sx_n, sizeof(cudaTextureObject_t));

    // sz_n
    resDesc.res.linear.devPtr = sz_n;
    resDesc.res.linear.sizeInBytes = offset_sz_n * sizeof(FD_n);
    cudaCreateTextureObject(&tex_sz_n, &resDesc, &texDesc, NULL);
    cudaMemcpyToSymbol(sz_n_tex, &tex_sz_n, sizeof(cudaTextureObject_t));

    // txz_n
    resDesc.res.linear.devPtr = txz_n;
    resDesc.res.linear.sizeInBytes = offset_txz_n * sizeof(FD_n);
    cudaCreateTextureObject(&tex_txz_n, &resDesc, &texDesc, NULL);
    cudaMemcpyToSymbol(txz_n_tex, &tex_txz_n, sizeof(cudaTextureObject_t));

    delete[] vx_n_h;
    delete[] vz_n_h;
    delete[] sx_n_h;
    delete[] sz_n_h;
    delete[] txz_n_h;
}

void GridManager::build_constant() {
    cudaMemcpyToSymbol(nx, &nx_coarse, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nz, &nz_coarse, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dx, &dx_coarse, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dz, &dz_coarse, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(offset_vx_all, &offset_time_vx, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(offset_vz_all, &offset_time_vz, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(offset_sx_all, &offset_time_sx, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(offset_sz_all, &offset_time_sz, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(offset_txz_all, &offset_time_txz, sizeof(int), 0, cudaMemcpyHostToDevice);

    if (FINE == FINE_OFF) return;
    int f_size = fine_info.size();
    cudaMemcpyToSymbol(num_fine, &f_size, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(fines, fine_info.data(), f_size * sizeof(FineInfo), 0, cudaMemcpyHostToDevice);

    std::vector<int> off_vx(f_size, 0);
    std::vector<int> off_vz(f_size, 0);
    std::vector<int> off_sx(f_size, 0);
    std::vector<int> off_sz(f_size, 0);
    std::vector<int> off_txz(f_size, 0);

    off_vx[0] = (nx_coarse - 1) * nz_coarse;
    off_vz[0] = nx_coarse * (nz_coarse - 1);
    off_sx[0] = nx_coarse * nz_coarse;
    off_sz[0] = nx_coarse * nz_coarse;
    off_txz[0] = (nx_coarse - 1) * (nz_coarse - 1);

    for (int i = 1; i < f_size; i++) {
        off_vx[i] = (fine_info[i - 1].lenx - 1) * fine_info[i - 1].lenz + off_vx[i - 1];
        off_vz[i] = fine_info[i - 1].lenx * (fine_info[i - 1].lenz - 1) + off_vz[i - 1];
        off_sx[i] = fine_info[i - 1].lenx * fine_info[i - 1].lenz + off_sx[i - 1];
        off_sz[i] = fine_info[i - 1].lenx * fine_info[i - 1].lenz + off_sz[i - 1];
        off_txz[i] = (fine_info[i - 1].lenx - 1) * (fine_info[i - 1].lenz - 1) + off_txz[i - 1];
    }

    cudaMemcpyToSymbol(stride_vx, off_vx.data(), sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(stride_vz, off_vz.data(), sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(stride_sx, off_sx.data(), sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(stride_sz, off_sz.data(), sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(stride_txz, off_txz.data(), sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(sum_offset_fine_vx, off_vx.data(), f_size * sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(sum_offset_fine_vz, off_vz.data(), f_size * sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(sum_offset_fine_sx, off_sx.data(), f_size * sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(sum_offset_fine_sz, off_sz.data(), f_size * sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(sum_offset_fine_txz, off_txz.data(), f_size * sizeof(int), 0, cudaMemcpyHostToDevice);
}

void GridManager::memory_allocate() {
    // host core
    core_h.vx = new float[offset_time_vx]();
    core_h.vz = new float[offset_time_vz]();
    core_h.sx = new float[offset_time_sx]();
    core_h.sz = new float[offset_time_sz]();
    core_h.txz = new float[offset_time_txz]();

    // device core (2 time layers)
    cudaMalloc((void**)&core_d.vx, offset_time_vx * sizeof(float) * 2);
    cudaMalloc((void**)&core_d.vz, offset_time_vz * sizeof(float) * 2);
    cudaMalloc((void**)&core_d.sx, offset_time_sx * sizeof(float) * 2);
    cudaMalloc((void**)&core_d.sz, offset_time_sz * sizeof(float) * 2);
    cudaMalloc((void**)&core_d.txz, offset_time_txz * sizeof(float) * 2);

    cudaMemset(core_d.vx, 0, offset_time_vx * sizeof(float) * 2);
    cudaMemset(core_d.vz, 0, offset_time_vz * sizeof(float) * 2);
    cudaMemset(core_d.sx, 0, offset_time_sx * sizeof(float) * 2);
    cudaMemset(core_d.sz, 0, offset_time_sz * sizeof(float) * 2);
    cudaMemset(core_d.txz, 0, offset_time_txz * sizeof(float) * 2);

    // mask arrays (device)
    cudaMalloc((void**)&core_mask.vx, (nx_coarse - 1) * nz_coarse * sizeof(int));
    cudaMalloc((void**)&core_mask.vz, nx_coarse * (nz_coarse - 1) * sizeof(int));
    cudaMalloc((void**)&core_mask.sx, nx_coarse * nz_coarse * sizeof(int));
    cudaMalloc((void**)&core_mask.sz, nx_coarse * nz_coarse * sizeof(int));
    cudaMalloc((void**)&core_mask.txz, (nx_coarse - 1) * (nz_coarse - 1) * sizeof(int));

    // host model
    model_h.rho = new float[offset_time_sx]();
    model_h.C11 = new float[offset_time_sx]();
    model_h.C13 = new float[offset_time_sx]();
    model_h.C33 = new float[offset_time_sx]();
    model_h.C55 = new float[offset_time_sx]();

    // device model
    cudaMalloc((void**)&model_d.rho, offset_time_sx * sizeof(float));
    cudaMalloc((void**)&model_d.C11, offset_time_sx * sizeof(float));
    cudaMalloc((void**)&model_d.C13, offset_time_sx * sizeof(float));
    cudaMalloc((void**)&model_d.C33, offset_time_sx * sizeof(float));
    cudaMalloc((void**)&model_d.C55, offset_time_sx * sizeof(float));

    // compute n arrays total size
    int vx_n_size = 0, vz_n_size = 0, sx_n_size = 0, sz_n_size = 0, txz_n_size = 0;
    for (int i = 0; i < fine_info.size(); i++) {
        int lz = (fine_info[i].z_end - fine_info[i].z_start) * fine_info[i].N + 1;
        int lx = (fine_info[i].x_end - fine_info[i].x_start) * fine_info[i].N + 1;
        vx_n_size += (lx - 1) * lz;
        vz_n_size += lx * (lz - 1);
        sx_n_size += lx * lz;
        sz_n_size += lx * lz;
        txz_n_size += (lx - 1) * (lz - 1);
    }

    // device n arrays
    cudaMalloc((void**)&vx_n, vx_n_size * sizeof(FD_n));
    cudaMalloc((void**)&vz_n, vz_n_size * sizeof(FD_n));
    cudaMalloc((void**)&sx_n, sx_n_size * sizeof(FD_n));
    cudaMalloc((void**)&sz_n, sz_n_size * sizeof(FD_n));
    cudaMalloc((void**)&txz_n, txz_n_size * sizeof(FD_n));
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

    // device core
    if (core_d.vx) { cudaFree(core_d.vx); core_d.vx = nullptr; }
    if (core_d.vz) { cudaFree(core_d.vz); core_d.vz = nullptr; }
    if (core_d.sx) { cudaFree(core_d.sx); core_d.sx = nullptr; }
    if (core_d.sz) { cudaFree(core_d.sz); core_d.sz = nullptr; }
    if (core_d.txz) { cudaFree(core_d.txz); core_d.txz = nullptr; }

    // mask arrays (device)
    if (core_mask.vx) { cudaFree(core_mask.vx); core_mask.vx = nullptr; }
    if (core_mask.vz) { cudaFree(core_mask.vz); core_mask.vz = nullptr; }
    if (core_mask.sx) { cudaFree(core_mask.sx); core_mask.sx = nullptr; }
    if (core_mask.sz) { cudaFree(core_mask.sz); core_mask.sz = nullptr; }
    if (core_mask.txz) { cudaFree(core_mask.txz); core_mask.txz = nullptr; }

    // host model
    if (model_h.rho) { delete[] model_h.rho; model_h.rho = nullptr; }
    if (model_h.C11) { delete[] model_h.C11; model_h.C11 = nullptr; }
    if (model_h.C13) { delete[] model_h.C13; model_h.C13 = nullptr; }
    if (model_h.C33) { delete[] model_h.C33; model_h.C33 = nullptr; }
    if (model_h.C55) { delete[] model_h.C55; model_h.C55 = nullptr; }

    // device model
    if (model_d.rho) { cudaFree(model_d.rho); model_d.rho = nullptr; }
    if (model_d.C11) { cudaFree(model_d.C11); model_d.C11 = nullptr; }
    if (model_d.C13) { cudaFree(model_d.C13); model_d.C13 = nullptr; }
    if (model_d.C33) { cudaFree(model_d.C33); model_d.C33 = nullptr; }
    if (model_d.C55) { cudaFree(model_d.C55); model_d.C55 = nullptr; }

    // 先销毁纹理对象（它们依赖下面的内存）
    cudaDestroyTextureObject(tex_vx_mask);
    cudaDestroyTextureObject(tex_vz_mask);
    cudaDestroyTextureObject(tex_sx_mask);
    cudaDestroyTextureObject(tex_sz_mask);
    cudaDestroyTextureObject(tex_txz_mask);

    cudaDestroyTextureObject(tex_vx_n);
    cudaDestroyTextureObject(tex_vz_n);
    cudaDestroyTextureObject(tex_sx_n);
    cudaDestroyTextureObject(tex_sz_n);
    cudaDestroyTextureObject(tex_txz_n);

    // FD_n arrays (device)
    if (vx_n) { cudaFree(vx_n); vx_n = nullptr; }
    if (vz_n) { cudaFree(vz_n); vz_n = nullptr; }
    if (sx_n) { cudaFree(sx_n); sx_n = nullptr; }
    if (sz_n) { cudaFree(sz_n); sz_n = nullptr; }
    if (txz_n) { cudaFree(txz_n); txz_n = nullptr; }
}