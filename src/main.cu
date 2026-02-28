#include <vector>
#include <iostream>
#include <filesystem>
#include "kernels.cuh"
#include "params.cuh"

namespace fs = std::filesystem;

float buffer[2048];

void output_snapshots(GridManager &gm, int it, float dt, int time);
void output_record(const GridManager &gm, int z, FILE *fp, int time);
bool clear_folder(const fs::path& dir);

__global__ void debug_kernel(float *f, int time, int it) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("sx(%d, %d)=%f, it=%d\n", 125, 125, f[125 * 500 + 125 + time * offset_sx_all], it);
    // printf("sx(%d, %d)=%f, it=%d\n", 125, 126, f[126 * 500 + 125 + time * offset_sx_all], it);
    // printf("sx(%d, %d)=%f, it=%d\n", 126, 125, f[125 * 500 + 126 + time * offset_sx_all], it);
    // printf("sx(%d, %d)=%f, it=%d\n", 124, 125, f[125 * 500 + 124 + time * offset_sx_all], it);
    // printf("sx(%d, %d)=%f, it=%d\n", 125, 124, f[124 * 500 + 125 + time * offset_sx_all], it);
    if (ix == 0 && iz == 0) {
        printf("\n========== Constant Memory Dump ==========\n");

        // 基本网格参数
        printf("dx = %f, dz = %f\n", dx, dz);
        printf("nx = %d, nz = %d\n", nx, nz);

        // 各场的时间层偏移量
        printf("offset_vx_all = %d\n", offset_vx_all);
        printf("offset_vz_all = %d\n", offset_vz_all);
        printf("offset_sx_all = %d\n", offset_sx_all);
        printf("offset_sz_all = %d\n", offset_sz_all);
        printf("offset_txz_all = %d\n", offset_txz_all);

        // 震源位置和时间步长
        printf("posx_d = %d, posz_d = %d\n", posx_d, posz_d);
        printf("dt_d = %f, nt_d = %d\n", dt_d, nt_d);

        // CPML 参数
        printf("thickness_d = %d\n", thickness_d);
        for (int i = 0; i < thickness_d; i++) {
            printf("a_int_d[%d] = %e, b_int_d[%d] = %e, kappa_int_d[%d] = %f\n",
                   i, a_int_d[i], i, b_int_d[i], i, kappa_int_d[i]);
        }
        for (int i = 0; i < thickness_d - 1; i++) {
            printf("a_half_d[%d] = %e, b_half_d[%d] = %e, kappa_half_d[%d] = %f\n",
                   i, a_half_d[i], i, b_half_d[i], i, kappa_half_d[i]);
        }

        // 细网格信息
        printf("num_fine = %d\n", num_fine);
        for (int i = 0; i < num_fine; i++) {
            printf("fines[%d]: x_start=%d, x_end=%d, z_start=%d, z_end=%d, lenx=%d, lenz=%d, N=%d, dx_fine=%f, dz_fine=%f\n",
                   i,
                   fines[i].x_start,
                   fines[i].x_end,
                   fines[i].z_start,
                   fines[i].z_end,
                   fines[i].lenx,
                   fines[i].lenz,
                   fines[i].N,
                   fines[i].dx_fine,
                   fines[i].dz_fine);
        }

        // 细网格在总数组中的起始偏移
        for (int i = 0; i < num_fine; i++) {
            printf("sum_offset_fine_vx[%d] = %d\n", i, sum_offset_fine_vx[i]);
            printf("sum_offset_fine_vz[%d] = %d\n", i, sum_offset_fine_vz[i]);
            printf("sum_offset_fine_sx[%d] = %d\n", i, sum_offset_fine_sx[i]);
            printf("sum_offset_fine_sz[%d] = %d\n", i, sum_offset_fine_sz[i]);
            printf("sum_offset_fine_txz[%d] = %d\n", i, sum_offset_fine_txz[i]);
        }

    //     // 拉格朗日插值系数（可选，注释掉以节省输出）
    //     /*
    //     for (int i = 0; i < LUT_SIZE * LAGRANGE_ORDER; i++) {
    //         printf("lagrange_coeff[%d] = %f\n", i, lagrange_coeff[i]);
    //     }
    //     */

    //     printf("========== End of Dump ==========\n");
    }
}


int main() {
    GridManager gm("models/models.json");
    Params params("models/params.json");
    Cpml cpml("models/params.json");
    gm.memcpy_model_h2d();
    
    std::unique_ptr<float[]> wavelet = params.ricker_wavelet();
    const int NUM_STREAM = gm.fine_info.size() + 1;

    cudaStream_t stream_co;
    cudaStreamCreate(&stream_co);
    std::vector<cudaStream_t> stream_fi(gm.fine_info.size());
    for (int i = 0; i < gm.fine_info.size(); ++i) {
        cudaStreamCreate(&stream_fi[i]);
    }

    clear_folder("./output");
    system("mkdir -p ./output/record");
    system("mkdir -p ./output/vx");
    system("mkdir -p ./output/vz");
    system("mkdir -p ./output/sx");
    system("mkdir -p ./output/sz");
    system("mkdir -p ./output/txz");

    FILE *fp = fopen("output/record/record_vz.bin", "wb");
    if (!fp) {
        std::cerr << "Failed to open output file for recording." << std::endl;
        return -1;
    }

    // int *temp = new int[500 * 500]();
    // cudaMemcpy(temp, gm.core_mask.sx, 500 * 500 * sizeof(int), cudaMemcpyDeviceToHost);
    // char buf[32];
    // snprintf(buf, sizeof(buf), "output/mask/mask.bin");
    // std::string filename = buf;
    // FILE *fpp = fopen(filename.c_str(), "wb");
    // fwrite(temp, sizeof(int), 500 * 500, fpp);
    // fclose(fpp);
    // delete[] temp;

    // debug_kernel<<<1, 1>>>(nullptr, 0, 0);
    for (int it = 0; it < params.nt; it++) {
        int cur = it & 1;
        
        dim3 grid_co((gm.nx_coarse + 15) / 16, (gm.nz_coarse + 15) / 16);
        dim3 block_co(16, 16);
        update_stress_coarse<<<grid_co, block_co, 0, stream_co>>>(gm.core_d, gm.model_d, cpml.psi_vel, cur);
        for (int i = 0; i < gm.fine_info.size(); i++) {
            dim3 grid_fi((gm.fine_info[i].lenx + 15) / 16, (gm.fine_info[i].lenz + 15) / 16);
            dim3 block_fi(16, 16);
            update_stress_fine<<<grid_fi, block_fi, 0, stream_fi[i]>>>(gm.core_d, gm.model_d, cur, i);
        }
        apply_source<<<1, 1, 0, stream_co>>>(gm.core_d, wavelet[it], cur);
        update_velocity_coarse<<<grid_co, block_co, 0, stream_co>>>(gm.core_d, gm.model_d, cpml.psi_str, cur);
        
        for (int i = 0; i < gm.fine_info.size(); i++) {
            dim3 grid_fi((gm.fine_info[i].lenx + 15) / 16, (gm.fine_info[i].lenz + 15) / 16);
            dim3 block_fi(16, 16);

            update_velocity_fine<<<grid_fi, block_fi, 0, stream_fi[i]>>>(gm.core_d, gm.model_d, cur, i);
        }

        cudaStreamSynchronize(stream_co);
        for (int i = 0; i < gm.fine_info.size(); i++) {
            cudaStreamSynchronize(stream_fi[i]);
        }

        output_record(gm, params.posz, fp, cur);

        if (it % params.snapshot == 0) {
            output_snapshots(gm, it, params.dt, cur);
            printf("finished %0.2f%%\r", 100.0 * it / params.nt);
            fflush(stdout);
        }
    }
    
    printf("finished 100.00%%\n");
    fclose(fp);
    cudaStreamDestroy(stream_co);
    for (int i = 0; i < gm.fine_info.size(); ++i) {
        cudaStreamDestroy(stream_fi[i]);
    }
}

void output_snapshots(GridManager &gm, int it, float dt, int time) {
    float time_sec = it * dt;
    int time_ms = static_cast<int>(time_sec * 1000);
    
    char buf[32];

    snprintf(buf, sizeof(buf), "output/vx/vx_%05dms.bin", time_ms);
    std::string filename_vx = buf;
    
    snprintf(buf, sizeof(buf), "output/vz/vz_%05dms.bin", time_ms);
    std::string filename_vz = buf;

    snprintf(buf, sizeof(buf), "output/sx/sx_%05dms.bin", time_ms);
    std::string filename_sx = buf;

    snprintf(buf, sizeof(buf), "output/sz/sz_%05dms.bin", time_ms);
    std::string filename_sz = buf;
    
    snprintf(buf, sizeof(buf), "output/txz/txz_%05dms.bin", time_ms);
    std::string filename_txz = buf;

    gm.memcpy_core_d2h(time);
    
    FILE *fp_vx = fopen(filename_vx.c_str(), "wb");
    FILE *fp_vz = fopen(filename_vz.c_str(), "wb");
    FILE *fp_sx = fopen(filename_sx.c_str(), "wb");
    FILE *fp_sz = fopen(filename_sz.c_str(), "wb");
    FILE *fp_txz = fopen(filename_txz.c_str(), "wb");

    fwrite(gm.core_h.vx, sizeof(float), gm.offset_time_vx, fp_vx);
    fwrite(gm.core_h.vz, sizeof(float), gm.offset_time_vz, fp_vz);
    fwrite(gm.core_h.sx, sizeof(float), gm.offset_time_sx, fp_sx);
    fwrite(gm.core_h.sz, sizeof(float), gm.offset_time_sz, fp_sz);
    fwrite(gm.core_h.txz, sizeof(float), gm.offset_time_txz, fp_txz);

    fclose(fp_vx);
    fclose(fp_vz);
    fclose(fp_sx);
    fclose(fp_sz);
    fclose(fp_txz);
}

void output_record(const GridManager &gm, int z, FILE *fp, int time) {
    cudaMemcpy(
        buffer, 
        gm.core_d.vx + time * gm.offset_time_sx + z * (gm.nx_coarse - 1), 
        sizeof(float) * (gm.nx_coarse - 1), 
        cudaMemcpyDeviceToHost
    );
    fwrite(buffer, sizeof(float), gm.nx_coarse - 1, fp);
}

bool clear_folder(const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    if (ec) {
        return 0;
    }
    return fs::create_directory(dir, ec);
}