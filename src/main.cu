#include <vector>
#include <iostream>
#include "kernels.cuh"
#include "params.cuh"

float buffer[2048];

void output_snapshots(GridManager &gm, int it, float dt, int time);
void output_record(const GridManager &gm, int z, FILE *fp, int time);
// __global__ void check_mask(cudaTextureObject_t tex) {
//     int ix = blockIdx.x * blockDim.x + threadIdx.x + 418;
//     int iz = blockIdx.y * blockDim.y + threadIdx.y + 418;
//     if (ix >= 500 || iz >= 500) {
//         return;
//     }
//     int mask = tex1Dfetch<int>(vx_mask, iz * 499 + ix);
//     // int tex_z = tex1D<int2>(vx_n_tex, iz * (fines[0].lenx - 1) + ix).y;
//     printf("ix: %d, iz: %d, mask: %d\n", ix, iz, mask);
// }


int main() {
    GridManager gm("models/models.json");
    Params params("models/params.json");
    Cpml cpml("models/params.json");
    gm.memcpy_model_h2d();
    
    std::unique_ptr<float[]> wavelet = params.ricker_wavelet();
    const int NUM_STREAM = gm.fine_info.size() + 1;

    cudaStream_t stream_co;
    cudaStreamCreate(&stream_co);
    std::vector<cudaStream_t> stream_fi(NUM_STREAM);
    for (int i = 0; i < NUM_STREAM; ++i) {
        cudaStreamCreate(&stream_fi[i]);
    }

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
    dim3 g((gm.fine_info[0].lenx + 15) / 16, (gm.fine_info[0].lenz + 15) / 16);
    dim3 bb(16, 16);
    // check_mask<<<g, bb>>>(gm.tex_vx_n);

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
    for (int i = 0; i < NUM_STREAM; ++i) {
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