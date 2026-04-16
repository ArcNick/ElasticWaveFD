#include <vector>
#include <iostream>
#include <filesystem>
#include "kernels.cuh"
#include "params.cuh"
#include "fd_stencil.cuh"

namespace fs = std::filesystem;

float buffer[2048];

class StreamManager {
public:
    cudaStream_t stream_vx, stream_vz;
    cudaStream_t stream_sx, stream_sz, stream_txz, stream_p;
    cudaStream_t stream_rx, stream_rz, stream_rxz;
    StreamManager() {
        cudaStreamCreate(&stream_vx);
        cudaStreamCreate(&stream_vz);
        cudaStreamCreate(&stream_sx);
        cudaStreamCreate(&stream_sz);
        cudaStreamCreate(&stream_txz);
        cudaStreamCreate(&stream_p);
        cudaStreamCreate(&stream_rx);
        cudaStreamCreate(&stream_rz);
        cudaStreamCreate(&stream_rxz);
    }
    ~StreamManager() {
        cudaStreamDestroy(stream_vx);
        cudaStreamDestroy(stream_vz);
        cudaStreamDestroy(stream_sx);
        cudaStreamDestroy(stream_sz);
        cudaStreamDestroy(stream_p);
        cudaStreamDestroy(stream_rx);
        cudaStreamDestroy(stream_rz);
        cudaStreamDestroy(stream_rxz);
    }
};

void smooth_fine(GridManager &gm, StreamManager &stream_manager, int time);
void output_snapshots(GridManager &gm, int it, float dt, int time);
void output_record(const GridManager &gm, int z, FILE *fp_vz, int time);
bool clear_folder(const fs::path& dir);

// __global__ void debug_kernel(Model model, int it, int time) {
//     int ix = blockIdx.x * blockDim.x + threadIdx.x;
//     int iz = blockIdx.y * blockDim.y + threadIdx.y;
//     printf("%f\n", __ldg((float *)(0x5088f51e8)));
//     printf("%f\n", model.rho[IdxSigFi(1, 0, 0, 0)]);
//     printf("%p\n", model.rho + IdxSigFi(1, 0, 0, 0));
// }


int main() {
    GridManager gm("models/models.json");
    Params params("models/params.json");
    Cpml cpml("models/params.json");
    gm.memcpy_model_h2d();
    
    std::unique_ptr<float[]> wavelet = params.ricker_wavelet();

    cudaStream_t stream_co;
    cudaStreamCreate(&stream_co);
    std::vector<cudaStream_t> stream_fi(gm.fine_info.size());
    for (int i = 0; i < gm.fine_info.size(); ++i) {
        cudaStreamCreate(&stream_fi[i]);
    }
    StreamManager stream_manager;

    clear_folder("./output");
    system("mkdir -p ./output/record");
    system("mkdir -p ./output/vx");
    system("mkdir -p ./output/vz");
    system("mkdir -p ./output/sx");
    system("mkdir -p ./output/sz");
    system("mkdir -p ./output/txz");

    FILE *fp_record_vz = fopen("output/record/record_vz.bin", "wb");
    if (!fp_record_vz) {
        std::cerr << "Failed to open output file for recording.\n";
        return -1;
    }

    for (int it = 0; it < params.nt; it++) {
        int cur = it & 1;

        dim3 grid_co((gm.nx_coarse + 15) / 16, (gm.nz_coarse + 15) / 16);
        dim3 block(16, 16);
        update_sigma_coarse<<<grid_co, block, 0, stream_co>>>(gm.core_d, gm.model_d, cpml.psi_vel, cur, it);
        update_tau_coarse<<<grid_co, block, 0, stream_co>>>(gm.core_d, gm.model_d, cpml.psi_vel, cur, it);
        for (int i = 0; i < gm.fine_info.size(); i++) {
            dim3 grid_fi((gm.fine_info[i].lenx + 15) / 16, (gm.fine_info[i].lenz + 15) / 16);
            update_sigma_fine<<<grid_fi, block, 0, stream_fi[i]>>>(gm.core_d, gm.model_d, cur, i);
            update_tau_fine<<<grid_fi, block, 0, stream_fi[i]>>>(gm.core_d, gm.model_d, cur, i);
        }

        // apply_fluid_boundary_coarse<<<grid_co, block, 0, stream_co>>>(gm.core_d, cur);
        // for (int i = 0; i < gm.fine_info.size(); i++) {
        //     dim3 grid_fi((gm.fine_info[i].lenx + 15) / 16, (gm.fine_info[i].lenz + 15) / 16);
        //     apply_fluid_boundary_fine<<<grid_fi, block, 0, stream_fi[i]>>>(gm.core_d, cur, i);
        // }

        apply_source<<<1, 1, 0, stream_co>>>(gm.core_d, wavelet[it], cur);
        update_velocity_coarse<<<grid_co, block, 0, stream_co>>>(gm.core_d, gm.model_d, cpml.psi_str, cur, it);
        
        for (int i = 0; i < gm.fine_info.size(); i++) {
            dim3 grid_fi((gm.fine_info[i].lenx + 15) / 16, (gm.fine_info[i].lenz + 15) / 16);
            dim3 block_fi(16, 16);
            update_velocity_fine<<<grid_fi, block_fi, 0, stream_fi[i]>>>(gm.core_d, gm.model_d, cur, i);
        }

        cudaStreamSynchronize(stream_co);
        for (int i = 0; i < gm.fine_info.size(); i++) {
            cudaStreamSynchronize(stream_fi[i]);
        }

        output_record(gm, params.posz - 5, fp_record_vz, cur);
        if (it % 50 == 0) {
            smooth_fine(gm, stream_manager, cur);
        }

        if (it % params.snapshot == 0) {
            output_snapshots(gm, it, params.dt, cur);
            printf("finished %0.2f%%\r", 100.0 * it / params.nt);
            fflush(stdout);
        }
    }
    
    printf("finished 100.00%%\n");
    fclose(fp_record_vz);
    cudaStreamDestroy(stream_co);
    for (int i = 0; i < gm.fine_info.size(); ++i) {
        cudaStreamDestroy(stream_fi[i]);
    }
    return 0;
}

void smooth_fine(GridManager &gm, StreamManager &sm, int time) {
    if (gm.fine_info.size() == 0) return;
    dim3 block(16, 16);
    for (int i = 0; i < gm.fine_info.size(); i++) {
        dim3 grid_fi((gm.fine_info[i].lenx + 15) / 16, (gm.fine_info[i].lenz + 15) / 16);
        smooth_fine_vx<<<grid_fi, block, 0, sm.stream_vx>>>(gm.core_d.vx, gm.core_temp.vx, time, i, 3);
        smooth_fine_vz<<<grid_fi, block, 0, sm.stream_vz>>>(gm.core_d.vz, gm.core_temp.vz, time, i, 3);
        smooth_fine_txz<<<grid_fi, block, 0, sm.stream_txz>>>(gm.core_d.txz, gm.core_temp.txz, time, i, 3);
        smooth_fine_sig<<<grid_fi, block, 0, sm.stream_sx>>>(gm.core_d.sx, gm.core_temp.sx, time, i, 3);
        smooth_fine_sig<<<grid_fi, block, 0, sm.stream_sz>>>(gm.core_d.sz, gm.core_temp.sz, time, i, 3);
        smooth_fine_p<<<grid_fi, block, 0, sm.stream_p>>>(gm.core_d.p, gm.core_temp.p, time, i, 3);
        smooth_fine_rx<<<grid_fi, block, 0, sm.stream_rx>>>(gm.core_d.rx, gm.core_temp.rx, time, i, 3);
        smooth_fine_rz<<<grid_fi, block, 0, sm.stream_rz>>>(gm.core_d.rz, gm.core_temp.rz, time, i, 3);
        smooth_fine_rxz<<<grid_fi, block, 0, sm.stream_rxz>>>(gm.core_d.rxz, gm.core_temp.rxz, time, i, 3);
    }

    int bytes_vx = (gm.offset_time_vx - gm.offset_coarse_vx) * sizeof(float);
    int bytes_vz = (gm.offset_time_vz - gm.offset_coarse_vz) * sizeof(float);
    int bytes_sig = (gm.offset_time_sig - gm.offset_coarse_sig) * sizeof(float);
    int bytes_txz = (gm.offset_time_txz - gm.offset_coarse_txz) * sizeof(float);
    cudaMemcpyAsync(gm.core_d.vx + time * bytes_vx + gm.offset_coarse_vx, gm.core_temp.vx, bytes_vx, cudaMemcpyDeviceToDevice, sm.stream_vx);
    cudaMemcpyAsync(gm.core_d.vz + time * bytes_vz + gm.offset_coarse_vz, gm.core_temp.vz, bytes_vz, cudaMemcpyDeviceToDevice, sm.stream_vz);
    cudaMemcpyAsync(gm.core_d.sx + time * bytes_sig + gm.offset_coarse_sig, gm.core_temp.sx, bytes_sig, cudaMemcpyDeviceToDevice, sm.stream_sx);
    cudaMemcpyAsync(gm.core_d.sz + time * bytes_sig + gm.offset_coarse_sig, gm.core_temp.sz, bytes_sig, cudaMemcpyDeviceToDevice, sm.stream_sz);
    cudaMemcpyAsync(gm.core_d.txz + time * bytes_txz + gm.offset_coarse_txz, gm.core_temp.txz, bytes_txz, cudaMemcpyDeviceToDevice, sm.stream_txz);
    cudaMemcpyAsync(gm.core_d.p + time * bytes_sig + gm.offset_coarse_sig, gm.core_temp.p, bytes_sig, cudaMemcpyDeviceToDevice, sm.stream_p);
    cudaMemcpyAsync(gm.core_d.rx + time * bytes_sig + gm.offset_coarse_sig, gm.core_temp.rx, bytes_sig, cudaMemcpyDeviceToDevice, sm.stream_rx);
    cudaMemcpyAsync(gm.core_d.rz + time * bytes_sig + gm.offset_coarse_sig, gm.core_temp.rz, bytes_sig, cudaMemcpyDeviceToDevice, sm.stream_rz);
    cudaMemcpyAsync(gm.core_d.rxz + time * bytes_txz + gm.offset_coarse_txz, gm.core_temp.rxz, bytes_txz, cudaMemcpyDeviceToDevice, sm.stream_rxz);
    cudaStreamSynchronize(sm.stream_vx);
    cudaStreamSynchronize(sm.stream_vz);
    cudaStreamSynchronize(sm.stream_sx);
    cudaStreamSynchronize(sm.stream_sz);
    cudaStreamSynchronize(sm.stream_txz);
    cudaStreamSynchronize(sm.stream_p);
    cudaStreamSynchronize(sm.stream_rx);
    cudaStreamSynchronize(sm.stream_rz);
    cudaStreamSynchronize(sm.stream_rxz);
}

void output_snapshots(GridManager &gm, int it, float dt, int time) {
    float time_sec = it * dt;
    int time_ms = static_cast<int>(time_sec * 1000);
    
    static char buf[32];

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
    fwrite(gm.core_h.sx, sizeof(float), gm.offset_time_sig, fp_sx);
    fwrite(gm.core_h.sz, sizeof(float), gm.offset_time_sig, fp_sz);
    fwrite(gm.core_h.txz, sizeof(float), gm.offset_time_txz, fp_txz);

    fclose(fp_vx);
    fclose(fp_vz);
    fclose(fp_sx);
    fclose(fp_sz);
    fclose(fp_txz);
}

void output_record(const GridManager &gm, int z, FILE *fp_vz, int time) {
    cudaMemcpy(
        buffer, 
        gm.core_d.vz + time * gm.offset_time_vz + z * gm.nx_coarse, 
        sizeof(float) * gm.nx_coarse, 
        cudaMemcpyDeviceToHost
    );
    fwrite(buffer, sizeof(float), gm.nx_coarse, fp_vz);
}

bool clear_folder(const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    if (ec) {
        return 0;
    }
    return fs::create_directory(dir, ec);
}