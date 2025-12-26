#include "common.cuh"
#include "cpml.cuh"
#include "update.cuh"
#include "output.cuh"
#include "differentiate.cuh"
#include <memory>
const float gain = 1e6;

int main() {
    Params par;
    par.read("models/params.txt");
    
    Grid_Model gm_device(par.nx, par.nz);
    gm_device.read({
        "models/vp.bin", 
        "models/vs.bin", 
        "models/rho.bin",
        "models/C11.bin", 
        "models/C13.bin", 
        "models/C33.bin", 
        "models/C55.bin"
    });
    
    Grid_Core gc_readin(par.nx, par.nz, HOST_MEM);
    Cpml cpml(par.view(), "models/cpml.txt");
    
    printf("网格尺寸: %d x %d\n", par.nx, par.nz);
    printf("空间步长: dx = %f, dz = %f\n", par.dx, par.dz);
    printf("时间步长: dt = %f\n", par.dt);
    printf("总时间步: %d\n", par.nt);
    printf("震源频率: %f Hz\n", par.fpeak);
    printf("震源位置: (%d, %d)\n", par.posx, par.posz);

    // 检查CFL条件
    float dt_max = 0.5f * std::min(par.dx, par.dz) / cpml.cp_max;
    printf("CFL: 最大dt = %f, 实际dt = %f\n", dt_max, par.dt);
    if (par.dt > dt_max) {
        printf("不符合CFL条件\n");
        exit(1);
    }

    // 检查PPW条件
    float ppw = 1900.0f / (2.1 * par.fpeak * par.dx);
    printf("PPW: 最小ppw = 7, 实际ppw = %f\n", ppw);
    
    // 初始化核心计算网格
    Grid_Core gc_host(par.nx, par.nz, HOST_MEM);
    Grid_Core gc_device(par.nx, par.nz, DEVICE_MEM);

    // 生成雷克子波
    std::unique_ptr<float[]> wl = ricker_wavelet(par.nt, par.dt, par.fpeak);
    
    Snapshot sshot(gc_host);
    for (int it = 0; it < par.nt; it++) {
        dim3 gridSize((par.nx + 15) / 16, (par.nz + 15) / 16);
        dim3 blockSize(16, 16);

        int cur = it & 1;
        int pre = cur ^ 1;

        // 应力更新
        update_stress<<<gridSize, blockSize>>>(
            gc_device.view(), gm_device.view(), cpml.view(), 
            par.dx, par.dz, par.dt, cur, pre
        );

        // 加入震源
        apply_source<<<1, 1>>>(
            gc_device.view(), wl[it] * gain, par.posx, par.posz, cur
        );
        cudaDeviceSynchronize();
        
        // ψ_stress 更新
        cpml_update_psi_stress<<<gridSize, blockSize>>>(
            gc_device.view(), cpml.view(), 
            par.dx, par.dz, par.dt, cur
        );
        cudaDeviceSynchronize();

        // 速度更新
        update_velocity<<<gridSize, blockSize>>>(
            gc_device.view(), gm_device.view(), cpml.view(), 
            par.dx, par.dz, par.dt, cur, pre
        );
        cudaDeviceSynchronize();
        
        // ψ_vel 更新
        cpml_update_psi_vel<<<gridSize, blockSize>>>(
            gc_device.view(), cpml.view(), 
            par.dx, par.dz, par.dt, cur
        );

        // 自由边界
        apply_free_boundary<<<1, std::max(par.nx, par.nz)>>>(gc_device.view(), cur);
        cudaDeviceSynchronize();
        
        if (it % 100 == 0) {
            printf("\r%%%0.2f finished.", 1.0f * it / par.nt * 100);
            fflush(stdout);
        }

        // 输出波场快照
        if (it % par.snapshot == 0) {
            // 拷贝到 host 输出波场快照
            gc_host.memcpy_to_host_from(gc_device);
            sshot.output(it, par.dt, cur);
        }
    }
    printf("\r%%100.00 finished.\n");
    fflush(stdout);
}