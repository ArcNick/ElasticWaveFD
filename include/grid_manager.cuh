#ifndef GRID_MANAGER_CUH
#define GRID_MANAGER_CUH

#include <string>
#include <vector>
#include <cuda_runtime.h>

#define LUT_SIZE 1000
#define LAGRANGE_ORDER 8

enum FINE_FLAG { FINE_OFF = 0, FINE_ON = 1 };
enum MAT_FLAG { SOLID = 0, FLUID = 1 };

struct FD_n {
    int n_x;
    int n_z;
};

struct Mask {
    int *vx;
    int *vz;
    int *sig;
    int *txz;
};

struct Core {
    float *vx;
    float *vz;
    float *p;
    float *sx;
    float *sz;
    float *txz;
    float *r;
};

struct Model {
    float *rho = nullptr;
    float *C11 = nullptr;
    float *C13 = nullptr;
    float *C33 = nullptr;
    float *C55 = nullptr;
    float *tau = nullptr;
    float *inv_taus = nullptr;
    MAT_FLAG *mat = nullptr;
};

struct FineInfo {
    int x_start;
    int x_end;
    int z_start;
    int z_end;
    int lenx;
    int lenz;
    int N;
    float dx_fine;
    float dz_fine;
};

class GridManager {
public:
    int nx_coarse;
    int nz_coarse;
    float dx_coarse;
    float dz_coarse;

    int offset_time_vx;
    int offset_time_vz;
    int offset_time_sig;
    int offset_time_txz;

    int offset_coarse_vx;
    int offset_coarse_vz;
    int offset_coarse_sig;
    int offset_coarse_txz;

    Mask core_mask;
    Core core_h, core_d, core_temp;
    Model model_h, model_d;

    FD_n *vx_n, *vz_n, *sig_n, *txz_n;

    std::vector<FineInfo> fine_info;

    GridManager(const std::string &file);
    ~GridManager();

    void memcpy_model_h2d();
    void memcpy_core_d2h(int time);

    // 用于保存 n 数组纹理对象
    cudaTextureObject_t tex_vx_n;
    cudaTextureObject_t tex_vz_n;
    cudaTextureObject_t tex_sig_n;
    cudaTextureObject_t tex_txz_n;

    // 用于保存 mask 纹理对象
    cudaTextureObject_t tex_vx_mask;
    cudaTextureObject_t tex_vz_mask;
    cudaTextureObject_t tex_sig_mask;
    cudaTextureObject_t tex_txz_mask;

    cudaTextureObject_t tex_mat;

private:
    FINE_FLAG FINE;

    void load_from_file(const std::string &file);
    void memory_allocate();
    void memory_release();
    void build_texture();
    void build_n();
    void build_constant();
    void build_insterp_LUT();
};

// 设备端全局符号声明
extern __device__ cudaTextureObject_t vx_mask;
extern __device__ cudaTextureObject_t vz_mask;
extern __device__ cudaTextureObject_t sig_mask;
extern __device__ cudaTextureObject_t txz_mask;
extern __device__ cudaTextureObject_t mat_tex;

extern __device__ cudaTextureObject_t vx_n_tex;
extern __device__ cudaTextureObject_t vz_n_tex;
extern __device__ cudaTextureObject_t sig_n_tex;
extern __device__ cudaTextureObject_t txz_n_tex;

extern __constant__ float dx, dz;
extern __constant__ int nx, nz;
extern __constant__ int offset_vx_all;
extern __constant__ int offset_vz_all;
extern __constant__ int offset_sig_all;
extern __constant__ int offset_txz_all;

extern float dx_host_coarse, dz_host_coarse;
extern int nx_host_coarse, nz_host_coarse;

extern __constant__ int num_fine;
extern __constant__ FineInfo fines[12];
extern __constant__ int sum_offset_fine_vx[12];
extern __constant__ int sum_offset_fine_vz[12];
extern __constant__ int sum_offset_fine_sig[12];
extern __constant__ int sum_offset_fine_txz[12];

extern __constant__ float lagrange_coeff[LUT_SIZE * LAGRANGE_ORDER];

#endif