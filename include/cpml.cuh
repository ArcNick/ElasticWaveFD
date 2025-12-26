#ifndef CPML_CUH
#define CPML_CUH

#include "grid_core.cuh"
#include "params.cuh"

#define PVX_X(ix, iz) (cpml.psi_vx_x[(iz) * (cpml.nx - 1) + (ix)])
#define PVX_Z(ix, iz) (cpml.psi_vx_z[(iz) * (cpml.nx - 1) + (ix)])
#define PVZ_X(ix, iz) (cpml.psi_vz_x[(iz) * cpml.nx + (ix)])
#define PVZ_Z(ix, iz) (cpml.psi_vz_z[(iz) * cpml.nx + (ix)])
#define PSX_X(ix, iz) (cpml.psi_sx_x[(iz) * cpml.nx + (ix)])
#define PSX_Z(ix, iz) (cpml.psi_sx_z[(iz) * cpml.nx + (ix)])
#define PSZ_X(ix, iz) (cpml.psi_sz_x[(iz) * cpml.nx + (ix)])
#define PSZ_Z(ix, iz) (cpml.psi_sz_z[(iz) * cpml.nx + (ix)])
#define PTXZ_X(ix, iz) (cpml.psi_txz_x[(iz) * (cpml.nx - 1) + (ix)])
#define PTXZ_Z(ix, iz) (cpml.psi_txz_z[(iz) * (cpml.nx - 1) + (ix)])

class Cpml {
public:
    int thickness;                  // CPML层厚度（网格点数）
    float N;                        // 阻尼剖面指数
    float cp_max;                   // 最大纵波波速
    float L;                        // CPML层厚度
    float Rc;                       // 反射系数
    float damp0;                    // 最大阻尼系数
    float alpha0;                   // 最大频移因子
    float kappa0;                   // 最大拉伸因子

    float *alpha_int;
    float *alpha_half;
    float *damp_int;
    float *damp_half;
    float *a_int;
    float *a_half;
    float *b_int;
    float *b_half;
    float *kappa_int;
    float *kappa_half;

    float *psi_vx_x, *psi_vx_z;     // 记忆变量
    float *psi_vz_x, *psi_vz_z;
    float *psi_sx_x, *psi_sx_z;
    float *psi_sz_x, *psi_sz_z;
    float *psi_txz_x, *psi_txz_z;

    int nx, nz;
    bool mem_location;

    struct View {
        int thickness;
        float N;
        float cp_max;
        float L;
        float Rc;
        float damp0;
        float alpha0;
        float kappa0;

        float *alpha_int;
        float *alpha_half;
        float *damp_int;
        float *damp_half;
        float *a_int;
        float *a_half;
        float *b_int;
        float *b_half;
        float *kappa_int;
        float *kappa_half;

        float *psi_vx_x, *psi_vx_z;
        float *psi_vz_x, *psi_vz_z;
        float *psi_sx_x, *psi_sx_z;
        float *psi_sz_x, *psi_sz_z;
        float *psi_txz_x, *psi_txz_z;

        int nx, nz;
        bool mem_location;
    };

    View view() {
        return (View){
            thickness, N, cp_max, L, Rc, 
            damp0, alpha0, kappa0,
            alpha_int, alpha_half, 
            damp_int, damp_half, 
            a_int, a_half, b_int, b_half, 
            kappa_int, kappa_half,
            psi_vx_x, psi_vx_z,
            psi_vz_x, psi_vz_z,
            psi_sx_x, psi_sx_z,
            psi_sz_x, psi_sz_z,
            psi_txz_x, psi_txz_z,
            nx, nz, mem_location
        };
    }

    Cpml(Params::View par, const char *file);
    ~Cpml();

private:
    void read(const char *file);
    void init(Params::View par);
};

__global__ void cpml_init_params(
    Cpml::View cpml, float dx, float dz, float dt
);

__device__ int get_cpml_idx_x_int(int lx, int ix, int thickness);
__device__ int get_cpml_idx_z_int(int lz, int iz, int thickness);
__device__ int get_cpml_idx_x_half(int lx, int ix, int thickness);
__device__ int get_cpml_idx_z_half(int lz, int iz, int thickness);

__global__ void cpml_update_psi_vel(
    Grid_Core::View gc, Cpml::View cpml, 
    float dx, float dz, float dt, int cur
);

__global__ void cpml_update_psi_stress(
    Grid_Core::View gc, Cpml::View cpml, 
    float dx, float dz, float dt, int cur
);

#endif
