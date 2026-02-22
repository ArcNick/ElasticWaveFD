#ifndef CPML_CUH
#define CPML_CUH

#include <string>

#define CPMLMAX 40

struct Params;

struct PsiVel {
    float *psi_vx_x = nullptr;
    float *psi_vx_z = nullptr;
    float *psi_vz_x = nullptr;
    float *psi_vz_z = nullptr;
};

struct PsiStr {
    float *psi_sx_x = nullptr;
    float *psi_sx_z = nullptr;
    float *psi_sz_x = nullptr;
    float *psi_sz_z = nullptr;
    float *psi_txz_x = nullptr;
    float *psi_txz_z = nullptr;
};

class Cpml {
public:
    int thickness;                  // CPML层厚度（网格点数）
    float NPOW;                     // 阻尼剖面指数
    float cp_max;                   // 最大纵波波速
    float L;                        // CPML层厚度
    float Rc;                       // 反射系数
    float damp0;                    // 最大阻尼系数
    float alpha0;                   // 最大频移因子
    float kappa0;                   // 最大拉伸因子

    float *alpha_int = nullptr;  // 整数网格点的频移因子
    float *alpha_half = nullptr;
    float *damp_int = nullptr;
    float *damp_half = nullptr;
    float *a_int = nullptr;
    float *a_half = nullptr;
    float *b_int = nullptr;
    float *b_half = nullptr;
    float *kappa_int = nullptr;
    float *kappa_half = nullptr;

    PsiStr psi_str;
    PsiVel psi_vel;

    Cpml(const std::string &file);
    ~Cpml();
private:
    void load(const std::string &file);
    void mem_allocate(int lx, int lz);
    void build_constant();
    void mem_release();
};

extern __constant__ int thickness_d;
extern __constant__ float a_int_d[CPMLMAX];
extern __constant__ float a_half_d[CPMLMAX];
extern __constant__ float b_int_d[CPMLMAX];
extern __constant__ float b_half_d[CPMLMAX];
extern __constant__ float kappa_int_d[CPMLMAX];
extern __constant__ float kappa_half_d[CPMLMAX];

#endif