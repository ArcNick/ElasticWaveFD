#ifndef PARAMS_CUH
#define PARAMS_CUH

#include <cuda_runtime.h>

class Params {
public:
    float fpeak;        // 雷克子波频率
    int nx, nz;         // 网格尺寸
    float dx, dz;       // 网格步长
    int nt;             // 模拟时间步数
    float dt;           // 模拟时间步长
    int posx, posz;     // 炮点位置
    int snapshot;       // 波场快照间隔
    
    struct View {
        float fpeak;
        int nx, nz;
        float dx, dz;
        int nt;
        float dt;
        int posx, posz;
        int snapshot;
    };
    
    View view() {
        return (View){
            fpeak, nx, nz, dx, dz, nt, dt, posx, posz, snapshot
        };
    }

    Params() : fpeak(0), nx(0), nz(0), dx(0), dz(0), nt(0), dt(0),
               posx(0), posz(0), snapshot(0) {};
    ~Params() = default;
    void read(const char *file);
};

#endif
