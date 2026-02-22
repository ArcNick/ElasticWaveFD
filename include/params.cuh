#ifndef PARAMS_CUH
#define PARAMS_CUH

#include <string>
#include <memory>

class Params {
public:
    float fpeak;  
    float dt;
    int nt;
    int posx;
    int posz;
    int snapshot;

    Params(const std::string &file);
    ~Params() = default;

    std::unique_ptr<float[]> ricker_wavelet();

private:
    void read(const std::string &file);
    void build_constant();
};

extern __constant__ float dt_d;
extern __constant__ int nt_d;
extern __constant__ int posx_d;
extern __constant__ int posz_d;

#endif