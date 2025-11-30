#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>

float a[500][500];

std::unordered_map<std::string, std::pair<float, float>> params;
int main() {
    int nz = 500, nx = 500;

    params["vp0"] = {4000, 4000};
    params["vs0"] = {2100, 2100};
    params["rho"] = {2400, 2400};
    params["epsilon"] = {0, 0};
    params["delta"] = {0, 0};
    params["gamma"] = {0, 0};

    // ======密度ρ=====
    FILE *fp = fopen("rho.bin", "wb");
    for (int i = 0; i < nz; i++) {
        for (int j = 0; j < nx; j++) {
            if (i < nz / 2) {
                a[i][j] = params["rho"].first;
            } else {
                a[i][j] = params["rho"].second;
            }
        }
    }
    for (int i = 0; i < nz; i++) {
        fwrite(a[i], sizeof(float), nx, fp);
    }
    
    // ======波速vp=====
    fp = fopen("vp.bin", "wb");
    for (int i = 0; i < nz; i++) {
        for (int j = 0; j < nx; j++) {
            if (i < nz / 2) {
                a[i][j] = params["vp0"].first;
            } else {
                a[i][j] = params["vp0"].second;
            }
        }
    }
    for (int i = 0; i < nz; i++) {
        fwrite(a[i], sizeof(float), nx, fp);
    }
    
    // ======波速vs=====
    fp = fopen("vs.bin", "wb");
    for (int i = 0; i < nz; i++) {
        for (int j = 0; j < nx; j++) {
            if (i < nz / 2) {
                a[i][j] = params["vs0"].first;
            } else {
                a[i][j] = params["vs0"].second;
            }
        }
    }
    for (int i = 0; i < nz; i++) {
        fwrite(a[i], sizeof(float), nx, fp);
    }

    // ======γ=====
    // fp = fopen("gamma.bin", "wb");
    // for (int i = 0; i < nz; i++) {
    //     for (int j = 0; j < nx; j++) {
    //         if (i < nz / 2) {
    //             a[i][j] = params["gamma"].first;
    //         } else {
    //             a[i][j] = params["gamma"].second;
    //         }
    //     }
    // }
    // for (int i = 0; i < nz; i++) {
    //     fwrite(a[i], sizeof(float), nx, fp);
    // }

    // // ======ε=====
    // fp = fopen("epsilon.bin", "wb");
    // for (int i = 0; i < nz; i++) {
    //     for (int j = 0; j < nx; j++) {
    //         if (i < nz / 2) {
    //             a[i][j] = params["epsilon"].first;
    //         } else {
    //             a[i][j] = params["epsilon"].second;
    //         }
    //     }
    // }
    // for (int i = 0; i < nz; i++) {
    //     fwrite(a[i], sizeof(float), nx, fp);
    // }

    // // ======δ=====
    // fp = fopen("delta.bin", "wb");
    // for (int i = 0; i < nz; i++) {
    //     for (int j = 0; j < nx; j++) {
    //         if (i < nz / 2) {
    //             a[i][j] = params["delta"].first;
    //         } else {
    //             a[i][j] = params["delta"].second;
    //         }
    //     }
    // }
    // for (int i = 0; i < nz; i++) {
    //     fwrite(a[i], sizeof(float), nx, fp);
    // }

    fclose(fp);
}
