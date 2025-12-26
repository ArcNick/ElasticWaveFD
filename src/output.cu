#include "output.cuh"
#include <cstdlib>
#include <sstream>
#include <string>

Array_Info::Array_Info(
    float* const d[2], int lx, int lz, const std::string &n
) : lenx(lx), lenz(lz), name(n) {
    data[0] = d[0];
    data[1] = d[1];    
}

Snapshot::Snapshot(const Grid_Core &g) {
    nx = g.nx;
    nz = g.nz;
    arrays.emplace_back(g.sx, nx, nz, std::string("sx"));
    arrays.emplace_back(g.sz, nx, nz, std::string("sz")); 
    arrays.emplace_back(g.txz, nx - 1, nz - 1, std::string("txz"));
    arrays.emplace_back(g.vx, nx - 1, nz, std::string("vx"));
    arrays.emplace_back(g.vz, nx, nz - 1, std::string("vz"));

    output_dir = fs::current_path() / "output";
}

void Snapshot::output(int it, float dt, int cur) {
    for (const auto& info : arrays) {
        fs::path full_path = output_dir / info.name;
        if (!fs::exists(full_path)) {
            fs::create_directories(full_path);
        }

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << it * dt * 1000;

        std::string filename = info.name + "_" + oss.str() + "ms.bin";
        fs::path file_path = full_path / filename;
        
        FILE *fp = fopen(file_path.string().c_str(), "wb");
        for (int i = 0; i < info.lenz; i++) {
            fwrite(&info.data[cur][i * info.lenx], sizeof(float), info.lenx, fp);
        }
        fclose(fp);
    }
}