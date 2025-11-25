#include "output.cuh"
#include <cstdlib>
#include <sstream>
#include <string>

Snapshot::Snapshot(const Grid_Core &g) {
    nx = g.nx;
    nz = g.nz;
    arrays.emplace_back(g.sx, nx, nz, "sx");
    arrays.emplace_back(g.sz, nx, nz, "sz"); 
    arrays.emplace_back(g.txz, nx - 1, nz - 1, "txz");
    arrays.emplace_back(g.vx, nx - 1, nz, "vx");
    arrays.emplace_back(g.vz, nx, nz - 1, "vz");

    output_dir = fs::current_path() / "output";
}

void Snapshot::output(int it, float dt) {
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
            fwrite(&info.data[i * info.lenx], sizeof(float), info.lenx, fp);
        }
        fclose(fp);
    }
}