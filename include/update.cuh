#ifndef UPDATE_CUH
#define UPDATE_CUH

#include "grid_core.cuh"
#include "grid_model.cuh"
#include "cpml.cuh"

__global__ void apply_source(
    Grid_Core::View gc, float src, int posx, int posz, int cur
);

__global__ void update_stress(
    Grid_Core::View gc, Grid_Model::View gm, 
    Cpml::View pml, float dx, float dz, 
    float dt, int cur, int pre
);

__global__ void update_velocity(
    Grid_Core::View gc, Grid_Model::View gm, 
    Cpml::View pml, float dx, float dz, 
    float dt, int cur, int pre
);

__global__ void apply_free_boundary(Grid_Core::View gc, int cur);

#endif