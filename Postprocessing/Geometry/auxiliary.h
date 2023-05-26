#ifndef AUXILIARY_H
#define AUXILIARY_H

#include <iostream>
#include <string.h>
#include <cstdint>
#include <vector>

namespace aux
{
    /*String-Manipulation
    *********************************************************/
    std::string zfill_int2string(int inint, const unsigned int &zfill);
    std::string get_active_directory();

    /*Numpy-like
    *********************************************************/
    std::vector<double> linspace(double startval, double endval, uint64_t bins);

    float linearinterpolate_coordinate(float x, float y, float z, float *image, int shape[3]);

    std::vector<std::vector<float>> linearinterpolate_coordinatevector(std::vector<std::vector<float>> &coordinates, float *ux, float *uy, float *uz, int imgshape[3]);
    std::vector<float> linearinterpolate_coordinatevector(std::vector<std::vector<float>> &coordinates, float *scalar, int imgshape[3]);
    std::vector<float> maxval_at_coordinatevector(std::vector<std::vector<float>> &coordinates, float *scalar, int imgshape[3]);
    std::vector<std::vector<float>> linearinterpolate_averagecellvalue(std::vector<std::vector<float>> &coordinates, std::vector<std::vector<int64_t>> &tetrahedra, float *ux, float *uy, float *uz, int imgshape[3]);
    std::vector<float> linearinterpolate_averagecellvalue(std::vector<std::vector<float>> &coordinates, std::vector<std::vector<int64_t>> &tetrahedra, float *scalar, int imgshape[3]);
    std::vector<std::vector<float>> get_cell_centerofgravity(std::vector<std::vector<float>> &coordinates, std::vector<std::vector<int64_t>> &tetrahedra);
    std::vector<std::vector<float>> get_triangle_centerofgravity(std::vector<std::vector<float>> &coordinates, std::vector<std::vector<int64_t>> &triangles);
    std::vector<std::vector<float>> get_triangle_centerofgravity_minzpos(std::vector<std::vector<float>> &coordinates, std::vector<std::vector<int64_t>> &triangles, float min_zpos);

    std::vector<std::vector<float>> get_inspheres(std::vector<std::vector<float>> &coordinates, std::vector<std::vector<int64_t>> &tetrahedra);
    std::vector<std::vector<float>> vertex_subset(int valid_label, std::vector<std::vector<float>> &vertices, std::vector<std::vector<int64_t>> &tetrahedra, std::vector<int> &celllabel);

    //for ray cartilage
    float* calc_meanzpos_label(uint8_t *labels, int shape[3], float sigma);
    void relabel_as_abovebelow_meanzpos(float* meanzpos, int shape[2], std::vector<int> &mesh_labels, std::vector<std::vector<float>> &center_of_gravity);
    float* get_interface(uint8_t* labelimage, int shape[3]);
}

#endif // AUXILIARY_H
