#ifndef TRIANGLE_QUERY_H
#define TRIANGLE_QUERY_H

#include <vector>
#include <cstdint>
#include <iostream>

namespace surfacemesh_compare
{
    typedef int labeltype;

    bool identical_triangle(const std::vector<int64_t> &A, const std::vector<int64_t> &B);

    std::vector<std::vector<int64_t>> extract_unique_triangles(std::vector<std::vector<int64_t>> &tetrahedra);

    std::vector<std::vector<int64_t>> extract_exterior_surface(std::vector<std::vector<int64_t>> &tetrahedra);
    std::vector<std::vector<int64_t>> extract_exterior_surface_normalized(std::vector<std::vector<float>> &vertices, std::vector<std::vector<int64_t>> &tetrahedra);
    std::vector<std::vector<int64_t>> extract_exterior_surface_normalized(std::vector<std::vector<float>> &vertices, std::vector<std::vector<int64_t>> &tetrahedra, std::vector<int64_t> &cell_id);
    std::vector<std::vector<int64_t>> extract_exterior_surface_normalized(std::vector<std::vector<float>> &vertices, std::vector<std::vector<int64_t>> &tetrahedra, std::vector<labeltype> &labels, labeltype valid_label, bool labeled = false);
    std::vector<std::vector<int64_t>> extract_exterior_surface_indexed(std::vector<std::vector<int64_t>> &tetrahedra);
    std::vector<std::vector<int64_t>> extract_interfaces_and_reorder_for_abaqus(std::vector<std::vector<int64_t>> &tetrahedra, std::vector<labeltype> &labels);

    float relabel_debugging(std::vector<std::vector<int64_t>> &tetrahedra, std::vector<int> &labels,  int label2relabel, int interface_label, std::vector<float> &edm_screw);

    int64_t remove_disconnected_bone(labeltype label_screw, std::vector<std::vector<int64_t>> &tetrahedra, std::vector<labeltype> &labels);
    int64_t reduce2dirichletboundaries(std::vector<labeltype> label_colors, std::vector<std::vector<int64_t>> &tetrahedra, std::vector<labeltype> &labels);
}

#endif //TRIANGLE_QUERY

