#ifndef MESHIO_H
#define MESHIO_H

#include <vector>

namespace meshio
{
    typedef float vertextype;
    typedef int64_t celltype;
    typedef int labeltype_discrete;
    typedef float labeltype_floating;

    struct Mesh
    {
        std::string type = "volume";

        //geometric information
        int64_t n_tetrahedra = 0;
        int64_t n_vertices = 0;
        int64_t n_triangles = 0;
        std::vector<std::vector<vertextype>> vertices;
        std::vector<std::vector<celltype>> triangles, tetrahedra;

        //color information
        std::vector<std::string> cell_labels_discrete_name, cell_labels_floating_name, vertex_labels_discrete_name, vertex_labels_floating_name,
            cell_vectors_name, vertex_vectors_name;
        std::vector<std::vector<labeltype_discrete>> cell_labels_discrete, vertex_labels_discrete;
        std::vector<std::vector<labeltype_floating>> cell_labels_floating, vertex_labels_floating;
        std::vector<std::vector<std::vector<labeltype_floating>>> cell_vectors, vertex_vectors;
    };

    void read_shapefile(std::string filename, int shape[3], std::string sample_ID);

    //Surface Meshes
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void write_off(std::string path, std::string name, std::vector<std::vector<float>> &points, std::vector<std::vector<int64_t>> &triangles);
    void write_colored_off(std::string path, std::string name, std::vector<std::vector<float>> &points, std::vector<std::vector<int64_t>> &triangles,
                        std::vector<int64_t> &shared_triangles, int color_unique[3], int color_shared[3]);
    void write_colored_off(std::string path, std::string name, std::vector<std::vector<float>> &points, std::vector<std::vector<int64_t>> &triangles,
                        std::vector<int> &threePhaseLabels, int color0[3], int color1[3], int color2[3]);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //Old functions
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Mesh read_off(std::string filename);
    int64_t write_subset_off(int32_t active_label, std::string path, std::string name, std::vector<int32_t> &labels, std::vector<std::vector<float>> &points, std::vector<std::vector<int64_t>> &triangles);
    void merge_meshfiles(std::string outpath, std::string outfilename, std::vector<std::string> meshfiles, bool save_triangles=true);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //Volume Meshes
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Mesh read_meditmesh_labeled(std::string filename);
    Mesh read_unlabeled_vtk_mesh(std::string filename);

    void read_geometry_off_vtk_volumemesh(std::string filename, std::vector<std::vector<float>> &vertices, std::vector<std::vector<int64_t>> &tetrahedra);
    void save_mesh2vtk(std::string path, std::string name, const Mesh &mesh, std::string header = "");

    void read_volumemesh_vtk_floatlabels_cell3dvectors(std::string filename, std::vector<std::vector<float>> &vertices, std::vector<std::vector<int64_t>> &tetrahedra,
        std::vector<float> &labels, std::vector<std::vector<float>> &vectors);
    meshio::Mesh read_comsol_vtu_volmesh(std::string filename, bool merge_velocitycomponents);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //Simplification
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void remove_unusedvertices(std::vector<std::vector<float>> &vertices, std::vector<std::vector<int64_t>> &tetrahedra);
    void remove_unusedvertices_surfacemesh(std::vector<std::vector<float>> &vertices, std::vector<std::vector<int64_t>> &triangles);
    void remove_unusedvertices_surfacemesh(meshio::Mesh &mesh); //incorporates mapped values
    void map_tetlabels2trilabels(Mesh &surfmesh, std::vector<int64_t> &tet_idx);
    void create_subset_from_label(uint8_t active_label, std::vector<std::vector<float>> &vertices, std::vector<std::vector<int64_t>> &tetrahedra, std::vector<uint8_t> &labels,
                                std::vector<std::vector<float>> &out_vertices, std::vector<std::vector<int64_t>> &out_tetrahedra);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //Auxiliary
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T> int count_uniquelabels(std::vector<T> &labels);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}

#endif //MESHIO_H

