#include "triangles.h"
#include <algorithm>

namespace surfacemesh_compare
{
    bool identical_triangle(const std::vector<int64_t> &A, const std::vector<int64_t> &B)
    {
        if (A[0] == B[0] && A[1] == B[1] && A[2] == B[2]) return true;
        if (A[0] == B[0] && A[1] == B[2] && A[2] == B[1]) return true;
        if (A[0] == B[1] && A[1] == B[0] && A[2] == B[2]) return true;
        if (A[0] == B[1] && A[1] == B[2] && A[2] == B[0]) return true;
        if (A[0] == B[2] && A[1] == B[0] && A[2] == B[1]) return true;
        if (A[0] == B[2] && A[1] == B[1] && A[2] == B[0]) return true;

        return false;
    }
    struct sort_triangles
    {
        inline bool operator() (const std::vector<int64_t>&A, const std::vector<int64_t>&B)
        {
            if (A[0] < B[0]) return true;
            else if (A[0] == B[0])
            {
                if (A[1] < B[1]) return true;
                else if (A[1] == B[1] && A[2] < B[2]) return true;
            }

            return false;
        }
    };

    std::vector<std::vector<int64_t>> extract_unique_triangles(std::vector<std::vector<int64_t>> &tetrahedra)
    {
        std::vector<std::vector<int64_t>> triangles, unique_triangles;

        for (uint64_t idx = 0; idx < tetrahedra.size(); idx++)
        {
            std::vector<int64_t> active_tet = tetrahedra[idx];

            std::sort(active_tet.begin(), active_tet.end());

            std::vector<int64_t> triang0 = {active_tet[0], active_tet[1], active_tet[2]};
            std::vector<int64_t> triang1 = {active_tet[0], active_tet[1], active_tet[3]};
            std::vector<int64_t> triang2 = {active_tet[0], active_tet[2], active_tet[3]};
            std::vector<int64_t> triang3 = {active_tet[1], active_tet[2], active_tet[3]};

            triangles.push_back(triang0);
            triangles.push_back(triang1);
            triangles.push_back(triang2);
            triangles.push_back(triang3);
        }

        //sort triangles
        std::sort(triangles.begin(), triangles.end(), sort_triangles());
        for (uint64_t idx = 0; idx < triangles.size(); idx++)
        {
            unique_triangles.push_back(triangles[idx]);

            if (idx == triangles.size()-1)
                break;

            if(identical_triangle(triangles[idx], triangles[idx+1]))
                idx++;
        }

        return unique_triangles;
    }

    std::vector<std::vector<int64_t>> extract_exterior_surface(std::vector<std::vector<int64_t>> &tetrahedra)
    {
        std::vector<std::vector<int64_t>> triangles, exterior_surface;

        for (uint64_t idx = 0; idx < tetrahedra.size(); idx++)
        {
            std::vector<int64_t> active_tet = tetrahedra[idx];

            std::sort(active_tet.begin(), active_tet.end());

            std::vector<int64_t> triang0 = {active_tet[0], active_tet[1], active_tet[2]};
            std::vector<int64_t> triang1 = {active_tet[0], active_tet[1], active_tet[3]};
            std::vector<int64_t> triang2 = {active_tet[0], active_tet[2], active_tet[3]};
            std::vector<int64_t> triang3 = {active_tet[1], active_tet[2], active_tet[3]};

            triangles.push_back(triang0);
            triangles.push_back(triang1);
            triangles.push_back(triang2);
            triangles.push_back(triang3);
        }

        //sort triangles
        std::sort(triangles.begin(), triangles.end(), sort_triangles());
        for (uint64_t idx = 0; idx < triangles.size(); idx++)
        {
            if (idx == triangles.size()-1){
                exterior_surface.push_back(triangles[idx]);
                break;
            }

            if(identical_triangle(triangles[idx], triangles[idx+1]))
                idx++;
            else
                exterior_surface.push_back(triangles[idx]);
        }

        return exterior_surface;
    }
    std::vector<std::vector<int64_t>> extract_exterior_surface_indexed(std::vector<std::vector<int64_t>> &tetrahedra)
    {
        std::vector<std::vector<int64_t>> triangles, exterior_surface;

        for (int64_t idx = 0; idx < tetrahedra.size(); idx++)
        {
            std::vector<int64_t> active_tet = tetrahedra[idx];

            std::sort(active_tet.begin(), active_tet.end());

            std::vector<int64_t> triang0 = {active_tet[0], active_tet[1], active_tet[2], idx};
            std::vector<int64_t> triang1 = {active_tet[0], active_tet[1], active_tet[3], idx};
            std::vector<int64_t> triang2 = {active_tet[0], active_tet[2], active_tet[3], idx};
            std::vector<int64_t> triang3 = {active_tet[1], active_tet[2], active_tet[3], idx};

            triangles.push_back(triang0);
            triangles.push_back(triang1);
            triangles.push_back(triang2);
            triangles.push_back(triang3);
        }

        //sort triangles
        std::sort(triangles.begin(), triangles.end(), sort_triangles());
        for (uint64_t idx = 0; idx < (uint64_t) triangles.size(); idx++)
        {
            if (idx == triangles.size()-1){
                exterior_surface.push_back(triangles[idx]);
                break;
            }

            if(identical_triangle(triangles[idx], triangles[idx+1]))
                idx++;
            else
                exterior_surface.push_back(triangles[idx]);
        }

        return exterior_surface;
    }
    std::vector<std::vector<int64_t>> extract_exterior_surface_normalized(std::vector<std::vector<float>> &vertices, std::vector<std::vector<int64_t>> &tetrahedra)
    {
        std::vector<std::vector<int64_t>> triangles, exterior_surface;

        for (int64_t idx = 0; idx < (int64_t) tetrahedra.size(); idx++)
        {
            std::vector<int64_t> active_tet = tetrahedra[idx];

            std::sort(active_tet.begin(), active_tet.end());

            std::vector<int64_t> triang0 = {active_tet[0], active_tet[1], active_tet[2], idx};
            std::vector<int64_t> triang1 = {active_tet[0], active_tet[1], active_tet[3], idx};
            std::vector<int64_t> triang2 = {active_tet[0], active_tet[2], active_tet[3], idx};
            std::vector<int64_t> triang3 = {active_tet[1], active_tet[2], active_tet[3], idx};

            triangles.push_back(triang0);
            triangles.push_back(triang1);
            triangles.push_back(triang2);
            triangles.push_back(triang3);
        }

        //sort triangles
        std::sort(triangles.begin(), triangles.end(), sort_triangles());
        for (uint64_t idx = 0; idx < triangles.size(); idx++)
        {
            if (idx == triangles.size()-1){
                exterior_surface.push_back(triangles[idx]);
                break;
            }

            if(identical_triangle(triangles[idx], triangles[idx+1]))
                idx++;
            else
                exterior_surface.push_back(triangles[idx]);
        }


        //reorder the surface triangles to point outwards
        for (uint64_t idx = 0; idx < exterior_surface.size(); idx++)
        {
            std::vector<int64_t> active_tet = tetrahedra[exterior_surface[idx][3]];

            std::sort(active_tet.begin(), active_tet.end());

            std::vector<std::vector<int64_t>> triang = {{active_tet[0], active_tet[1], active_tet[2]},{active_tet[0], active_tet[1], active_tet[3]},
                {active_tet[0], active_tet[2], active_tet[3]},{active_tet[1], active_tet[2], active_tet[3]}};

            int inward_point = 0;
            int tri = 0; //the triangle

            //record inward point and the triangle to be reordered
            if(identical_triangle(triang[0], exterior_surface[idx])) inward_point = 3;
            else if(identical_triangle(triang[1], exterior_surface[idx])) {inward_point = 2; tri = 1;}
            else if(identical_triangle(triang[2], exterior_surface[idx])) {inward_point = 1; tri = 2;}
            else if(identical_triangle(triang[3], exterior_surface[idx])) {inward_point = 0; tri = 3;}

            std::vector<float> Pe = vertices[active_tet[inward_point]];
            std::vector<float> P1 = vertices[triang[tri][0]];
            std::vector<float> P2 = vertices[triang[tri][1]];
            std::vector<float> P3 = vertices[triang[tri][2]];

            std::vector<float> normal1 = {(P2[1]-P1[1])*(P3[2]-P1[2])-(P3[1]-P1[1])*(P2[2]-P1[2]),
                                          (P2[2]-P1[2])*(P3[0]-P1[0])-(P2[0]-P1[0])*(P3[2]-P1[2]),
                                          (P2[0]-P1[0])*(P3[1]-P1[1])-(P3[0]-P1[0])*(P2[1]-P1[1])};
            std::vector<float> normal2 = {(P2[1]-P3[1])*(P1[2]-P3[2])-(P1[1]-P3[1])*(P2[2]-P3[2]),
                                          (P2[2]-P3[2])*(P1[0]-P3[0])-(P2[0]-P3[0])*(P1[2]-P3[2]),
                                          (P2[0]-P3[0])*(P1[1]-P3[1])-(P1[0]-P3[0])*(P2[1]-P3[1])};

            float dist1 = (Pe[0]-(P1[0]+normal1[0]))*(Pe[0]-(P1[0]+normal1[0]))+(Pe[1]-(P1[1]+normal1[1]))*(Pe[1]-(P1[1]+normal1[1]))+(Pe[2]-(P1[2]+normal1[2]))*(Pe[2]-(P1[2]+normal1[2]));
            float dist2 = (Pe[0]-(P1[0]+normal2[0]))*(Pe[0]-(P1[0]+normal2[0]))+(Pe[1]-(P1[1]+normal2[1]))*(Pe[1]-(P1[1]+normal2[1]))+(Pe[2]-(P1[2]+normal2[2]))*(Pe[2]-(P1[2]+normal2[2]));

            if (dist1 >= dist2) exterior_surface[idx] = {triang[tri][0], triang[tri][1], triang[tri][2]};
            else exterior_surface[idx] = {triang[tri][2], triang[tri][1], triang[tri][0]};
        }

        return exterior_surface;
    }
    std::vector<std::vector<int64_t>> extract_exterior_surface_normalized(std::vector<std::vector<float>> &vertices, std::vector<std::vector<int64_t>> &tetrahedra, std::vector<int64_t> &cell_id)
    {
        std::vector<std::vector<int64_t>> triangles, exterior_surface;
        cell_id.clear();

        for (int64_t idx = 0; idx < (int64_t) tetrahedra.size(); idx++)
        {
            std::vector<int64_t> active_tet = tetrahedra[idx];

            std::sort(active_tet.begin(), active_tet.end());

            std::vector<int64_t> triang0 = {active_tet[0], active_tet[1], active_tet[2], idx};
            std::vector<int64_t> triang1 = {active_tet[0], active_tet[1], active_tet[3], idx};
            std::vector<int64_t> triang2 = {active_tet[0], active_tet[2], active_tet[3], idx};
            std::vector<int64_t> triang3 = {active_tet[1], active_tet[2], active_tet[3], idx};

            triangles.push_back(triang0);
            triangles.push_back(triang1);
            triangles.push_back(triang2);
            triangles.push_back(triang3);
        }

        //sort triangles
        std::sort(triangles.begin(), triangles.end(), sort_triangles());
        for (uint64_t idx = 0; idx < triangles.size(); idx++)
        {
            if (idx == triangles.size()-1){
                exterior_surface.push_back(triangles[idx]);
                cell_id.push_back(triangles[idx][3]);
                break;
            }

            if(identical_triangle(triangles[idx], triangles[idx+1]))
                idx++;
            else{
                exterior_surface.push_back(triangles[idx]);
                cell_id.push_back(triangles[idx][3]);
                }
        }
        //reorder the surface triangles to point outwards
        for (uint64_t idx = 0; idx < exterior_surface.size(); idx++)
        {
            std::vector<int64_t> active_tet = tetrahedra[exterior_surface[idx][3]];

            std::sort(active_tet.begin(), active_tet.end());

            std::vector<std::vector<int64_t>> triang = {{active_tet[0], active_tet[1], active_tet[2]},{active_tet[0], active_tet[1], active_tet[3]},
                {active_tet[0], active_tet[2], active_tet[3]},{active_tet[1], active_tet[2], active_tet[3]}};

            int inward_point = 0;
            int tri = 0; //the triangle

            //record inward point and the triangle to be reordered
            if(identical_triangle(triang[0], exterior_surface[idx])) inward_point = 3;
            else if(identical_triangle(triang[1], exterior_surface[idx])) {inward_point = 2; tri = 1;}
            else if(identical_triangle(triang[2], exterior_surface[idx])) {inward_point = 1; tri = 2;}
            else if(identical_triangle(triang[3], exterior_surface[idx])) {inward_point = 0; tri = 3;}

            std::vector<float> Pe = vertices[active_tet[inward_point]];
            std::vector<float> P1 = vertices[triang[tri][0]];
            std::vector<float> P2 = vertices[triang[tri][1]];
            std::vector<float> P3 = vertices[triang[tri][2]];

            std::vector<float> normal1 = {(P2[1]-P1[1])*(P3[2]-P1[2])-(P3[1]-P1[1])*(P2[2]-P1[2]),
                                          (P2[2]-P1[2])*(P3[0]-P1[0])-(P2[0]-P1[0])*(P3[2]-P1[2]),
                                          (P2[0]-P1[0])*(P3[1]-P1[1])-(P3[0]-P1[0])*(P2[1]-P1[1])};
            std::vector<float> normal2 = {(P2[1]-P3[1])*(P1[2]-P3[2])-(P1[1]-P3[1])*(P2[2]-P3[2]),
                                          (P2[2]-P3[2])*(P1[0]-P3[0])-(P2[0]-P3[0])*(P1[2]-P3[2]),
                                          (P2[0]-P3[0])*(P1[1]-P3[1])-(P1[0]-P3[0])*(P2[1]-P3[1])};

            float dist1 = (Pe[0]-(P1[0]+normal1[0]))*(Pe[0]-(P1[0]+normal1[0]))+(Pe[1]-(P1[1]+normal1[1]))*(Pe[1]-(P1[1]+normal1[1]))+(Pe[2]-(P1[2]+normal1[2]))*(Pe[2]-(P1[2]+normal1[2]));
            float dist2 = (Pe[0]-(P1[0]+normal2[0]))*(Pe[0]-(P1[0]+normal2[0]))+(Pe[1]-(P1[1]+normal2[1]))*(Pe[1]-(P1[1]+normal2[1]))+(Pe[2]-(P1[2]+normal2[2]))*(Pe[2]-(P1[2]+normal2[2]));

            if (dist1 >= dist2) exterior_surface[idx] = {triang[tri][0], triang[tri][1], triang[tri][2]};
            else exterior_surface[idx] = {triang[tri][2], triang[tri][1], triang[tri][0]};
        }

        return exterior_surface;
    }

    std::vector<std::vector<int64_t>> extract_exterior_surface_normalized(std::vector<std::vector<float>> &vertices, std::vector<std::vector<int64_t>> &tetrahedra, std::vector<labeltype> &labels, labeltype valid_label, bool labeled)
    {
        std::vector<std::vector<int64_t>> triangles, exterior_surface;

        for (int64_t idx = 0; idx < (int64_t) tetrahedra.size(); idx++)
        {
            if (valid_label != 0 && labels[idx] != valid_label)
                continue;

            std::vector<int64_t> active_tet = tetrahedra[idx];

            std::sort(active_tet.begin(), active_tet.end());

            std::vector<int64_t> triang0 = {active_tet[0], active_tet[1], active_tet[2], idx};
            std::vector<int64_t> triang1 = {active_tet[0], active_tet[1], active_tet[3], idx};
            std::vector<int64_t> triang2 = {active_tet[0], active_tet[2], active_tet[3], idx};
            std::vector<int64_t> triang3 = {active_tet[1], active_tet[2], active_tet[3], idx};

            triangles.push_back(triang0);
            triangles.push_back(triang1);
            triangles.push_back(triang2);
            triangles.push_back(triang3);
        }

        //sort triangles
        std::sort(triangles.begin(), triangles.end(), sort_triangles());
        for (uint64_t idx = 0; idx < triangles.size(); idx++)
        {
            if (idx == triangles.size()-1){
                exterior_surface.push_back(triangles[idx]);
                break;
            }

            if(identical_triangle(triangles[idx], triangles[idx+1]))
                idx++;
            else
                exterior_surface.push_back(triangles[idx]);
        }


        //reorder the surface triangles to point outwards
        for (uint64_t idx = 0; idx < exterior_surface.size(); idx++)
        {
            int64_t tet_idx = exterior_surface[idx][3];
            std::vector<int64_t> active_tet = tetrahedra[tet_idx];

            std::sort(active_tet.begin(), active_tet.end());

            std::vector<std::vector<int64_t>> triang = {{active_tet[0], active_tet[1], active_tet[2]},{active_tet[0], active_tet[1], active_tet[3]},
                {active_tet[0], active_tet[2], active_tet[3]},{active_tet[1], active_tet[2], active_tet[3]}};

            int inward_point = 0;
            int tri = 0; //the triangle

            //record inward point and the triangle to be reordered
            if(identical_triangle(triang[0], exterior_surface[idx])) inward_point = 3;
            else if(identical_triangle(triang[1], exterior_surface[idx])) {inward_point = 2; tri = 1;}
            else if(identical_triangle(triang[2], exterior_surface[idx])) {inward_point = 1; tri = 2;}
            else if(identical_triangle(triang[3], exterior_surface[idx])) {inward_point = 0; tri = 3;}

            std::vector<float> Pe = vertices[active_tet[inward_point]];
            std::vector<float> P1 = vertices[triang[tri][0]];
            std::vector<float> P2 = vertices[triang[tri][1]];
            std::vector<float> P3 = vertices[triang[tri][2]];

            std::vector<float> normal1 = {(P2[1]-P1[1])*(P3[2]-P1[2])-(P3[1]-P1[1])*(P2[2]-P1[2]),
                                          (P2[2]-P1[2])*(P3[0]-P1[0])-(P2[0]-P1[0])*(P3[2]-P1[2]),
                                          (P2[0]-P1[0])*(P3[1]-P1[1])-(P3[0]-P1[0])*(P2[1]-P1[1])};
            std::vector<float> normal2 = {(P2[1]-P3[1])*(P1[2]-P3[2])-(P1[1]-P3[1])*(P2[2]-P3[2]),
                                          (P2[2]-P3[2])*(P1[0]-P3[0])-(P2[0]-P3[0])*(P1[2]-P3[2]),
                                          (P2[0]-P3[0])*(P1[1]-P3[1])-(P1[0]-P3[0])*(P2[1]-P3[1])};

            float dist1 = (Pe[0]-(P1[0]+normal1[0]))*(Pe[0]-(P1[0]+normal1[0]))+(Pe[1]-(P1[1]+normal1[1]))*(Pe[1]-(P1[1]+normal1[1]))+(Pe[2]-(P1[2]+normal1[2]))*(Pe[2]-(P1[2]+normal1[2]));
            float dist2 = (Pe[0]-(P1[0]+normal2[0]))*(Pe[0]-(P1[0]+normal2[0]))+(Pe[1]-(P1[1]+normal2[1]))*(Pe[1]-(P1[1]+normal2[1]))+(Pe[2]-(P1[2]+normal2[2]))*(Pe[2]-(P1[2]+normal2[2]));

            if(labeled)
            {
                if (dist1 >= dist2) exterior_surface[idx] = {triang[tri][0], triang[tri][1], triang[tri][2], tet_idx};
                else exterior_surface[idx] = {triang[tri][2], triang[tri][1], triang[tri][0], tet_idx};
            }
            else
            {
                if (dist1 >= dist2) exterior_surface[idx] = {triang[tri][0], triang[tri][1], triang[tri][2]};
                else exterior_surface[idx] = {triang[tri][2], triang[tri][1], triang[tri][0]};
            }
        }

        return exterior_surface;
    }
    int get_abaqus_facet_label(const std::vector<int64_t> &tet, const std::vector<int64_t> &triangle)
    {
        if (tet[3] != triangle[0] && tet[3] != triangle[1] && tet[3] != triangle[2]) return 1;
        if (tet[2] != triangle[0] && tet[2] != triangle[1] && tet[2] != triangle[2]) return 2;
        if (tet[0] != triangle[0] && tet[0] != triangle[1] && tet[0] != triangle[2]) return 3;
        if (tet[1] != triangle[0] && tet[1] != triangle[1] && tet[1] != triangle[2]) return 4;

        std::cout << "triangle is not part of this tetrahedron" << std::endl;
        return -1;
    }
    std::vector<std::vector<int64_t>> extract_interfaces_and_reorder_for_abaqus(std::vector<std::vector<int64_t>> &tetrahedra, std::vector<labeltype> &labels)
    {
        //extract material interfaces and reorders the nodal values to be ascending
        std::vector<std::vector<int64_t>> triangles, interfaces;

        for (int64_t idx = 0; idx < (int64_t) tetrahedra.size(); idx++)
        {
            std::vector<int64_t> active_tet = tetrahedra[idx];
            int64_t active_label = (int64_t) labels[idx];

            std::sort(active_tet.begin(), active_tet.end());

            //Following S1,S2,S3,S4 convention from abaqus for column 5 but with ascending tetrahedra index
            std::vector<int64_t> triang0 = {active_tet[0], active_tet[1], active_tet[2], active_label, idx, -1};
            std::vector<int64_t> triang1 = {active_tet[0], active_tet[1], active_tet[3], active_label, idx, -1};
            std::vector<int64_t> triang2 = {active_tet[1], active_tet[2], active_tet[3], active_label, idx, -1};
            std::vector<int64_t> triang3 = {active_tet[0], active_tet[2], active_tet[3], active_label, idx, -1};

            //correct Sx convention for original tetrahedron with positive volume
            triang0[5] = get_abaqus_facet_label(tetrahedra[idx], triang0);
            triang1[5] = get_abaqus_facet_label(tetrahedra[idx], triang1);
            triang2[5] = get_abaqus_facet_label(tetrahedra[idx], triang2);
            triang3[5] = get_abaqus_facet_label(tetrahedra[idx], triang3);

            triangles.push_back(triang0);
            triangles.push_back(triang1);
            triangles.push_back(triang2);
            triangles.push_back(triang3);
        }

        //sort triangles
        std::sort(triangles.begin(), triangles.end(), sort_triangles());
        for (uint64_t idx = 0; idx < triangles.size(); idx++)
        {
            if (idx == triangles.size()-1){
                interfaces.push_back(triangles[idx]);
                break;
            }

            if(identical_triangle(triangles[idx], triangles[idx+1]))
            {
                if (triangles[idx][3] < triangles[idx+1][3])
                {
                    triangles[idx][3] = 0; //label zero marks interior surface for bone
                    interfaces.push_back(triangles[idx]);

                    triangles[idx+1][3] = -1; //label zero marks interior surface for screw
                    interfaces.push_back(triangles[idx+1]);
                }
                else if (triangles[idx][3] > triangles[idx+1][3])
                {
                    triangles[idx][3] = -1;  interfaces.push_back(triangles[idx]);
                    triangles[idx+1][3] = 0; interfaces.push_back(triangles[idx+1]);
                }
                idx++;
            }
            else
                interfaces.push_back(triangles[idx]);
        }

        return interfaces;
    }
    int64_t remove_disconnected_bone(labeltype label_screw, std::vector<std::vector<int64_t>> &tetrahedra, std::vector<labeltype> &labels)
    {
        //burn all labels with the screw label and remove disconnected components

        std::vector<std::vector<int64_t>> triangles;
        std::vector<labeltype> burned_labels = labels;

        //create list of surface facets and reorder tetrahedra nodes
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        for (int64_t idx = 0; idx < (int64_t) tetrahedra.size(); idx++)
        {
            std::vector<int64_t> active_tet = tetrahedra[idx];
            std::sort(active_tet.begin(), active_tet.end());

            //tetrahedra index needs to be ascending for sorting query
            std::vector<int64_t> triang0 = {active_tet[0], active_tet[1], active_tet[2], idx};
            std::vector<int64_t> triang1 = {active_tet[0], active_tet[1], active_tet[3], idx};
            std::vector<int64_t> triang2 = {active_tet[1], active_tet[2], active_tet[3], idx};
            std::vector<int64_t> triang3 = {active_tet[0], active_tet[2], active_tet[3], idx};

            triangles.push_back(triang0);
            triangles.push_back(triang1);
            triangles.push_back(triang2);
            triangles.push_back(triang3);
        }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        //sort triangles
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        std::sort(triangles.begin(), triangles.end(), sort_triangles());
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        //burn from screw
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        bool change = true;
        //int64_t iteration = 0;

        while(change)
        {
            change = false;
            //iteration++;

            #pragma omp parallel for
            for (int64_t idx = 0; idx < (int64_t) triangles.size()-1; idx++) //cannot be uint
            {
                if(identical_triangle(triangles[idx], triangles[idx+1]))
                {
                    int64_t tet_idx0 = triangles[idx][3];
                    int64_t tet_idx1 = triangles[idx+1][3];

                    //different phase
                    if (burned_labels[tet_idx0] != 0 && burned_labels[tet_idx0] != label_screw && burned_labels[tet_idx1] == label_screw)
                    {
                        change = true;
                        burned_labels[tet_idx0] = label_screw;
                    }
                    else if (burned_labels[tet_idx0] == label_screw && burned_labels[tet_idx1] != 0 && burned_labels[tet_idx1] != label_screw)
                    {
                        change = true;
                        burned_labels[tet_idx1] = label_screw;
                    }

                    idx++;
                }
            }
        }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        //erase tetrahedra that are still bone phase
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        int64_t n_removed = 0;

        std::vector<std::vector<int64_t>> new_tetrahedra;
        std::vector<labeltype> new_labels;

        for (int64_t idx = 0; idx < (int64_t) tetrahedra.size(); idx++) //cannot be uint!
        {
            if (burned_labels[idx] != label_screw)
                n_removed++;
            else
            {
                new_tetrahedra.push_back(tetrahedra[idx]);
                new_labels.push_back(labels[idx]);
            }
        }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        labels = new_labels;
        tetrahedra = new_tetrahedra;

        return n_removed;
    }
    int64_t reduce2dirichletboundaries(std::vector<labeltype> label_colors, std::vector<std::vector<int64_t>> &tetrahedra, std::vector<labeltype> &labels)
    {
        //burn all labels with the screw label. Then remove disconnected components and all tetrahedra that are neither bone nor screw

        labeltype label_bone = label_colors[0];
        labeltype label_screw = label_colors[1];
        labeltype label_pin = label_colors[2];
        labeltype label_cement = label_colors[3];
        labeltype label_bone_next2pin = label_colors[4];
        labeltype label_screw_next2pin = label_colors[5];

        std::vector<std::vector<int64_t>> triangles;
        std::vector<labeltype> burned_labels = labels;

        //create list of surface facets and reorder tetrahedra nodes
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        for (int64_t idx = 0; idx < (int64_t) tetrahedra.size(); idx++)
        {
            std::vector<int64_t> active_tet = tetrahedra[idx];
            std::sort(active_tet.begin(), active_tet.end());

            //tetrahedra index needs to be ascending for sorting query
            std::vector<int64_t> triang0 = {active_tet[0], active_tet[1], active_tet[2], idx};
            std::vector<int64_t> triang1 = {active_tet[0], active_tet[1], active_tet[3], idx};
            std::vector<int64_t> triang2 = {active_tet[1], active_tet[2], active_tet[3], idx};
            std::vector<int64_t> triang3 = {active_tet[0], active_tet[2], active_tet[3], idx};

            triangles.push_back(triang0);
            triangles.push_back(triang1);
            triangles.push_back(triang2);
            triangles.push_back(triang3);
        }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        //sort triangles
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        std::sort(triangles.begin(), triangles.end(), sort_triangles());
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        //burn from screw
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        bool change = true;
        //int64_t iteration = 0;

        while(change)
        {
            change = false;
            //iteration++;

            #pragma omp parallel for
            for (int64_t idx = 0; idx < (int64_t) triangles.size()-1; idx++) //cannot be uint
            {
                if(identical_triangle(triangles[idx], triangles[idx+1]))
                {
                    int64_t tet_idx0 = triangles[idx][3];
                    int64_t tet_idx1 = triangles[idx+1][3];

                    //bone phase
                    if (burned_labels[tet_idx0] == label_bone && burned_labels[tet_idx1] == label_screw)
                    {
                        change = true;
                        burned_labels[tet_idx0] = label_screw;
                    }
                    else if (burned_labels[tet_idx0] == label_screw && burned_labels[tet_idx1] == label_bone)
                    {
                        change = true;
                        burned_labels[tet_idx1] = label_screw;
                    }
                    //next to cement phase
                    else if (burned_labels[tet_idx0] == label_cement && burned_labels[tet_idx1] == label_screw && labels[tet_idx1] != label_cement)
                    {
                        change = true;
                        labels[tet_idx1] = label_cement;
                    }
                    else if (burned_labels[tet_idx0] == label_screw && burned_labels[tet_idx1] == label_cement && labels[tet_idx0] != label_cement)
                    {
                        change = true;
                        labels[tet_idx0] = label_cement;
                    }
                    //bone next to pin phase
                    else if (burned_labels[tet_idx0] == label_pin && burned_labels[tet_idx1] == label_screw && labels[tet_idx1] == label_bone)
                    {
                        change = true;
                        labels[tet_idx1] = label_bone_next2pin;
                    }
                    else if (burned_labels[tet_idx0] == label_screw && burned_labels[tet_idx1] == label_pin && labels[tet_idx0] == label_bone)
                    {
                        change = true;
                        labels[tet_idx0] = label_bone_next2pin;
                    }
                    //screw next to pin phase
                    else if (burned_labels[tet_idx0] == label_pin && burned_labels[tet_idx1] == label_screw && labels[tet_idx1] == label_screw)
                    {
                        change = true;
                        labels[tet_idx1] = label_screw_next2pin;
                    }
                    else if (burned_labels[tet_idx0] == label_screw && burned_labels[tet_idx1] == label_pin && labels[tet_idx0] == label_screw)
                    {
                        change = true;
                        labels[tet_idx0] = label_screw_next2pin;
                    }

                    idx++;
                }
            }
        }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        //erase tetrahedra that are still bone phase
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        int64_t n_removed = 0;

        std::vector<std::vector<int64_t>> new_tetrahedra;
        std::vector<labeltype> new_labels;

        for (int64_t idx = 0; idx < (int64_t) tetrahedra.size(); idx++) //cannot be uint!
        {
            if (burned_labels[idx] != label_screw)
                n_removed++;
            else
            {
                new_tetrahedra.push_back(tetrahedra[idx]);
                new_labels.push_back(labels[idx]);
            }
        }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        labels = new_labels;
        tetrahedra = new_tetrahedra;

        return n_removed;
    }

    float relabel_debugging(std::vector<std::vector<int64_t>> &tetrahedra, std::vector<int> &labels, int label2relabel, int interface_label, std::vector<float> &edm_screw)
    {
        std::vector<std::vector<int64_t>> triangles;

        for (uint64_t idx = 0; idx < tetrahedra.size(); idx++)
        {
            std::vector<int64_t> active_tet = tetrahedra[idx];

            std::sort(active_tet.begin(), active_tet.end());

            std::vector<int64_t> triang0 = {active_tet[0], active_tet[1], active_tet[2], labels[idx], idx};
            std::vector<int64_t> triang1 = {active_tet[0], active_tet[1], active_tet[3], labels[idx], idx};
            std::vector<int64_t> triang2 = {active_tet[0], active_tet[2], active_tet[3], labels[idx], idx};
            std::vector<int64_t> triang3 = {active_tet[1], active_tet[2], active_tet[3], labels[idx], idx};

            triangles.push_back(triang0);
            triangles.push_back(triang1);
            triangles.push_back(triang2);
            triangles.push_back(triang3);
        }

        float maxedm = 0.0;

        //sort triangles
        std::sort(triangles.begin(), triangles.end(), sort_triangles());
        for (uint64_t idx = 0; idx < triangles.size(); idx++)
        {
            if (idx == triangles.size()-1){
                break;
            }

            if(identical_triangle(triangles[idx], triangles[idx+1]))
            {
                if (triangles[idx][3] == label2relabel && triangles[idx+1][3] != label2relabel) {
                    labels[triangles[idx][4]] = interface_label;
                    maxedm = (edm_screw[triangles[idx][4]] > maxedm) ? maxedm = edm_screw[triangles[idx][4]] : maxedm;
                }
                else if (triangles[idx][3] != label2relabel && triangles[idx+1][3] == label2relabel) {labels[triangles[idx+1][4]] = interface_label;
                    maxedm = (edm_screw[triangles[idx+1][4]] > maxedm) ? maxedm = edm_screw[triangles[idx+1][4]] : maxedm;
                }
                //if (triangles[idx][3] == label2relabel && triangles[idx+1][3] != label2relabel) {labels[triangles[idx][4]] = interface_label; labels[triangles[idx+1][4]] = interface_label;}
                //else if (triangles[idx][3] != label2relabel && triangles[idx+1][3] == label2relabel) {labels[triangles[idx+1][4]] = interface_label; labels[triangles[idx][4]] = interface_label;}

                idx++;
            }
        }

        return maxedm;
    }
}
