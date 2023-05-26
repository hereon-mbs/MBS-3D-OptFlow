#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <ctime>

#include "meshio.h"

namespace meshio
{
    int makepath(std::string s){
        size_t pos=0;
        std::string dir;
        int mdret = -1;

        if(s[s.size()-1]!='/'){s+='/';}

        while((pos=s.find_first_of('/',pos))!=std::string::npos){
            dir=s.substr(0,pos++);
            if(dir.size()==0) continue; // if leading / first time is 0 length
            if((mdret=mkdir(dir.c_str(),0777)) && errno!=EEXIST){
                return mdret;
            }
        }
        return mdret;
    }
    bool path_exists(const std::string& path){
        struct stat info;
        if (stat(path.c_str(), &info) != 0)
        {
            return false;
        }
        return (info.st_mode & S_IFDIR) != 0;
    }

    void read_shapefile(std::string filename, int shape[3], std::string sample_ID){
        std::string line;
        std::ifstream file (filename);

        if (file.is_open())
        {
            getline (file,line);
            if (sample_ID != line)
                std::cout << "shapefile does not match sample_ID!" << std::endl;
            getline (file,line);
            shape[0] = std::stoi(line);
            getline (file,line);
            shape[1] = std::stoi(line);
            getline (file,line);
            shape[2] = std::stoi(line);
        }
        else
        {
            std::cout << "Can't read " << filename << std::endl;
        }
        file.close();
        return;
    }

    //Surface Meshes
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Mesh read_off(std::string filename){
        std::string line;
        std::ifstream offfile (filename);

        std::vector<std::vector<float>> out_points;
        std::vector<std::vector<int64_t>> out_triangles;

        if (offfile.is_open())
        {
            int64_t linecount = 0;
            int64_t lastpointline = 10000000;
            int64_t lasttriangleline = 10000000;

            while ( getline (offfile,line) )
            {
                linecount++;

                std::istringstream iss(line);
                std::string word;

                std::vector<float> coordinates;

                if (linecount == 2 || linecount >= 3)
                {
                    int counter=0;
                    //extract header and coordinates
                    while(std::getline(iss, word, ' '))
                    {
                        counter++;
                        if (linecount <= lastpointline || (counter > 1 && word != ""))
                        {
                            coordinates.push_back(std::stod(word));
                        }
                    }
                }

                if (linecount == 2)
                {
                    lastpointline = (int) coordinates[0] + 2; //get the number of vertices
                    lasttriangleline = (int) coordinates[1] + lastpointline;
                }
                if (linecount >= 3 && linecount <= lastpointline)
                {
                    if (line == "")
                    {
                        lastpointline++;
                        lasttriangleline++;
                    }
                    else
                        out_points.push_back({coordinates[0], coordinates[1], coordinates[2]});
                }
                if (linecount > lastpointline)
                {
                    if (line == "")
                    {
                        lasttriangleline++;
                    }
                    else
                        out_triangles.push_back({(int64_t) coordinates[0], (int64_t) coordinates[1], (int64_t) coordinates[2]});
                }
                if (linecount == lasttriangleline) //next line will be triangles
                    break;
            }
            offfile.close();
        }

        Mesh outmesh;
        outmesh.type = "surface";
        outmesh.triangles = out_triangles;
        outmesh.vertices = out_points;

        return outmesh;    }
    void write_off(std::string path, std::string name, std::vector<std::vector<float>> &points, std::vector<std::vector<int64_t>> &triangles){
        int64_t n_points = points.size();
        int64_t n_triangles = triangles.size();

        if (n_points == 0) return;

        makepath(path);
        std::ofstream outfile;
        outfile.open(path + "/" + name);

        outfile << "OFF\n";
        outfile << n_points << " " << n_triangles << " 0\n\n";

        for (uint64_t i = 0; i < points.size(); i++)
        {
            outfile << points[i][0] << " " << points[i][1] << " " << points[i][2] << "\n";
        }
        for (uint64_t i = 0; i < triangles.size(); i++)
        {
            int64_t point0 = triangles[i][0];
            int64_t point1 = triangles[i][1];
            int64_t point2 = triangles[i][2];

            outfile << "3  " << point0 << " " << point1 << " " << point2 << "\n";
        }
        outfile << "\n";
        outfile.close();

        return;
    }
    void write_colored_off(std::string path, std::string name, std::vector<std::vector<float>> &points, std::vector<std::vector<int64_t>> &triangles,
                        std::vector<int64_t> &shared_triangles, int color_unique[3], int color_shared[3]){
        int64_t n_points = points.size();
        int64_t n_triangles = triangles.size();

        if (n_points == 0) return;

        makepath(path);
        std::ofstream outfile;
        outfile.open(path + "/" + name);

        outfile << "OFF\n";
        outfile << n_points << " " << n_triangles << " 0\n\n";

        for (uint64_t i = 0; i < points.size(); i++)
        {
            outfile << points[i][0] << " " << points[i][1] << " " << points[i][2] << "\n";
        }
        for (uint64_t i = 0; i < triangles.size(); i++)
        {
            int64_t point0 = triangles[i][0];
            int64_t point1 = triangles[i][1];
            int64_t point2 = triangles[i][2];

            int64_t is_shared = shared_triangles[i];

            if (is_shared != -1)
                outfile << "3  " << point0 << " " << point1 << " " << point2 << " " << color_shared[0] << " " << color_shared[1] << " " << color_shared[2] << "\n";
            else
                outfile << "3  " << point0 << " " << point1 << " " << point2 << " " << color_unique[0] << " " << color_unique[1] << " " << color_unique[2] << "\n";
        }
        outfile << "\n";
        outfile.close();

        return;
    }
    void write_colored_off(std::string path, std::string name, std::vector<std::vector<float>> &points, std::vector<std::vector<int64_t>> &triangles,
                        std::vector<int> &threePhaseLabels, int color0[3], int color1[3], int color2[3]){
        int64_t n_points = points.size();
        int64_t n_triangles = triangles.size();

        if (n_points == 0) return;

        makepath(path);
        std::ofstream outfile;
        outfile.open(path + "/" + name);

        outfile << "OFF\n";
        outfile << n_points << " " << n_triangles << " 0\n\n";

        for (uint64_t i = 0; i < points.size(); i++)
        {
            outfile << points[i][0] << " " << points[i][1] << " " << points[i][2] << "\n";
        }
        for (uint64_t i = 0; i < triangles.size(); i++)
        {
            int64_t point0 = triangles[i][0];
            int64_t point1 = triangles[i][1];
            int64_t point2 = triangles[i][2];

            int color = threePhaseLabels[i];

            if (color == 0)
                outfile << "3  " << point0 << " " << point1 << " " << point2 << " " << color0[0] << " " << color0[1] << " " << color0[2] << "\n";
            else if (color == 1)
                outfile << "3  " << point0 << " " << point1 << " " << point2 << " " << color1[0] << " " << color1[1] << " " << color1[2] << "\n";
            else if (color == 2)
                outfile << "3  " << point0 << " " << point1 << " " << point2 << " " << color2[0] << " " << color2[1] << " " << color2[2] << "\n";
            else
                std::cout << "Warning: unknown label " << color << std::endl;
        }
        outfile << "\n";
        outfile.close();

        return;
    }
    int64_t write_subset_off(int32_t active_label, std::string path, std::string name, std::vector<int32_t> &labels, std::vector<std::vector<float>> &points, std::vector<std::vector<int64_t>> &triangles){
        int64_t n_points = 0;
        int64_t n_triangles = 0;
        std::vector<int64_t> new_points_id(points.size(), -1);

        for (uint64_t i = 0; i < points.size(); i++)
            if (labels[i] == active_label) n_points++;
        for (uint64_t i = 0; i < triangles.size(); i++)
            if (labels[i+points.size()] == active_label) n_triangles++;

        if (n_triangles == 0) return 0;

        makepath(path);
        std::ofstream outfile;
        outfile.open(path + "/" + name);

        outfile << "OFF\n";
        outfile << n_points << " " << n_triangles << " 0\n\n";

        int64_t id = 0;
        for (uint64_t i = 0; i < points.size(); i++)
        {
            if (labels[i] == active_label)
            {
                new_points_id[i] = id;
                id++;

                outfile << points[i][0] << " " << points[i][1] << " " << points[i][2] << "\n";
            }
        }
        for (uint64_t i = 0; i < triangles.size(); i++)
        {
            if (labels[i+points.size()] == active_label)
                outfile << "3  " << new_points_id[triangles[i][0]] << " " << new_points_id[triangles[i][1]] << " " << new_points_id[triangles[i][2]] << "\n";
        }
        outfile << "\n";
        outfile.close();

        return n_triangles;
    }

    void merge_meshfiles(std::string outpath, std::string outfilename, std::vector<std::string> meshfiles, bool save_triangles)
    {
        std::vector<std::vector<float>> vertices;
        std::vector<std::vector<int32_t>> triangles;
        std::vector<std::vector<int32_t>> tetrahedra;

        int32_t component_label = 0;

        for (uint64_t i = 0; i < meshfiles.size(); i++)
        {
            component_label++;

            //attach to output
            std::string line;
            std::ifstream file (meshfiles[i]);

            std::vector<std::vector<float>> new_vertices;
            std::vector<std::vector<int32_t>> new_triangles;
            std::vector<std::vector<int32_t>> new_tetrahedra;
            std::vector<int> active_vertices;

            int64_t vertex_offset = vertices.size();

            if (file.is_open())
            {
                bool vert_active = false;
                bool tri_active = false;
                bool tet_active = false;

                int64_t linecount = 0;

                getline(file,line);
                getline(file,line);

                while ( getline(file,line))
                {
                    int wordcount = 0;

                    std::istringstream iss(line);
                    std::string word;

                    std::vector<float> coordinates;

                    std::getline(iss, word, ' ');
                    linecount++;

                    if (word == "Vertices")  {vert_active = true;  tri_active = false; tet_active = false; linecount = 0; continue;}
                    if (word == "Triangles") {vert_active = false; tri_active = true;  tet_active = false; linecount = 0; continue;}
                    if (word == "Tetrahedra") {vert_active = false; tri_active = false; tet_active = true; linecount = 0; continue;}
                    if (word == "End") break;

                    if (linecount > 1)
                    {
                        coordinates.push_back(std::stod(word));
                        while(std::getline(iss, word, ' ')){
                            coordinates.push_back(std::stod(word));
                        }

                        if (vert_active)
                        {
                            active_vertices.push_back(0);
                            new_vertices.push_back({coordinates[0], coordinates[1], coordinates[2], (int) coordinates[3]});
                            continue;
                        }
                        if (tri_active)
                        {
                            if (save_triangles == false) continue;
                            new_triangles.push_back({(int32_t) coordinates[0], (int32_t) coordinates[1], (int32_t) coordinates[2], component_label});
                            active_vertices[new_triangles[new_triangles.size()-1][0]-1] = 1;
                            active_vertices[new_triangles[new_triangles.size()-1][1]-1] = 1;
                            active_vertices[new_triangles[new_triangles.size()-1][2]-1] = 1;
                            continue;
                        }
                        if (tet_active)
                        {
                            new_tetrahedra.push_back({(int32_t) coordinates[0], (int32_t) coordinates[1], (int32_t) coordinates[2], (int32_t) coordinates[3], component_label});
                            active_vertices[new_tetrahedra[new_tetrahedra.size()-1][0]-1] = 1;
                            active_vertices[new_tetrahedra[new_tetrahedra.size()-1][1]-1] = 1;
                            active_vertices[new_tetrahedra[new_tetrahedra.size()-1][2]-1] = 1;
                            active_vertices[new_tetrahedra[new_tetrahedra.size()-1][3]-1] = 1;
                            continue;
                        }
                    }
                }
            }
            else continue;
            file.close();

            //delete unnecessary vertices
            std::vector<int32_t> shift(new_vertices.size(), 0);
            int32_t skips = 0;
            for (uint64_t n = 0; n < new_vertices.size(); n++)
            {
                if (active_vertices[n] == 0)
                {

                    skips++;
                }
                else
                    vertices.push_back(new_vertices[n]);

                shift[n] = skips;
            }
            for (uint64_t n = 0; n < new_triangles.size(); n++)
            {
                std::vector<int32_t> triangle(4,0);
                triangle[0] = new_triangles[n][0]+vertex_offset-shift[new_triangles[n][0]-1];
                triangle[1] = new_triangles[n][1]+vertex_offset-shift[new_triangles[n][1]-1];
                triangle[2] = new_triangles[n][2]+vertex_offset-shift[new_triangles[n][2]-1];
                triangle[3] = new_triangles[n][3];

                triangles.push_back(triangle);
            }
            for (uint64_t n = 0; n < new_tetrahedra.size(); n++)
            {
                std::vector<int32_t> tetrahedron(5,0);
                tetrahedron[0] = new_tetrahedra[n][0]+vertex_offset-shift[new_tetrahedra[n][0]-1];
                tetrahedron[1] = new_tetrahedra[n][1]+vertex_offset-shift[new_tetrahedra[n][1]-1];
                tetrahedron[2] = new_tetrahedra[n][2]+vertex_offset-shift[new_tetrahedra[n][2]-1];
                tetrahedron[3] = new_tetrahedra[n][3]+vertex_offset-shift[new_tetrahedra[n][3]-1];
                tetrahedron[4] = new_tetrahedra[n][4];

                tetrahedra.push_back(tetrahedron);
            }
        }
        if (tetrahedra.size() > 0)
        {
            //create output
            makepath(outpath);
            std::ofstream outfile;
            outfile.open(outpath + "/" + outfilename);

            outfile << "MeshVersionFormatted 1\n";
            outfile << "Dimension 3\n";
            outfile << "Vertices\n";
            outfile << vertices.size() << "\n";
            for (uint64_t i = 0; i < vertices.size(); i++)
                outfile << vertices[i][0] << " " << vertices[i][1] << " " << vertices[i][2] << " " << vertices[i][3] << "\n";
            if (save_triangles)
            {
            outfile << "Triangles\n";
            outfile << triangles.size() << "\n";
            for (uint64_t i = 0; i < triangles.size(); i++)
                outfile << triangles[i][0] << " " << triangles[i][1] << " " << triangles[i][2] << " " << triangles[i][3] << "\n";
            }
            outfile << "Tetrahedra\n";
            outfile << tetrahedra.size() << "\n";
            for (uint64_t i = 0; i < tetrahedra.size(); i++)
                outfile << tetrahedra[i][0] << " " << tetrahedra[i][1] << " " << tetrahedra[i][2] << " " << tetrahedra[i][3] << " " << tetrahedra[i][4] << "\n";
            outfile << "End\n";
            outfile.close();
        }
        return;
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //Volume Meshes
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Mesh read_meditmesh_labeled(std::string filename){
        Mesh mesh;

        int n_unique_labels = 0;
        std::vector<int> encountered(258, 0);

        std::string line, word;
        std::ifstream meshfile (filename);

        int64_t n_vertices = -1;
        int64_t n_triangles = -1;
        int64_t n_tetrahedra = -1;

        if (meshfile.is_open())
        {
            while (getline(meshfile,line))
            {
                if (line == "End") break;
                if (line == "Vertices"){
                    getline(meshfile,line);
                    n_vertices = std::stoi(line);

                    for (int64_t i = 0; i < n_vertices; i++)
                    {
                        getline(meshfile,line);
                        std::istringstream iss(line);

                        std::getline(iss, word, ' ');
                        vertextype coord0 = std::stod(word);
                        std::getline(iss, word, ' ');
                        vertextype coord1 = std::stod(word);
                        std::getline(iss, word, ' ');
                        vertextype coord2 = std::stod(word);

                        mesh.vertices.push_back({coord0,coord1,coord2});
                    }
                }
                if (line == "Triangles"){
                    getline(meshfile,line);
                    n_triangles = std::stoi(line);

                    for (int64_t i = 0; i < n_triangles; i++)
                    {
                        getline(meshfile,line);
                        std::istringstream iss(line);

                        //Subtract 1!!! Medit is 1 based!!!

                        std::getline(iss, word, ' ');
                        celltype vert0 = std::stoi(word)-1;
                        std::getline(iss, word, ' ');
                        celltype vert1 = std::stoi(word)-1;
                        std::getline(iss, word, ' ');
                        celltype vert2 = std::stoi(word)-1;
                        std::getline(iss, word, ' ');
                        int this_label = std::stoi(word);

                        mesh.triangles.push_back({vert0, vert1, vert2});
                    }
                }
                if (line == "Tetrahedra"){
                    getline(meshfile,line);
                    n_tetrahedra = std::stoi(line);

                    for (int64_t i = 0; i < n_tetrahedra; i++)
                    {
                        getline(meshfile,line);
                        std::istringstream iss(line);

                        //Acquire tetrahedra nodes
                        //Subtract 1!!! Medit is 1 based!!!
                        int64_t nodes[4];
                        int this_label = 0;

                        for (int p = 0; p < 4; p++)
                        {
                            std::getline(iss, word, ' ');
                            nodes[p] = std::stoi(word)-1;                        }
                        if(std::getline(iss, word, ' '))
                        {
                            if(i == 0)
                            {
                                mesh.cell_labels_discrete_name.push_back("phase");
                                mesh.cell_labels_discrete.push_back(std::vector<labeltype_discrete>(n_tetrahedra, 4));
                            }
                            this_label = std::stoi(word);
                        }

                        mesh.tetrahedra.push_back({nodes[0],nodes[1],nodes[2],nodes[3]});
                        mesh.cell_labels_discrete[mesh.cell_labels_discrete.size()-1][i] = this_label;
                    }
                }
            }
        }
        else
            std::cout << "Error! Cannot find " << filename << std::endl;

        mesh.n_tetrahedra = std::max((int64_t) 0, n_tetrahedra);
        mesh.n_vertices = std::max((int64_t) 0, n_vertices);
        mesh.n_triangles = std::max((int64_t) 0,n_triangles);

        return mesh;
    }
    Mesh read_unlabeled_vtk_mesh(std::string filename)
    {
        Mesh mesh;
        read_geometry_off_vtk_volumemesh(filename, mesh.vertices, mesh.tetrahedra);
        return mesh;
    }

    void save_mesh2vtk(std::string path, std::string name, const Mesh &mesh, std::string header)
    {
        std::string filename = path + "/" + name + ".vtk";

        int64_t n_vertices = mesh.vertices.size();
        int64_t n_cells = (mesh.type == "volume") ? mesh.tetrahedra.size() : mesh.triangles.size();

        //Create directory if necessary
        //********************************************
        char *dir_path, *header_;
        dir_path = new char[path.length()+1];
        strcpy(dir_path, path.c_str());

        header += "\n";
        header_ = new char[header.length()+1];
        strcpy(header_, header.c_str());

        if (!path_exists(dir_path)) makepath(dir_path);
        //********************************************

        std::ofstream outfile;
        outfile.open(filename);

        outfile << "# vtk DataFile Version 2.0\n";
        outfile << header << "\n";
        outfile << "ASCII\n";
        outfile << "DATASET UNSTRUCTURED_GRID\n";
        outfile << "POINTS " << n_vertices << " float\n";

        for (long long int pos = 0; pos < n_vertices; pos++)
            outfile << mesh.vertices[pos][0] << " " << mesh.vertices[pos][1] << " " << mesh.vertices[pos][2] << "\n";

        if (mesh.type == "surface")
        {
            outfile << "CELLS " << n_cells << " " << (n_cells*4) << "\n"; //second is size, i.e. number of points total following
            for (long long int pos = 0; pos < n_cells; pos++)
                outfile << "3 " << mesh.triangles[pos][0] << " " << mesh.triangles[pos][1] << " " << mesh.triangles[pos][2] << "\n";
            outfile << "CELL_TYPES " << n_cells << "\n"; //cell type for tetrahedra is 10
            for (long long int pos = 0; pos < n_cells; pos++)
                outfile << "5 ";
        }
        else
        {
            outfile << "CELLS " << n_cells << " " << (n_cells*5) << "\n"; //second is size, i.e. number of points total following
            for (long long int pos = 0; pos < n_cells; pos++)
                outfile << "4 " << mesh.tetrahedra[pos][0] << " " << mesh.tetrahedra[pos][1] << " " << mesh.tetrahedra[pos][2] << " " << mesh.tetrahedra[pos][3] << "\n";
            outfile << "CELL_TYPES " << n_cells << "\n"; //cell type for tetrahedra is 10
            for (long long int pos = 0; pos < n_cells; pos++)
                outfile << "10 ";
        }
        outfile << "\n";

        //write cell labels
        ////////////////////////////////////////////////////////////////////////////////////
        if (mesh.cell_labels_discrete.size() > 0 || mesh.cell_labels_floating.size() > 0 || mesh.cell_vectors.size() > 0)
        {
            outfile << "CELL_DATA " << n_cells << "\n";

            for (int i = 0; i < mesh.cell_labels_discrete.size(); i++)
            {
                outfile << "SCALARS " << mesh.cell_labels_discrete_name[i] << " int 1\n";
                outfile << "LOOKUP_TABLE default\n";
                for (long long int pos = 0; pos < n_cells; pos++)
                    outfile << ((int) mesh.cell_labels_discrete[i][pos]) << " ";
                outfile << "\n";
            }
            for (int i = 0; i < mesh.cell_labels_floating.size(); i++)
            {
                outfile << "SCALARS " << mesh.cell_labels_floating_name[i] << " float 1\n";
                outfile << "LOOKUP_TABLE default\n";
                for (long long int pos = 0; pos < n_cells; pos++)
                    outfile << ((float) mesh.cell_labels_floating[i][pos]) << " ";
                outfile << "\n";
            }
            for (int i = 0; i < mesh.cell_vectors.size(); i++)
            {
                outfile << "VECTORS " << mesh.cell_vectors_name[i] << " float\n";
                for (long long int pos = 0; pos < n_cells; pos++)
                {
                    outfile << ((float) mesh.cell_vectors[i][pos][0]) << " ";
                    outfile << ((float) mesh.cell_vectors[i][pos][1]) << " ";
                    outfile << ((float) mesh.cell_vectors[i][pos][2]) << " ";
                }
                outfile << "\n";
            }
        }
        ////////////////////////////////////////////////////////////////////////////////////

        //write vertex labels
        ////////////////////////////////////////////////////////////////////////////////////
        if (mesh.vertex_labels_discrete.size() > 0 || mesh.vertex_labels_floating.size() > 0 || mesh.vertex_vectors.size() > 0)
        {
            outfile << "POINT_DATA " << n_vertices << "\n";

            for (int i = 0; i < mesh.vertex_labels_discrete.size(); i++)
            {
                outfile << "SCALARS " << mesh.vertex_labels_discrete_name[i] << " int 1\n";
                outfile << "LOOKUP_TABLE default\n";
                for (long long int pos = 0; pos < n_vertices; pos++)
                    outfile << ((int) mesh.vertex_labels_discrete[i][pos]) << " ";
                outfile << "\n";
            }
            for (int i = 0; i < mesh.vertex_labels_floating.size(); i++)
            {
                outfile << "SCALARS " << mesh.vertex_labels_floating_name[i] << " float 1\n";
                outfile << "LOOKUP_TABLE default\n";
                for (long long int pos = 0; pos < n_vertices; pos++)
                    outfile << ((float) mesh.vertex_labels_floating[i][pos]) << " ";
                outfile << "\n";
            }
            for (int i = 0; i < mesh.vertex_vectors.size(); i++)
            {
                outfile << "VECTORS " << mesh.vertex_vectors_name[i] << " float\n";
                for (long long int pos = 0; pos < n_vertices; pos++)
                {
                    outfile << ((float) mesh.vertex_vectors[i][pos][0]) << " ";
                    outfile << ((float) mesh.vertex_vectors[i][pos][1]) << " ";
                    outfile << ((float) mesh.vertex_vectors[i][pos][2]) << " ";
                }
                outfile << "\n";
            }
        }
        ////////////////////////////////////////////////////////////////////////////////////

        outfile.close();

        return;
    }

    void read_geometry_off_vtk_volumemesh(std::string filename, std::vector<std::vector<float>> &vertices, std::vector<std::vector<int64_t>> &tetrahedra)
    {
        vertices.clear();
        tetrahedra.clear();
        std::string line, word;
        std::ifstream meshfile (filename);

        int64_t n_vertices = -1;
        int64_t n_tetrahedra = -1;

        if (meshfile.is_open())
        {
            while (getline(meshfile,line))
            {
                if (line.substr(0, 6) == "POINTS")
                {
                    std::istringstream iss2(line);
                    std::getline(iss2, word, ' ');
                    std::getline(iss2, word, ' ');
                    n_vertices = std::stoi(word);

                    for (int64_t i = 0; i < n_vertices; i++)
                    {
                        getline(meshfile,line);
                        std::istringstream iss(line);

                        std::vector<float> active_vertex(3 ,0.0f);

                        std::getline(iss, word, ' ');
                        active_vertex[0] = std::stod(word);
                        std::getline(iss, word, ' ');
                        active_vertex[1] = std::stod(word);
                        std::getline(iss, word, ' ');
                        active_vertex[2] = std::stod(word);
                        vertices.push_back(active_vertex);
                    }
                }
                else if (line.substr(0, 5) == "CELLS")
                {
                    std::istringstream iss2(line);
                    std::getline(iss2, word, ' ');
                    std::getline(iss2, word, ' ');
                    n_tetrahedra = std::stoi(word);

                    for (int64_t i = 0; i < n_tetrahedra; i++)
                    {
                        getline(meshfile,line);
                        std::istringstream iss(line);
                        std::getline(iss, word, ' ');

                        std::vector<int64_t> active_tet(4 ,0);

                        std::getline(iss, word, ' ');
                        active_tet[0] = std::stoi(word);
                        std::getline(iss, word, ' ');
                        active_tet[1] = std::stoi(word);
                        std::getline(iss, word, ' ');
                        active_tet[2] = std::stoi(word);
                        std::getline(iss, word, ' ');
                        active_tet[3] = std::stoi(word);

                        tetrahedra.push_back(active_tet);
                    }
                }
            }
        }
        else
            std::cout << "Error! Cannot find " << filename << std::endl;

        return;
    }
    void read_volumemesh_vtk_floatlabels_cell3dvectors(std::string filename, std::vector<std::vector<float>> &vertices, std::vector<std::vector<int64_t>> &tetrahedra,
        std::vector<float> &labels, std::vector<std::vector<float>> &vectors)
    {
        vertices.clear();
        tetrahedra.clear();
        labels.clear();
        vectors.clear();

        std::string line, word;
        std::ifstream meshfile (filename);

        int64_t n_vertices = -1;
        int64_t n_tetrahedra = -1;

        if (meshfile.is_open())
        {
            while (getline(meshfile,line))
            {
                if (line.substr(0, 6) == "POINTS")
                {
                    std::istringstream iss2(line);
                    std::getline(iss2, word, ' ');
                    std::getline(iss2, word, ' ');
                    n_vertices = std::stoi(word);

                    for (int64_t i = 0; i < n_vertices; i++)
                    {
                        getline(meshfile,line);
                        std::istringstream iss(line);

                        std::vector<float> active_vertex(3 ,0.0f);

                        std::getline(iss, word, ' ');
                        active_vertex[0] = std::stod(word);
                        std::getline(iss, word, ' ');
                        active_vertex[1] = std::stod(word);
                        std::getline(iss, word, ' ');
                        active_vertex[2] = std::stod(word);
                        vertices.push_back(active_vertex);
                    }
                }
                else if (line.substr(0, 5) == "CELLS")
                {
                    std::istringstream iss2(line);
                    std::getline(iss2, word, ' ');
                    std::getline(iss2, word, ' ');
                    n_tetrahedra = std::stoi(word);

                    for (int64_t i = 0; i < n_tetrahedra; i++)
                    {
                        getline(meshfile,line);
                        std::istringstream iss(line);
                        std::getline(iss, word, ' ');

                        std::vector<int64_t> active_tet(4 ,0);

                        std::getline(iss, word, ' ');
                        active_tet[0] = std::stoi(word);
                        std::getline(iss, word, ' ');
                        active_tet[1] = std::stoi(word);
                        std::getline(iss, word, ' ');
                        active_tet[2] = std::stoi(word);
                        std::getline(iss, word, ' ');
                        active_tet[3] = std::stoi(word);

                        tetrahedra.push_back(active_tet);
                    }
                }
                else if (line.substr(0, 9) == "CELL_DATA")
                {
                    std::istringstream iss2(line);
                    std::getline(iss2, word, ' ');
                    std::getline(iss2, word, ' ');
                    n_tetrahedra = std::stoi(word);

                    getline(meshfile,line);
                    getline(meshfile,line);
                    getline(meshfile,line);
                    std::istringstream iss(line);

                    for (int64_t i = 0; i < n_tetrahedra; i++)
                    {
                        std::getline(iss, word, ' ');
                        labels.push_back(std::stod(word));
                    }
                }
                else if (line.substr(0, 7) == "VECTORS")
                {
                    getline(meshfile,line);
                    std::istringstream iss(line);

                    for (int64_t i = 0; i < n_tetrahedra; i++)
                    {
                        std::vector<float> active_vector(3, 0.0f);
                        std::getline(iss, word, ' ');
                        active_vector[0] = std::stod(word);
                        std::getline(iss, word, ' ');
                        active_vector[1] = std::stod(word);
                        std::getline(iss, word, ' ');
                        active_vector[2] = std::stod(word);

                        vectors.push_back(active_vector);
                    }
                }
            }
        }
        else
            std::cout << "Error! Cannot find " << filename << std::endl;

        return;
    }
    meshio::Mesh read_comsol_vtu_volmesh(std::string filename, bool merge_velocitycomponents)
    {
        meshio::Mesh outmesh;

        std::string line, word;
        std::ifstream meshfile (filename);

        int64_t n_vertices = -1;
        int64_t n_tetrahedra = -1;

        if (meshfile.is_open())
        {
            while (getline(meshfile,line))
            {
                if (line.substr(0, 8) == "<Points>")
                {
                    getline(meshfile,line); //read data array line
                    n_vertices = 0;

                    while(0==0)
                    {
                        getline(meshfile,line);

                        if(line.substr(0,1) == "<")
                            break;
                        else
                        {
                            n_vertices++;

                            std::istringstream iss(line);
                            std::vector<float> active_vertex(3 ,0.0f);

                            std::getline(iss, word, ' ');
                            active_vertex[0] = std::stod(word);
                            std::getline(iss, word, ' ');
                            active_vertex[1] = std::stod(word);
                            std::getline(iss, word, ' ');
                            active_vertex[2] = std::stod(word);
                            outmesh.vertices.push_back(active_vertex);
                        }
                    }
                }
                else if (line.substr(0, 7) == "<Cells>")
                {
                    getline(meshfile,line); //read data array line
                    n_tetrahedra = 0;
                    while (0 == 0)
                    {
                        getline(meshfile,line);

                        if(line.substr(0,1) == "<")
                            break;
                        else
                        {
                            n_tetrahedra++;

                            std::istringstream iss(line);

                            std::vector<int64_t> active_tet(4 ,0);

                            std::getline(iss, word, ' ');
                            active_tet[0] = std::stoi(word);
                            std::getline(iss, word, ' ');
                            active_tet[1] = std::stoi(word);
                            std::getline(iss, word, ' ');
                            active_tet[2] = std::stoi(word);
                            std::getline(iss, word, ' ');
                            active_tet[3] = std::stoi(word);

                            outmesh.tetrahedra.push_back(active_tet);
                        }
                    }
                }
                else if (line.substr(0, 11) == "<PointData>")
                {
                    while (0 == 0)
                    {
                        getline(meshfile,line);
                        if (line.substr(0, 10) == "<DataArray")
                        {
                            int namepos = line.find("Name=");
                            int typepos = line.find("type=");
                            int formatpos = line.find(" Format=");

                            std::string namestring = line.substr(namepos+6,formatpos-namepos-7);
                            std::cout << namestring << std::endl;

                            if(merge_velocitycomponents && (namestring == "Velocity_field,_x_component" || namestring == "Velocity_field,_y_component"
                                || namestring == "Velocity_field,_z_component"))
                            {
                                std::cout << "merging component to vector: " << namestring << std::endl;
                                bool first_component = true;
                                int vectorpos = 0;

                                for (int i = 0; i < outmesh.vertex_vectors_name.size(); i++)
                                {
                                    if (outmesh.vertex_vectors_name[i] == "velocity_field")
                                    {
                                        first_component = false;
                                        vectorpos = i;
                                        break;
                                    }
                                }
                                if (first_component) {
                                    std::vector<std::vector<float>> dummy;
                                    outmesh.vertex_vectors_name.push_back("velocity_field");
                                    outmesh.vertex_vectors.push_back(dummy);
                                    long long int counter = 0;

                                    while(0==0)
                                    {
                                        getline(meshfile,line);

                                        if(line.substr(0,1) == "<")
                                            break;
                                        else
                                        {
                                            if (namestring == "Velocity_field,_x_component")
                                                outmesh.vertex_vectors[outmesh.vertex_vectors.size()-1].push_back({std::stod(line), 0.0, 0.0});
                                            else if (namestring == "Velocity_field,_y_component")
                                                outmesh.vertex_vectors[outmesh.vertex_vectors.size()-1].push_back({0.0, std::stod(line), 0.0});
                                            else if (namestring == "Velocity_field,_z_component")
                                                outmesh.vertex_vectors[outmesh.vertex_vectors.size()-1].push_back({0.0, 0.0, std::stod(line)});
                                            counter++;
                                        }

                                    }
                                }
                                else {
                                    long long int counter = 0;

                                    while(0==0)
                                    {
                                        getline(meshfile,line);

                                        if(line.substr(0,1) == "<")
                                            break;
                                        else
                                        {
                                            if (namestring == "Velocity_field,_x_component")
                                                outmesh.vertex_vectors[vectorpos][counter][0] = std::stod(line);
                                            else if (namestring == "Velocity_field,_y_component")
                                                outmesh.vertex_vectors[vectorpos][counter][1] = std::stod(line);
                                            else if (namestring == "Velocity_field,_z_component")
                                                outmesh.vertex_vectors[vectorpos][counter][2] = std::stod(line);
                                            counter++;
                                        }
                                    }
                                }
                            }
                            else if(merge_velocitycomponents && (namestring == "Vorticity_field,_x_component" || namestring == "Vorticity_field,_y_component"
                                || namestring == "Vorticity_field,_z_component"))
                            {
                                std::cout << "merging component to vector: " << namestring << std::endl;
                                bool first_component = true;
                                int vectorpos = 0;

                                for (int i = 0; i < outmesh.vertex_vectors_name.size(); i++)
                                {
                                    if (outmesh.vertex_vectors_name[i] == "vorticity_field")
                                    {
                                        first_component = false;
                                        vectorpos = i;
                                        break;
                                    }
                                }
                                if (first_component) {
                                    std::vector<std::vector<float>> dummy;
                                    outmesh.vertex_vectors_name.push_back("vorticity_field");
                                    outmesh.vertex_vectors.push_back(dummy);
                                    long long int counter = 0;

                                    while(0==0)
                                    {
                                        getline(meshfile,line);

                                        if(line.substr(0,1) == "<")
                                            break;
                                        else
                                        {
                                            if (namestring == "Vorticity_field,_x_component")
                                                outmesh.vertex_vectors[outmesh.vertex_vectors.size()-1].push_back({std::stod(line), 0.0, 0.0});
                                            else if (namestring == "Vorticity_field,_y_component")
                                                outmesh.vertex_vectors[outmesh.vertex_vectors.size()-1].push_back({0.0, std::stod(line), 0.0});
                                            else if (namestring == "Vorticity_field,_z_component")
                                                outmesh.vertex_vectors[outmesh.vertex_vectors.size()-1].push_back({0.0, 0.0, std::stod(line)});
                                            counter++;
                                        }

                                    }
                                }
                                else {
                                    long long int counter = 0;

                                    while(0==0)
                                    {
                                        getline(meshfile,line);

                                        if(line.substr(0,1) == "<")
                                            break;
                                        else
                                        {
                                            if (namestring == "Vorticity_field,_x_component")
                                                outmesh.vertex_vectors[vectorpos][counter][0] = std::stod(line);
                                            else if (namestring == "Vorticity_field,_y_component")
                                                outmesh.vertex_vectors[vectorpos][counter][1] = std::stod(line);
                                            else if (namestring == "Vorticity_field,_z_component")
                                                outmesh.vertex_vectors[vectorpos][counter][2] = std::stod(line);
                                            counter++;
                                        }
                                    }
                                }
                            }
                            else if (line.substr(typepos+6,7) == "Float64")
                            {
                                std::cout << "adding scalar: " << namestring << std::endl;

                                std::vector<float> dummy;
                                outmesh.vertex_labels_floating.push_back(dummy);
                                outmesh.vertex_labels_floating_name.push_back(namestring);

                                while(0==0)
                                {
                                    getline(meshfile,line);

                                    if(line.substr(0,1) == "<")
                                        break;
                                    else
                                        outmesh.vertex_labels_floating[outmesh.vertex_labels_floating.size()-1].push_back(std::stod(line));
                                }
                            }
                            else
                            {
                                std::cout << "Error! DataArray type not implemented in read_comsol_vtu_volmesh: " << std::endl;
                                std::cout << line << std::endl;
                            }

                        }

                        if (line.substr(0, 12) == "</PointData>")
                            break;
                    }
                }
                else if (line.substr(0, 1) == "<" && line.substr(0, 2) != "</")
                {
                    if(line.substr(0,5) == "<?xml") continue;
                    if(line.substr(0,8) == "<VTKFile") continue;
                    if(line.substr(0,13) == "<Unstructured") continue;
                    if(line.substr(0,6) == "<Piece") continue;
                    std::cout << "ignoring: " << line << std::endl;
                }
            }
        }
        else
            std::cout << "Error! Cannot find " << filename << std::endl;

        outmesh.n_vertices = n_vertices;
        outmesh.n_tetrahedra = n_tetrahedra;

        std::cout << "found " << n_vertices << " vertices and " << n_tetrahedra << " tetrahedra" << std::endl;

        return outmesh;
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //Simplification
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void remove_unusedvertices(std::vector<std::vector<float>> &vertices, std::vector<std::vector<int64_t>> &tetrahedra)
    {
        std::vector<int64_t> new_idx(vertices.size(), 0);
        std::vector<bool> used_vertex(vertices.size(), false);
        std::vector<std::vector<float>> new_vertices;

        for (uint64_t idx = 0; idx < tetrahedra.size(); idx++)
        {
            used_vertex[tetrahedra[idx][0]] = true;
            used_vertex[tetrahedra[idx][1]] = true;
            used_vertex[tetrahedra[idx][2]] = true;
            used_vertex[tetrahedra[idx][3]] = true;
        }
        uint64_t active_idx = 0;

        for (uint64_t idx = 0; idx < vertices.size(); idx++)
        {
            if(used_vertex[idx]){
                new_idx[idx] = active_idx;
                active_idx++;
                new_vertices.push_back(vertices[idx]);
            }
        }

        #pragma omp parallel for
        for (uint64_t idx = 0; idx < tetrahedra.size(); idx++)
        {
            tetrahedra[idx][0] = new_idx[tetrahedra[idx][0]];
            tetrahedra[idx][1] = new_idx[tetrahedra[idx][1]];
            tetrahedra[idx][2] = new_idx[tetrahedra[idx][2]];
            tetrahedra[idx][3] = new_idx[tetrahedra[idx][3]];
        }
        vertices = new_vertices;
        return;
    }
    void remove_unusedvertices_surfacemesh(std::vector<std::vector<float>> &vertices, std::vector<std::vector<int64_t>> &triangles)
    {
        std::vector<int64_t> new_idx(vertices.size(), 0);
        std::vector<bool> used_vertex(vertices.size(), false);
        std::vector<std::vector<float>> new_vertices;

        for (uint64_t idx = 0; idx < triangles.size(); idx++)
        {
            used_vertex[triangles[idx][0]] = true;
            used_vertex[triangles[idx][1]] = true;
            used_vertex[triangles[idx][2]] = true;
        }
        uint64_t active_idx = 0;

        for (uint64_t idx = 0; idx < vertices.size(); idx++)
        {
            if(used_vertex[idx]){
                new_idx[idx] = active_idx;
                active_idx++;
                new_vertices.push_back(vertices[idx]);
            }
        }

        #pragma omp parallel for
        for (uint64_t idx = 0; idx < triangles.size(); idx++)
        {
            triangles[idx][0] = new_idx[triangles[idx][0]];
            triangles[idx][1] = new_idx[triangles[idx][1]];
            triangles[idx][2] = new_idx[triangles[idx][2]];
        }
        vertices = new_vertices;
        return;
    }
    void remove_unusedvertices_surfacemesh(meshio::Mesh &mesh)
    {
        std::vector<int> dummy0; std::vector<float> dummy1;
        std::vector<std::vector<int>> new_discrete_labels(mesh.vertex_labels_discrete.size(), dummy0);
        std::vector<std::vector<float>> new_floating_labels(mesh.vertex_labels_floating.size(), dummy1);
        std::vector<std::vector<std::vector<float>>> new_vertex_vectors(mesh.vertex_vectors.size(), new_floating_labels);

        std::vector<int64_t> new_idx(mesh.vertices.size(), 0);
        std::vector<bool> used_vertex(mesh.vertices.size(), false);
        std::vector<std::vector<float>> new_vertices;

        for (uint64_t idx = 0; idx < mesh.triangles.size(); idx++)
        {
            used_vertex[mesh.triangles[idx][0]] = true;
            used_vertex[mesh.triangles[idx][1]] = true;
            used_vertex[mesh.triangles[idx][2]] = true;
        }
        uint64_t active_idx = 0;

        for (uint64_t idx = 0; idx < mesh.vertices.size(); idx++)
        {
            if(used_vertex[idx])
            {
                new_idx[idx] = active_idx;
                active_idx++;
                new_vertices.push_back(mesh.vertices[idx]);

                for (int i = 0; i < mesh.vertex_labels_discrete.size(); i++)
                    new_discrete_labels[i].push_back(mesh.vertex_labels_discrete[i][idx]);
                for (int i = 0; i < mesh.vertex_labels_floating.size(); i++)
                    new_floating_labels[i].push_back(mesh.vertex_labels_floating[i][idx]);
                for (int i = 0; i < mesh.vertex_vectors.size(); i++)
                    new_vertex_vectors[i].push_back(mesh.vertex_vectors[i][idx]);
            }
        }

        #pragma omp parallel for
        for (uint64_t idx = 0; idx < mesh.triangles.size(); idx++)
        {
            mesh.triangles[idx][0] = new_idx[mesh.triangles[idx][0]];
            mesh.triangles[idx][1] = new_idx[mesh.triangles[idx][1]];
            mesh.triangles[idx][2] = new_idx[mesh.triangles[idx][2]];
        }

        //apply new values
        mesh.vertices = new_vertices;
        mesh.vertex_labels_discrete = new_discrete_labels;
        mesh.vertex_labels_floating = new_floating_labels;
        mesh.vertex_vectors = new_vertex_vectors;
        return;
    }
    void map_tetlabels2trilabels(Mesh &surfmesh, std::vector<int64_t> &tet_idx)
    {
        for (int i = 0; i < surfmesh.cell_labels_floating.size(); i++)
        {
            std::vector<labeltype_floating> new_mapping(tet_idx.size(), 0.0);

            #pragma omp parallel for
            for (long long int idx = 0; idx < tet_idx.size(); idx++)
                new_mapping[idx] = surfmesh.cell_labels_floating[i][tet_idx[idx]];

            surfmesh.cell_labels_floating[i] = new_mapping;
        }
        for (int i = 0; i < surfmesh.cell_labels_discrete.size(); i++)
        {
            std::vector<labeltype_discrete> new_mapping(tet_idx.size(), 0.0);

            #pragma omp parallel for
            for (long long int idx = 0; idx < tet_idx.size(); idx++)
                new_mapping[idx] = surfmesh.cell_labels_discrete[i][tet_idx[idx]];

            surfmesh.cell_labels_discrete[i] = new_mapping;
        }
        for (int i = 0; i < surfmesh.cell_vectors.size(); i++)
        {
            std::vector<std::vector<labeltype_floating>> new_mapping(tet_idx.size(), {0.0,0.0,0.0});

            for (long long int idx = 0; idx < tet_idx.size(); idx++)
                new_mapping[idx] = surfmesh.cell_vectors[i][tet_idx[idx]];

            surfmesh.cell_vectors[i] = new_mapping;
        }
        return;
    }
    void create_subset_from_label(uint8_t active_label, std::vector<std::vector<float>> &vertices, std::vector<std::vector<int64_t>> &tetrahedra, std::vector<uint8_t> &labels,
                                std::vector<std::vector<float>> &out_vertices, std::vector<std::vector<int64_t>> &out_tetrahedra)
    {
        out_vertices.clear();
        out_tetrahedra.clear();

        std::vector<int64_t> new_idx(vertices.size(), 0);
        std::vector<bool> used_vertex(vertices.size(), false);
        std::vector<std::vector<float>> new_vertices;

        for (long long int idx = 0; idx < tetrahedra.size(); idx++)
        {
            if(labels[idx] == active_label)
            {
                used_vertex[tetrahedra[idx][0]] = true;
                used_vertex[tetrahedra[idx][1]] = true;
                used_vertex[tetrahedra[idx][2]] = true;
                used_vertex[tetrahedra[idx][3]] = true;
            }
        }
        long long int active_idx = 0;
        for (long long int idx = 0; idx < vertices.size(); idx++)
        {
            if(used_vertex[idx]){
                new_idx[idx] = active_idx;
                active_idx++;
                out_vertices.push_back(vertices[idx]);
            }
        }
        for (long long int idx = 0; idx < tetrahedra.size(); idx++)
        {
            if(labels[idx] == active_label)
                out_tetrahedra.push_back({new_idx[tetrahedra[idx][0]], new_idx[tetrahedra[idx][1]], new_idx[tetrahedra[idx][2]], new_idx[tetrahedra[idx][3]]});
        }
        return;
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //Auxiliary
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T> int count_uniquelabels(std::vector<T> &labels){
        std::vector<int64_t> counts;

        for (int64_t i = 0; i < labels.size(); i++)
        {
            T value = labels[i];
            if (value >= counts.size()) counts.resize(value+1);
            counts[value]++;
        }

        int n_unique = 0;
        for (int64_t i = 0; i < counts.size(); i++)
            if (counts[i] > 0) n_unique++;

        return n_unique;
    }
    template int count_uniquelabels<int>(std::vector<int> &labels);
    template int count_uniquelabels<uint8_t>(std::vector<uint8_t> &labels);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
}


