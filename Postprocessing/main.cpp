#include <iostream>
#include <omp.h>
#include <fstream>

#include "Geometry/hdcommunication.h"
#include "Geometry/meshio.h"
#include "Geometry/triangles.h"
#include "Geometry/taubin_smoothing.h"
#include "Geometry/auxiliary.h"
#include "Geometry/filtering.h"
#include "Geometry/rigid_body_motion.h"

#include "Derivatives/displacement_derivatives.h"

using namespace std;

int main(int argc, char* argv[])
{

    cout << "entering voxel2mesh mapper (basic)" << endl;
    cout << "----------------------------------------------------------" << endl;

    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    string inpath_mesh = "dummy";
    string inpath_displacements = "dummy";

    bool convert2physical = false;
    float vxlsize = 1.00; //a multiplier to scale your mesh and displacements to physical dimensions

    bool taubin_smoothing = false; //allows smoothing the surface of your mesh

    //In this version there is only Gaussian blurring available.
    //It is recommended to apply some king of blur for introducing ca correlation length to the data.
    //This is required to have a length scale for strains that otherwise do not make sense.
    bool blur_displacements = true;
    float blur_sigma = 3.0;

    bool map_cell_vectors = true;
    bool map_point_vectors = false;

    //Add the desired outputs to this vector and give them a name in the next vector (identical in the mini version)
    //Mini version allows for:
    //      Components of the Green-Lagrange strain tensor: Exx,Eyy,Ezz,Exy,Eyz,Exz
    //      Volumetric and maximum principal strain: volstrain, principal_strain_max, maxshear
    std::vector<string> grayscale_cell_labels = {};//  {"volstrain","Ezz",...};
    std::vector<string> grayscale_cell_ids ={};//  {"volstrain","Ezz",...};

    //Minimize rigid body motion to have deformations instead of displacements.
    //In this version motion elimination can only be performed on meshes.
    //Either on the entire volume or across the entire surface.
    bool eliminate_rigidbodymotions = true;
    string motionmask = "mesh_surface"; //"mesh" or "mesh_surface"
    uint8_t motionlabel = 0;
	
    //Basic text output for mapped surface and volume values
    bool print_meanvalues = true;
    bool print_meanvalues_surface = true;
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////
    if ("extract command line arguments)")
    {
        for (uint16_t i = 1; i < argc; i++)
        {
            std::string argument = std::string(argv[i]);

            if (argument == "-i_mesh" || argument == "-i"){
                i++;
                inpath_mesh = std::string(argv[i]);
            }
            else if (argument == "-i_disp")
            {
                i++;
                inpath_displacements = std::string(argv[i]);
                map_cell_vectors = true;
            }
            else if (argument == "-sigma"){
                i++;
                blur_sigma = std::atof(argv[i]);
            }
	    else if (argument == "-vxl"){
                i++;
                vxlsize = std::atof(argv[i]);
		convert2physical=true;
            }
            else if (argument == "--taubin")
                taubin_smoothing = true;
            else if (argument == "--vertices")
            {
                //map the displacements on the vertices instead of the cells CoG
                map_cell_vectors = false;
                map_point_vectors = true;
            }

            //derived outputs
            else if (argument == "Exx" || argument == "Eyy" || argument == "Ezz" || argument == "Exy" || argument == "Eyz" || argument == "Exz" ||
                     argument == "volstrain" || argument == "maxshear")
            {
                grayscale_cell_labels.push_back(argument);
                grayscale_cell_ids.push_back(argument);
            }
            else if (argument == "max_pstrain" || argument == "principal_strain_max")
            {
                grayscale_cell_labels.push_back("principal_strain_max");
                grayscale_cell_ids.push_back("max_principal_strain");
            }
        }
    }

    if (inpath_mesh == "dummy")
    {
        cout << "Error! No mesh with -i provided!" << endl;
        return -1;
    }
    //////////////////////////////////////////////////////////////////////////////////////

    //Manage directories
    //////////////////////////////////////////////////////////////////////////////////////
    hdcom::HdCommunication hdcom;
    std::string active_path = aux::get_active_directory();

    //test if provided path is absolute or relative
    struct stat buffer;   
    if ((stat (inpath_mesh.c_str(), &buffer) != 0) && (stat ((active_path+"//"+inpath_mesh).c_str(), &buffer) == 0))
	inpath_mesh = active_path+"//"+inpath_mesh;	
    if (inpath_displacements != "dummy" && !hdcom.is_absolute_path(inpath_displacements) && hdcom.path_exists(active_path+"//"+inpath_displacements))
        inpath_displacements = active_path+"//"+inpath_displacements;
    cout << "mesh_file: " << inpath_mesh << endl;
    cout << "displacements: " << inpath_displacements << endl;

    string outpath = inpath_mesh.substr(0, inpath_mesh.rfind("/")+1);
    string outname = inpath_mesh.substr(inpath_mesh.rfind("/")+1, inpath_mesh.rfind(".")-(inpath_mesh.rfind("/")+1));

    cout << "output directory: " << outpath << endl;
    cout << "------------------------" << endl;
    //////////////////////////////////////////////////////////////////////////////////////

    //Acquire Mesh (omit all previous labels)
    //////////////////////////////////////////////////////////////////////////////////////
    meshio::Mesh mesh;
    int shape[3];

    if ("acquire mesh without labels"){
        bool delete_phase = true;
        if (inpath_mesh.substr(inpath_mesh.size()-5,5) == ".mesh")
        {
            mesh = meshio::read_meditmesh_labeled(inpath_mesh);
            cout << "read " << mesh.vertices.size() << " mesh vertices and " << mesh.tetrahedra.size() << " tetrahedra" << endl;
        }
        else if (inpath_mesh.substr(inpath_mesh.size()-4,4) == ".vtk" || inpath_mesh.substr(inpath_mesh.size()-4,4) == ".vtu")
        {
            mesh = meshio::read_unlabeled_vtk_mesh(inpath_mesh);
            cout << "read " << mesh.vertices.size() << " mesh vertices and " << mesh.tetrahedra.size() << " tetrahedra" << endl;
        }
        else if (inpath_mesh.substr(inpath_mesh.size()-4,4) == ".off")
        {
            cout << "read off-file vertices and triangles but basic mapper does not support surface meshes" << endl;
            mesh = meshio::read_off(inpath_mesh);
            return -2;
        }
        else
        {
            cout << "only supporting Medit Volume Mesh!" << endl;
            return 1;
        }
        if (delete_phase)
        {
            mesh.cell_labels_discrete.clear();
            mesh.cell_labels_discrete_name.clear();
        }
    }
    //////////////////////////////////////////////////////////////////////////////////////

    //Displacement Vectors
    //////////////////////////////////////////////////////////////////////////////////////
    float *ux; float* uy; float* uz;

    if (inpath_displacements != "dummy")
    {
        //Read in image sequences with DVC result
        ///////////////////////////////////////////////////////////////////////////////
        cout << "reading in displacement vectors:" << endl;
        ux = hdcom.GetTif_unknowndim_32bit(inpath_displacements+"/dx/", shape, true);
        uy = hdcom.GetTif_unknowndim_32bit(inpath_displacements+"/dy/", shape, true);
        uz = hdcom.GetTif_unknowndim_32bit(inpath_displacements+"/dz/", shape, true);
        long long int nslice = shape[0]*shape[1]; long long int nstack = shape[2]*nslice;
        ///////////////////////////////////////////////////////////////////////////////

	cout << "------------------------" << endl;

        //Blur the displacements to create a correlation length for the strain
        ///////////////////////////////////////////////////////////////////////////////
        if (blur_displacements)
        {
            cout << "blurring displacement field with " << blur_sigma << " sigma...";
            cout.flush();
            filter::apply_3DGaussianFilter2Vector(ux, uy, uz, shape, blur_sigma);
            cout << "done" << endl;
        }
        ///////////////////////////////////////////////////////////////////////////////

        //Minimize rigid body motion
        ///////////////////////////////////////////////////////////////////////////////
        if (map_cell_vectors && mesh.type == "volume"){
            if (!eliminate_rigidbodymotions) mesh.cell_vectors_name.push_back("cell_displacements");
            else mesh.cell_vectors_name.push_back("cell_deformations");

            std::vector<std::vector<float>> center_of_gravity = aux::get_cell_centerofgravity(mesh.vertices, mesh.tetrahedra);

            mesh.cell_vectors.push_back(aux::linearinterpolate_coordinatevector(center_of_gravity, ux, uy, uz, shape));

            if (eliminate_rigidbodymotions && motionmask == "mesh")
                rbmotion::eliminate_rigidbody_motion(mesh.cell_vectors[mesh.cell_vectors.size()-1], center_of_gravity, mesh.cell_labels_discrete[0], shape, motionlabel, 0.1, 2.);
            else if (eliminate_rigidbodymotions && motionmask == "mesh_surface")
            {
                vector<vector<int64_t>> surface_triangles = surfacemesh_compare::extract_exterior_surface_normalized(mesh.vertices, mesh.tetrahedra, mesh.cell_labels_discrete[0], motionlabel, false);
                std::vector<std::vector<float>> center_of_gravity2 = aux::get_triangle_centerofgravity(mesh.vertices, surface_triangles);
                std::vector<std::vector<float>> displacements2 = aux::linearinterpolate_coordinatevector(center_of_gravity2, ux, uy, uz, shape);
                rbmotion::eliminate_rigidbody_motion(displacements2, center_of_gravity2, mesh.cell_vectors[mesh.cell_vectors.size()-1], center_of_gravity, shape, 0.1, 2.);
            }

            if (eliminate_rigidbodymotions) outname += "_deform";
            else outname += "_displ";
            ///////////////////////////////////////////////////////////////////////////////
        }
        else if (map_point_vectors){
            if (!eliminate_rigidbodymotions) mesh.vertex_vectors_name.push_back("point_displacements");
            else mesh.vertex_vectors_name.push_back("point_deformations");

            mesh.vertex_vectors.push_back(aux::linearinterpolate_coordinatevector(mesh.vertices, ux, uy, uz, shape));

            if (eliminate_rigidbodymotions && motionmask == "mesh")
                rbmotion::eliminate_rigidbody_motion(mesh.vertex_vectors[mesh.vertex_vectors.size()-1], mesh.vertices, mesh.cell_labels_discrete[0], shape, motionlabel, 0.1, 2.);
            else if (eliminate_rigidbodymotions && motionmask == "mesh_surface")
            {
                vector<vector<int64_t>> surface_triangles = surfacemesh_compare::extract_exterior_surface_normalized(mesh.vertices, mesh.tetrahedra, mesh.cell_labels_discrete[0], motionlabel, false);
                std::vector<std::vector<float>> center_of_gravity2 = aux::get_triangle_centerofgravity(mesh.vertices, surface_triangles);
                std::vector<std::vector<float>> displacements2 = aux::linearinterpolate_coordinatevector(center_of_gravity2, ux, uy, uz, shape);
                rbmotion::eliminate_rigidbody_motion(displacements2, center_of_gravity2, mesh.vertex_vectors[mesh.vertex_vectors.size()-1], mesh.vertices, shape, 0.1, 2.);
            }

            if (eliminate_rigidbodymotions) outname += "_pdeform";
            else outname += "_pdispl";
        }
        ///////////////////////////////////////////////////////////////////////////////
    }
    //////////////////////////////////////////////////////////////////////////////////////

    //Scale to voxel size
    //////////////////////////////////////////////////////////////////////////////////////
    if (convert2physical){
        cout << "scaling to voxel size of " << vxlsize << endl;
        for (int i = 0; i < mesh.vertices.size(); i++)
        {
            mesh.vertices[i][0] *= vxlsize;
            mesh.vertices[i][1] *= vxlsize;
            mesh.vertices[i][2] *= vxlsize;
        }
        for (int i = 0; i < mesh.cell_vectors.size(); i++)
            for (long long int idx = 0; idx < mesh.cell_vectors[i].size(); idx++){
                mesh.cell_vectors[i][idx][0] *= vxlsize;
                mesh.cell_vectors[i][idx][1] *= vxlsize;
                mesh.cell_vectors[i][idx][2] *= vxlsize;}
        for (int i = 0; i < mesh.vertex_vectors.size(); i++)
            for (long long int idx = 0; idx < mesh.vertex_vectors[i].size(); idx++){
                mesh.vertex_vectors[i][idx][0] *= vxlsize;
                mesh.vertex_vectors[i][idx][1] *= vxlsize;
                mesh.vertex_vectors[i][idx][2] *= vxlsize;}
    }
    //////////////////////////////////////////////////////////////////////////////////////

    //Grayscale labels on cells
    //////////////////////////////////////////////////////////////////////////////////////
    if(mesh.type == "volume" && grayscale_cell_labels.size() > 0)
    {
        std::vector<std::vector<float>> center_of_gravity = aux::get_cell_centerofgravity(mesh.vertices, mesh.tetrahedra);

        for (int i = 0; i < grayscale_cell_labels.size(); i++)
        {
            float *labelimage;

                 if (grayscale_cell_labels[i] == "Ezz") labelimage = derive::calc_from_green_strain(ux, uy, uz, shape, "Ezz", "Farid");
            else if (grayscale_cell_labels[i] == "Eyy") labelimage = derive::calc_from_green_strain(ux, uy, uz, shape, "Eyy", "Farid");
            else if (grayscale_cell_labels[i] == "Exx") labelimage = derive::calc_from_green_strain(ux, uy, uz, shape, "Exx", "Farid");
            else if (grayscale_cell_labels[i] == "Exy") labelimage = derive::calc_from_green_strain(ux, uy, uz, shape, "Exy", "Farid");
            else if (grayscale_cell_labels[i] == "Exz") labelimage = derive::calc_from_green_strain(ux, uy, uz, shape, "Exz", "Farid");
            else if (grayscale_cell_labels[i] == "Eyz") labelimage = derive::calc_from_green_strain(ux, uy, uz, shape, "Eyz", "Farid");
            else if (grayscale_cell_labels[i] == "maxshear") labelimage = derive::calc_from_green_strain(ux, uy, uz, shape, "maxshear", "Farid");
            else if (grayscale_cell_labels[i] == "principal_strain_max") labelimage = derive::calc_from_green_strain(ux, uy, uz, shape, "principal_strain_max", "Farid");
            else if (grayscale_cell_labels[i] == "volstrain") labelimage = derive::calc_from_green_strain(ux, uy, uz, shape, "volstrain", "Farid");

            std::vector<float> grayscale_labels = aux::linearinterpolate_coordinatevector(center_of_gravity, labelimage, shape);

            mesh.cell_labels_floating_name.push_back(grayscale_cell_ids[i]);
            mesh.cell_labels_floating.push_back(grayscale_labels);

            outname += "_"+grayscale_cell_ids[i];

            free(labelimage);
        }
    }
    //////////////////////////////////////////////////////////////////////////////////////
	
    //Mean values in tetrahedra labels
    //////////////////////////////////////////////////////////////////////////////////////
    if (mesh.cell_labels_floating.size() > 0 && (print_meanvalues || print_meanvalues_surface))
    {
        std::cout << "-----------------------------" << std::endl;
        if (print_meanvalues)
        {
            for (int p = 0; p < mesh.cell_labels_floating.size(); p++)
            {
                double mean = 0.0;
                double stddev = 0.0;

                for (long long int i = 0; i < mesh.cell_labels_floating[p].size(); i++)
                    mean += mesh.cell_labels_floating[p][i];
                mean /= mesh.cell_labels_floating[p].size();
                for (long long int i = 0; i < mesh.cell_labels_floating[p].size(); i++)
                    stddev += (mesh.cell_labels_floating[p][i]-mean)*(mesh.cell_labels_floating[p][i]-mean);

                stddev = sqrt(stddev/(mesh.cell_labels_floating[p].size()-1));
                cout << "mean " << mesh.cell_labels_floating_name[p] << ": " << mean << " (std: " << stddev << ")"<< endl;
            }
            std::cout << "-----------------------------" << std::endl;
        }
        if (print_meanvalues_surface)
        {
            std::vector<std::vector<int64_t>> exterior_surface = surfacemesh_compare::extract_exterior_surface_indexed(mesh.tetrahedra);
            
            double N = 0.0;
            double mean = 0.0;
            double stddev = 0.0;

	    for (int p = 0; p < mesh.cell_labels_floating.size(); p++)
            {
	        for (int q = 0; q < exterior_surface.size(); q++)
                {
                    int i = exterior_surface[q][3];
                    mean += mesh.cell_labels_floating[p][i];
                    N++;
                }
                mean /= N;
                for (int q = 0; q < exterior_surface.size(); q++)
                {
                    int i = exterior_surface[q][3];
                    stddev += (mesh.cell_labels_floating[p][i]-mean)*(mesh.cell_labels_floating[p][i]-mean);
		}
                stddev = sqrt(stddev/(N-1));

                cout << "surface mean " << mesh.cell_labels_floating_name[p] << ": " << mean << " (std: " << stddev << ")"<< endl;
            }
            std::cout << "-----------------------------" << std::endl;
        }
    }
    //////////////////////////////////////////////////////////////////////////////////////
	
    //Surface smoothing
    //////////////////////////////////////////////////////////////////////////////////////
    if(taubin_smoothing){
        cout << "smoothing surface with taubin filter" << endl;
        std::vector<std::vector<int64_t>> exterior_surface;

        if (mesh.type == "volume") exterior_surface= surfacemesh_compare::extract_exterior_surface(mesh.tetrahedra);
        else exterior_surface = mesh.triangles;

        mesh_filter::TaubinFilter taubin;
        taubin.iterations = 10; taubin.lmbda = 0.5; taubin.mu = -0.53;
        taubin.run_areafilter(mesh.vertices,exterior_surface);

        outname += "_taubin";
    }
    //////////////////////////////////////////////////////////////////////////////////////

    //Export mesh
    //////////////////////////////////////////////////////////////////////////////////////
    string outname2 = inpath_mesh.substr(inpath_mesh.rfind("/")+1, inpath_mesh.rfind(".")-(inpath_mesh.rfind("/")+1));
    if (outname == outname2) outname += "_remapped";

    cout << "exporting to " << outname << endl;
    meshio::save_mesh2vtk(outpath, outname, mesh);
    //////////////////////////////////////////////////////////////////////////////////////

    cout << "----------------------------------------------------------" << endl;
    return 0;
}
