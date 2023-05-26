#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <math.h>

#include <libgen.h>
#include <unistd.h>
#include <linux/limits.h>

#include "filtering.h"

namespace aux
{
    /*String-Manipulation
    *********************************************************/
    std::string zfill_int2string(int inint, const unsigned int &zfill)
    {
        std::string outstring = std::to_string(inint);
        while(outstring.length() < zfill)
            outstring = "0" + outstring;
        return outstring;
    }
    std::string get_active_directory()
    {
	char result[PATH_MAX];
	ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
        const char *tmppath;
        if (count != -1) {
            tmppath = dirname(result);
        }
        return std::string(tmppath);
    }

    /*Numpy-like
    *********************************************************/
    std::vector<double> linspace(double startval, double endval, uint64_t bins)
    {
        std::vector<double> linspaced(bins);
        double delta = (endval-startval)/(bins-1);
        for(uint64_t i = 0; i < (bins-1); i++)
        {
            linspaced[i] = startval + delta * i;
        }
        linspaced[bins-1] = endval;
        return linspaced;
    }

    /*Application specific
    *********************************************************/
    float linearinterpolate_coordinate(float x, float y, float z, float *image, int shape[3])
    {
        int xf = std::max(0, std::min(shape[0]-1, (int) floor(x))); int xc = std::max(0, std::min(shape[0]-1, (int) ceil(x))); float wx = x-xf;
        int yf = std::max(0, std::min(shape[1]-1, (int) floor(y))); int yc = std::max(0, std::min(shape[1]-1, (int) ceil(y))); float wy = y-yf;
        int zf = std::max(0, std::min(shape[2]-1, (int) floor(z))); int zc = std::max(0, std::min(shape[2]-1, (int) ceil(z))); float wz = z-zf;

        int nx = shape[0];
        long long int nslice = shape[0]*shape[1];

        float val000 = image[zf*nslice + yf*nx + xf]; float val100 = image[zf*nslice + yf*nx + xc];
        float val010 = image[zf*nslice + yc*nx + xf]; float val110 = image[zf*nslice + yc*nx + xc];
        float val001 = image[zc*nslice + yf*nx + xf]; float val101 = image[zc*nslice + yf*nx + xc];
        float val011 = image[zc*nslice + yc*nx + xf]; float val111 = image[zc*nslice + yc*nx + xc];

        float outval0 = (1.f-wx)*val000 + wx*val100;
        float outval1 = (1.f-wx)*val010 + wx*val110;
        float outval2 = (1.f-wx)*val001 + wx*val101;
        float outval3 = (1.f-wx)*val011 + wx*val111;

        outval0 = (1.f-wy)*outval0 + wy*outval1;
        outval1 = (1.f-wy)*outval2 + wy*outval3;

        return (1.f-wz)*outval0 + wz*outval1;
    }
    float maxval_at_coordinate(float x, float y, float z, float *image, int shape[3])
    {
        int xf = std::max(0, std::min(shape[0]-1, (int) floor(x))); int xc = std::max(0, std::min(shape[0]-1, (int) ceil(x))); float wx = x-xf;
        int yf = std::max(0, std::min(shape[1]-1, (int) floor(y))); int yc = std::max(0, std::min(shape[1]-1, (int) ceil(y))); float wy = y-yf;
        int zf = std::max(0, std::min(shape[2]-1, (int) floor(z))); int zc = std::max(0, std::min(shape[2]-1, (int) ceil(z))); float wz = z-zf;

        int nx = shape[0];
        long long int nslice = shape[0]*shape[1];

        float val000 = image[zf*nslice + yf*nx + xf];
        val000 = std::max(image[zf*nslice + yf*nx + xc], val000);
        val000 = std::max(image[zf*nslice + yc*nx + xf], val000);
        val000 = std::max(image[zf*nslice + yc*nx + xc], val000);
        val000 = std::max(image[zc*nslice + yf*nx + xf], val000);
        val000 = std::max(image[zc*nslice + yf*nx + xc], val000);
        val000 = std::max(image[zc*nslice + yc*nx + xf], val000);
        val000 = std::max(image[zc*nslice + yc*nx + xc], val000);

        return val000;
    }
    float linearinterpolate_coordinate2D(float x, float y, float *image, int shape[2])
    {
        int xf = std::max(0, std::min(shape[0]-1, (int) floor(x))); int xc = std::max(0, std::min(shape[0]-1, (int) ceil(x))); float wx = x-xf;
        int yf = std::max(0, std::min(shape[1]-1, (int) floor(y))); int yc = std::max(0, std::min(shape[1]-1, (int) ceil(y))); float wy = y-yf;

        int nx = shape[0];
        long long int nslice = shape[0]*shape[1];

        float val00 = image[yf*nx + xf]; float val10 = image[yf*nx + xc];
        float val01 = image[yc*nx + xf]; float val11 = image[yc*nx + xc];

        float outval0 = (1.f-wx)*val00 + wx*val10;
        float outval1 = (1.f-wx)*val01 + wx*val11;

        outval0 = (1.f-wy)*outval0 + wy*outval1;

        return outval0;
    }
    std::vector<std::vector<float>> linearinterpolate_coordinatevector(std::vector<std::vector<float>> &coordinates, float *ux, float *uy, float *uz, int imgshape[3])
    {
        std::vector<std::vector<float>> output;
        for (long long int i = 0; i < coordinates.size(); i++) output.push_back({0.0f, 0.0f, 0.0f});

        #pragma omp parallel for
        for (long long int i = 0; i < coordinates.size(); i++)
        {
            output[i][0] = linearinterpolate_coordinate(coordinates[i][0], coordinates[i][1], coordinates[i][2], ux, imgshape);
            output[i][1] = linearinterpolate_coordinate(coordinates[i][0], coordinates[i][1], coordinates[i][2], uy, imgshape);
            output[i][2] = linearinterpolate_coordinate(coordinates[i][0], coordinates[i][1], coordinates[i][2], uz, imgshape);
        }

        return output;
    }
    std::vector<float> linearinterpolate_coordinatevector(std::vector<std::vector<float>> &coordinates, float *scalar, int imgshape[3])
    {
        std::vector<float> output;
        for (long long int i = 0; i < coordinates.size(); i++) output.push_back({0.0f});

        #pragma omp parallel for
        for (long long int i = 0; i < coordinates.size(); i++)
            output[i] = linearinterpolate_coordinate(coordinates[i][0], coordinates[i][1], coordinates[i][2], scalar, imgshape);

        return output;
    }
    std::vector<float> maxval_at_coordinatevector(std::vector<std::vector<float>> &coordinates, float *scalar, int imgshape[3])
    {
        std::vector<float> output;
        for (long long int i = 0; i < coordinates.size(); i++) output.push_back({0.0f});

        #pragma omp parallel for
        for (long long int i = 0; i < coordinates.size(); i++)
            output[i] = maxval_at_coordinate(coordinates[i][0], coordinates[i][1], coordinates[i][2], scalar, imgshape);

        return output;
    }
    std::vector<std::vector<float>> linearinterpolate_averagecellvalue(std::vector<std::vector<float>> &coordinates, std::vector<std::vector<int64_t>> &tetrahedra, float *ux, float *uy, float *uz, int imgshape[3])
    {
        std::vector<std::vector<float>> output;
        for (long long int i = 0; i < tetrahedra.size(); i++) output.push_back({0.0f, 0.0f, 0.0f});

        #pragma omp parallel for
        for (long long int i = 0; i < tetrahedra.size(); i++)
        {
            for (int k = 0; k < 4; k++)
            {
                std::vector<float> active_coordinate = coordinates[tetrahedra[i][k]];
                output[i][0] += 0.25*linearinterpolate_coordinate(active_coordinate[0], active_coordinate[1], active_coordinate[2], ux, imgshape);
                output[i][1] += 0.25*linearinterpolate_coordinate(active_coordinate[0], active_coordinate[1], active_coordinate[2], uy, imgshape);
                output[i][2] += 0.25*linearinterpolate_coordinate(active_coordinate[0], active_coordinate[1], active_coordinate[2], uz, imgshape);
            }
        }

        return output;
    }
    std::vector<float> linearinterpolate_averagecellvalue(std::vector<std::vector<float>> &coordinates, std::vector<std::vector<int64_t>> &tetrahedra, float *scalar, int imgshape[3])
    {
        std::vector<float> output;
        for (long long int i = 0; i < tetrahedra.size(); i++) output.push_back({0.0f});

        #pragma omp parallel for
        for (long long int i = 0; i < tetrahedra.size(); i++)
        {
            for (int k = 0; k < 4; k++)
            {
                std::vector<float> active_coordinate = coordinates[tetrahedra[i][k]];
                output[i] += 0.25*linearinterpolate_coordinate(active_coordinate[0], active_coordinate[1], active_coordinate[2], scalar, imgshape);
            }
        }

        return output;
    }
    std::vector<std::vector<float>> get_cell_centerofgravity(std::vector<std::vector<float>> &coordinates, std::vector<std::vector<int64_t>> &tetrahedra)
    {
        std::vector<std::vector<float>> output;

        for (long long int i = 0; i < tetrahedra.size(); i++)
        {
            float x = 0.25f*(coordinates[tetrahedra[i][0]][0]+coordinates[tetrahedra[i][1]][0]+coordinates[tetrahedra[i][2]][0]+coordinates[tetrahedra[i][3]][0]);
            float y = 0.25f*(coordinates[tetrahedra[i][0]][1]+coordinates[tetrahedra[i][1]][1]+coordinates[tetrahedra[i][2]][1]+coordinates[tetrahedra[i][3]][1]);
            float z = 0.25f*(coordinates[tetrahedra[i][0]][2]+coordinates[tetrahedra[i][1]][2]+coordinates[tetrahedra[i][2]][2]+coordinates[tetrahedra[i][3]][2]);
            output.push_back({x,y,z});
        }

        return output;
    }
    std::vector<std::vector<float>> get_triangle_centerofgravity_minzpos(std::vector<std::vector<float>> &coordinates, std::vector<std::vector<int64_t>> &triangles, float min_zpos)
    {
        std::vector<std::vector<float>> output;

        for (long long int i = 0; i < triangles.size(); i++)
        {
            float x = 1.f/3.f*(coordinates[triangles[i][0]][0]+coordinates[triangles[i][1]][0]+coordinates[triangles[i][2]][0]);
            float y = 1.f/3.f*(coordinates[triangles[i][0]][1]+coordinates[triangles[i][1]][1]+coordinates[triangles[i][2]][1]);
            float z = 1.f/3.f*(coordinates[triangles[i][0]][2]+coordinates[triangles[i][1]][2]+coordinates[triangles[i][2]][2]);

            if(z >= min_zpos)
                output.push_back({x,y,z});
        }

        return output;
    }
    std::vector<std::vector<float>> get_triangle_centerofgravity(std::vector<std::vector<float>> &coordinates, std::vector<std::vector<int64_t>> &triangles)
    {
        std::vector<std::vector<float>> output;

        for (long long int i = 0; i < triangles.size(); i++)
        {
            float x = 1.f/3.f*(coordinates[triangles[i][0]][0]+coordinates[triangles[i][1]][0]+coordinates[triangles[i][2]][0]);
            float y = 1.f/3.f*(coordinates[triangles[i][0]][1]+coordinates[triangles[i][1]][1]+coordinates[triangles[i][2]][1]);
            float z = 1.f/3.f*(coordinates[triangles[i][0]][2]+coordinates[triangles[i][1]][2]+coordinates[triangles[i][2]][2]);
            output.push_back({x,y,z});
        }

        return output;
    }

    std::vector<std::vector<float>> get_inspheres(std::vector<std::vector<float>> &coordinates, std::vector<std::vector<int64_t>> &tetrahedra)
    {
        //calculates the radius of the insphere and its radius

        std::vector<std::vector<float>> output;

        for (int i = 0; i < tetrahedra.size(); i++)
        {
            //Ref: http://maths.ac-noumea.nc/polyhedr/stuff/tetra_rs_.htm

            std::vector<float> p0 = coordinates[tetrahedra[i][0]];
            std::vector<float> p1 = coordinates[tetrahedra[i][1]];
            std::vector<float> p2 = coordinates[tetrahedra[i][2]];
            std::vector<float> p3 = coordinates[tetrahedra[i][3]];

            float a[3] = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
            float b[3] = {p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]};
            float c[3] = {p3[0]-p0[0], p3[1]-p0[1], p3[2]-p0[2]};

            //cross-product
            float N012[3] = {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
            float N013[3] = {a[1]*c[2]-a[2]*c[1], a[2]*c[0]-a[0]*c[2], a[0]*c[1]-a[1]*c[0]};
            float N023[3] = {b[1]*c[2]-b[2]*c[1], b[2]*c[0]-b[0]*c[2], b[0]*c[1]-b[1]*c[0]};

            b[0] = p2[0]-p1[0]; b[1] = p2[1]-p1[1]; b[2] = p2[2]-p1[2];
            c[0] = p3[0]-p1[0]; c[1] = p3[1]-p1[1]; c[2] = p3[2]-p1[2];

            float N123[3] = {b[1]*c[2]-b[2]*c[1], b[2]*c[0]-b[0]*c[2], b[0]*c[1]-b[1]*c[0]};

            //norms
            float norm012 = sqrtf(N012[0]*N012[0] + N012[1]*N012[1] + N012[2]*N012[2]);
            float norm013 = sqrtf(N013[0]*N013[0] + N013[1]*N013[1] + N013[2]*N013[2]);
            float norm023 = sqrtf(N023[0]*N023[0] + N023[1]*N023[1] + N023[2]*N023[2]);
            float norm123 = sqrtf(N123[0]*N123[0] + N123[1]*N123[1] + N123[2]*N123[2]);
            float normsum = norm012+norm023+norm013+norm123;

            //center of insphere
            float Si[3] = {(norm012*p3[0] + norm013*p2[0]+ norm023*p1[0] + norm123*p0[0])/normsum,
                           (norm012*p3[1] + norm013*p2[1]+ norm023*p1[1] + norm123*p0[1])/normsum,
                           (norm012*p3[2] + norm013*p2[2]+ norm023*p1[2] + norm123*p0[2])/normsum};

            //determinant (Laplace)
            float ay = -p0[1]*(p1[0]*p2[2] + p1[2]*p3[0] + p2[0]*p3[2] - p3[0]*p2[2] - p3[2]*p1[0] - p2[0]*p1[2]);
            float by =  p1[1]*(p0[0]*p2[2] + p0[2]*p3[0] + p2[0]*p3[2] - p3[0]*p2[2] - p3[2]*p0[0] - p2[0]*p0[2]);
            float cy = -p2[1]*(p0[0]*p1[2] + p0[2]*p3[0] + p1[0]*p3[2] - p3[0]*p1[2] - p3[2]*p0[0] - p1[0]*p0[2]);
            float dy =  p3[1]*(p0[0]*p1[2] + p0[2]*p2[0] + p1[0]*p2[2] - p2[0]*p1[2] - p2[2]*p0[0] - p1[0]*p0[2]);
            float alpha = ay+by+cy+dy;

            //radius of insphere
            float ri = fabs(alpha/normsum);

            std::vector<float> result = {Si[0], Si[1], Si[2], ri};
            output.push_back(result);
        }

        return output;
    }

    std::vector<std::vector<float>> vertex_subset(int valid_label, std::vector<std::vector<float>> &vertices, std::vector<std::vector<int64_t>> &tetrahedra, std::vector<int> &celllabel)
    {
        std::vector<std::vector<float>> output;
        std::vector<bool> marked_vertex(vertices.size(), 0);

        for (int64_t idx = 0; idx < celllabel.size(); idx++)
        {
            if (celllabel[idx] == valid_label)
            {
                for (int i = 0; i < 4; i++)
                    marked_vertex[tetrahedra[idx][i]] = true;
            }
        }
        for (int64_t idx = 0; idx < marked_vertex.size(); idx++)
        {
            if (marked_vertex[idx])
                output.push_back(vertices[idx]);
        }
        return output;
    }

    float* calc_meanzpos_label(uint8_t *labels, int shape[3], float sigma)
    {
        int nx = shape[0];
        long long int nslice = shape[0]*shape[1];


        float* output = (float*) calloc(nslice, sizeof(*output));
        bool unassigned = false;

        //at each xy position measure the average z location of the labels
        #pragma omp parallel for
        for (long long int idx = 0; idx < nslice; idx++)
        {
            float meanzpos = 0.0f;
            float counter = 0.0f;

            for (int z = 0; z < shape[2]; z++)
            {
                if(labels[idx+z*nslice] != 0)
                {
                    meanzpos += z;
                    counter++;
                }
            }

            meanzpos = counter > 0 ? meanzpos/counter : -1;

            if (meanzpos == -1 && unassigned == false) unassigned = true;

            output[idx] = meanzpos;
        }

        //dilate until all pixels are filled
        while(unassigned)
        {
            unassigned = false;

            #pragma omp parallel for
            for (long long int idx = 0; idx < nslice; idx++)
            {
                if (output[idx] == -1)
                {
                    int y = idx/nx;
                    int x = idx-y*nx;

                    float val = 0.0;
                    int counter = 0;

                    if (x-1 > 0        && output[idx-1] != -1) {counter++; val += output[idx-1];}
                    if (x+1 < nx && output[idx+1] != -1) {counter++; val += output[idx+1];}
                    if (y-1 > 0        && output[idx-nx] != -1) {counter++; val += output[idx-nx];}
                    if (y+1 < shape[1] && output[idx+nx] != -1) {counter++; val += output[idx+nx];}

                    if (x-1 > 0 && y-1 > 0 && output[idx-nx-1] != -1) {counter++; val += output[idx-nx-1];}
                    if (x-1 > 0 && y+1 < shape[1] && output[idx+nx-1] != -1) {counter++; val += output[idx+nx-1];}
                    if (x+1 < nx && y-1 > 0 && output[idx-nx+1] != -1) {counter++; val += output[idx-nx+1];}
                    if (x+1 < nx && y+1 < shape[1] && output[idx+nx+1] != -1) {counter++; val += output[idx+nx+1];}

                    if (counter == 0) unassigned = true;
                    else
                        output[idx] = val/counter;
                }
            }
        }

        //apply filter
        if (sigma > 0.0)
            filter::apply_2DGaussianFilter(output, shape, sigma);

        return output;
    }
    void relabel_as_abovebelow_meanzpos(float* meanzpos, int shape[2], std::vector<int> &mesh_labels, std::vector<std::vector<float>> &center_of_gravity)
    {
        for (int i = 0; i < mesh_labels.size(); i++)
        {
            float x = center_of_gravity[i][0];
            float y = center_of_gravity[i][1];
            float z = center_of_gravity[i][2];
            float depth = linearinterpolate_coordinate2D(x, y, meanzpos, shape);

            if (z > depth) mesh_labels[i] = 2;
            else mesh_labels[i] = 1;
        }

        return;
    }

    float* get_interface(uint8_t* labelimage, int shape[3])
    {
        int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
        long long int nslice = shape[0]*shape[1]; long long int nstack = shape[2]*nslice;

        float* interfaces = (float*) calloc(nstack, sizeof(*interfaces));

        #pragma omp parallel for
        for (long long int idx = 0; idx < nstack; idx++)
        {
            int z = idx/nslice;
            int y = (idx-z*nslice)/shape[0];
            int x = idx-z*nslice-y*shape[0];
            uint8_t this_val = labelimage[idx];

            if      (x+1 < nx && this_val != labelimage[idx+1]) interfaces[idx] = 1.0f;
            else if (x-1 >= 0 && this_val != labelimage[idx-1]) interfaces[idx] = 1.0f;
            else if (y+1 < ny && this_val != labelimage[idx+nx]) interfaces[idx] = 1.0f;
            else if (y-1 >= 0 && this_val != labelimage[idx-nx]) interfaces[idx] = 1.0f;
            else if (z+1 < nz && this_val != labelimage[idx+nslice]) interfaces[idx] = 1.0f;
            else if (z-1 >= 0 && this_val != labelimage[idx-nslice]) interfaces[idx] = 1.0f;

            else if (x+1 < nx && y+1 < ny && this_val != labelimage[idx+1+nx]) interfaces[idx] = 1.0f;
            else if (x-1 >= 0 && y+1 < ny && this_val != labelimage[idx-1+nx]) interfaces[idx] = 1.0f;
            else if (x+1 < nx && y-1 >= 0 && this_val != labelimage[idx+1-nx]) interfaces[idx] = 1.0f;
            else if (x-1 >= 0 && y-1 >= 0 && this_val != labelimage[idx-1-nx]) interfaces[idx] = 1.0f;
            else if (x+1 < nx && z+1 < nz && this_val != labelimage[idx+1+nslice]) interfaces[idx] = 1.0f;
            else if (x-1 >= 0 && z+1 < nz && this_val != labelimage[idx-1+nslice]) interfaces[idx] = 1.0f;
            else if (x+1 < nx && z-1 >= 0 && this_val != labelimage[idx+1-nslice]) interfaces[idx] = 1.0f;
            else if (x-1 >= 0 && z-1 >= 0 && this_val != labelimage[idx-1-nslice]) interfaces[idx] = 1.0f;
            else if (y+1 < ny && z+1 < nz && this_val != labelimage[idx+nx+nslice]) interfaces[idx] = 1.0f;
            else if (y-1 >= 0 && z+1 < nz && this_val != labelimage[idx-nx+nslice]) interfaces[idx] = 1.0f;
            else if (y+1 < ny && z-1 >= 0 && this_val != labelimage[idx+nx-nslice]) interfaces[idx] = 1.0f;
            else if (y-1 >= 0 && z-1 >= 0 && this_val != labelimage[idx-nx-nslice]) interfaces[idx] = 1.0f;

            else if (x+1 < nx && y+1 < ny && z < nz && this_val != labelimage[idx+1+nx+nslice]) interfaces[idx] = 1.0f;
            else if (x-1 >= 0 && y+1 < ny && z < nz && this_val != labelimage[idx-1+nx+nslice]) interfaces[idx] = 1.0f;
            else if (x+1 < nx && y-1 >= 0 && z < nz && this_val != labelimage[idx+1-nx+nslice]) interfaces[idx] = 1.0f;
            else if (x-1 >= 0 && y-1 >= 0 && z < nz && this_val != labelimage[idx-1-nx+nslice]) interfaces[idx] = 1.0f;
            else if (x+1 < nx && y+1 < ny && z >= 0 && this_val != labelimage[idx+1+nx-nslice]) interfaces[idx] = 1.0f;
            else if (x-1 >= 0 && y+1 < ny && z >= 0 && this_val != labelimage[idx-1+nx-nslice]) interfaces[idx] = 1.0f;
            else if (x+1 < nx && y-1 >= 0 && z >= 0 && this_val != labelimage[idx+1-nx-nslice]) interfaces[idx] = 1.0f;
            else if (x-1 >= 0 && y-1 >= 0 && z >= 0 && this_val != labelimage[idx-1-nx-nslice]) interfaces[idx] = 1.0f;
        }

        return interfaces;
    }
}

