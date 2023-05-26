#include <math.h>
#include <omp.h>
#include "taubin_smoothing.h"

namespace mesh_filter
{
    void TaubinFilter::run_areafilter(std::vector<std::vector<float>> &vertices, std::vector<std::vector<int64_t>> &triangles, std::pair<float,float> zrange)
    {
        for (int iter = 0; iter < 2*iterations; iter++)
        {
            std::vector<float> area_sum(vertices.size(), 0.0f);
            std::vector<float> delta_x(vertices.size(), 0.0f);
            std::vector<float> delta_y(vertices.size(), 0.0f);
            std::vector<float> delta_z(vertices.size(), 0.0f);

            #pragma omp parallel for
            for (int64_t i = 0; i < (int64_t) triangles.size(); i++)
            {
                //Heron's formula for triangle area
                ////////////////////////////////////////////
                std::vector<float> P0 = vertices[triangles[i][0]];
                std::vector<float> P1 = vertices[triangles[i][1]];
                std::vector<float> P2 = vertices[triangles[i][2]];

                float a = sqrtf((P0[0]-P1[0])*(P0[0]-P1[0])+(P0[1]-P1[1])*(P0[1]-P1[1])+(P0[2]-P1[2])*(P0[2]-P1[2]));
                float b = sqrtf((P0[0]-P2[0])*(P0[0]-P2[0])+(P0[1]-P2[1])*(P0[1]-P2[1])+(P0[2]-P2[2])*(P0[2]-P2[2]));
                float c = sqrtf((P2[0]-P1[0])*(P2[0]-P1[0])+(P2[1]-P1[1])*(P2[1]-P1[1])+(P2[2]-P1[2])*(P2[2]-P1[2]));
                float s = (a+b+c)*0.5f;

                float area = sqrtf(s*(s-a)*(s-b)*(s-c));

                area_sum[triangles[i][0]] += 2*area;
                area_sum[triangles[i][1]] += 2*area;
                area_sum[triangles[i][2]] += 2*area;
                ////////////////////////////////////////////

                //add up weighted shifts
                ////////////////////////////////////////////
                delta_x[triangles[i][0]] += area*((P1[0]-P0[0])+(P2[0]-P0[0]));
                delta_y[triangles[i][0]] += area*((P1[1]-P0[1])+(P2[1]-P0[1]));
                delta_z[triangles[i][0]] += area*((P1[2]-P0[2])+(P2[2]-P0[2]));

                delta_x[triangles[i][1]] += area*((P0[0]-P1[0])+(P2[0]-P1[0]));
                delta_y[triangles[i][1]] += area*((P0[1]-P1[1])+(P2[1]-P1[1]));
                delta_z[triangles[i][1]] += area*((P0[2]-P1[2])+(P2[2]-P1[2]));

                delta_x[triangles[i][2]] += area*((P0[0]-P2[0])+(P1[0]-P2[0]));
                delta_y[triangles[i][2]] += area*((P0[1]-P2[1])+(P1[1]-P2[1]));
                delta_z[triangles[i][2]] += area*((P0[2]-P2[2])+(P1[2]-P2[2]));

                ////////////////////////////////////////////
            }

            //normalize and shift
            ////////////////////////////////////////////
            #pragma omp parallel for
            for (int64_t i = 0; i < (int64_t) vertices.size(); i++)
            {
                float z_coordinate = vertices[i][2];
                if (z_coordinate <= zrange.first || z_coordinate >= zrange.second)
                    continue;

                if ((iter%2) == 0)
                {
                    if(area_sum[i] > 0.0f)
                    {
                        vertices[i][0] += lmbda*delta_x[i]/area_sum[i];
                        vertices[i][1] += lmbda*delta_y[i]/area_sum[i];
                        vertices[i][2] += lmbda*delta_z[i]/area_sum[i];
                    }
                }
                else
                {
                    if(area_sum[i] > 0.0f)
                    {
                        vertices[i][0] += mu*delta_x[i]/area_sum[i];
                        vertices[i][1] += mu*delta_y[i]/area_sum[i];
                        vertices[i][2] += mu*delta_z[i]/area_sum[i];
                    }
                }
            }
            ////////////////////////////////////////////
        }

        return;
    }
}
