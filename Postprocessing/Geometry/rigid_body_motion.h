#ifndef RIGID_BODY_MOTION_H
#define RIGID_BODY_MOTION_H

#include <iostream>
#include <string.h>
#include <cstdint>
#include <vector>
#include <math.h>

namespace rbmotion
{
    typedef int mesh_labeltype;

    void eliminate_translation(float *dx, float *dy, float *dz, int shape[3], bool console_print)
    {
        int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
        long long int nslice = nx*ny;
        long long int nstack = nz*nslice;

        double ev_x = 0.0; double ev_y = 0.0; double ev_z = 0.0;

        //get average displacements
        #pragma omp parallel for reduction(+: ev_x, ev_y, ev_z)
        for(long long int idx = 0; idx < nstack; idx++)
        {
            ev_x += dx[idx];
            ev_y += dy[idx];
            ev_z += dz[idx];
        }

        ev_x /= nstack; ev_y /= nstack; ev_z /= nstack;

        //and subtract
        if(console_print)
            std::cout << "average translation: " << round(ev_x*100.)/100. << ", " << round(ev_y*100.)/100. << ", " << round(ev_z*100.)/100. << std::endl;

        #pragma omp parallel for
        for(long long int idx = 0; idx < nstack; idx++)
        {
            dx[idx] -= ev_x;
            dy[idx] -= ev_y;
            dz[idx] -= ev_z;
        }

        return;
    }
    void eliminate_translation(float *dx, float *dy, float *dz, uint8_t *mask, int shape[3], uint8_t motion_label, bool console_print)
    {
        int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
        long long int nslice = nx*ny;
        long long int nstack = nz*nslice;

        double ev_x = 0.0; double ev_y = 0.0; double ev_z = 0.0;
        double counter = 0.0;

        //get average displacements
        #pragma omp parallel for reduction(+: ev_x, ev_y, ev_z, counter)
        for(long long int idx = 0; idx < nstack; idx++)
        {
            if(mask[idx] != 0 && (motion_label == 0 || motion_label == mask[idx]))
            {
                ev_x += dx[idx];
                ev_y += dy[idx];
                ev_z += dz[idx];
                counter++;
            }
        }

        ev_x /= counter; ev_y /= counter; ev_z /= counter;

        //and subtract
        if(console_print)
        {
            std::cout << counter/nstack*100. << " % of voxels" << std::endl;
            std::cout << "average translation: " << round(ev_x*100.)/100. << ", " << round(ev_y*100.)/100. << ", " << round(ev_z*100.)/100. << std::endl;
        }

        #pragma omp parallel for
        for(long long int idx = 0; idx < nstack; idx++)
        {
            dx[idx] -= ev_x;
            dy[idx] -= ev_y;
            dz[idx] -= ev_z;
        }

        return;
    }
    void eliminate_translation(std::vector<std::vector<float>> &displacements, std::vector<mesh_labeltype> &mesh_labels, mesh_labeltype motion_label, bool console_print)
    {
        double ev_x = 0.0; double ev_y = 0.0; double ev_z = 0.0;
        double counter = 0.0;

        //get average displacements
        #pragma omp parallel for reduction(+: ev_x, ev_y, ev_z, counter)
        for(long long int idx = 0; idx < displacements.size(); idx++)
        {
            if(motion_label == 0 || motion_label == mesh_labels[idx])
            {
                ev_x += displacements[idx][0];
                ev_y += displacements[idx][1];
                ev_z += displacements[idx][2];
                counter++;
            }
        }

        ev_x /= counter; ev_y /= counter; ev_z /= counter;

        //and subtract
        if(console_print)
        {
            std::cout << counter/displacements.size()*100. << " % of cells" << std::endl;
            std::cout << "average translation: " << round(ev_x*100.)/100. << ", " << round(ev_y*100.)/100. << ", " << round(ev_z*100.)/100. << std::endl;
        }

        #pragma omp parallel for
        for(long long int idx = 0; idx < displacements.size(); idx++)
        {
            displacements[idx][0] -= ev_x;
            displacements[idx][1] -= ev_y;
            displacements[idx][2] -= ev_z;
        }

        return;
    }
     void eliminate_translation(std::vector<std::vector<float>> &eval_displacements, std::vector<std::vector<float>> &apply_displacements, bool console_print)
    {
        double ev_x = 0.0; double ev_y = 0.0; double ev_z = 0.0;
        double counter = 0.0;

        //get average displacements
        #pragma omp parallel for reduction(+: ev_x, ev_y, ev_z, counter)
        for(long long int idx = 0; idx < eval_displacements.size(); idx++)
        {
            ev_x += eval_displacements[idx][0];
            ev_y += eval_displacements[idx][1];
            ev_z += eval_displacements[idx][2];
            counter++;
        }

        ev_x /= counter; ev_y /= counter; ev_z /= counter;

        //and subtract
        if(console_print)
        {
            std::cout << counter/apply_displacements.size()*100. << " % of cells" << std::endl;
            std::cout << "average translation: " << round(ev_x*100.)/100. << ", " << round(ev_y*100.)/100. << ", " << round(ev_z*100.)/100. << std::endl;
        }

        #pragma omp parallel for
        for(long long int idx = 0; idx < eval_displacements.size(); idx++)
        {
            eval_displacements[idx][0] -= ev_x;
            eval_displacements[idx][1] -= ev_y;
            eval_displacements[idx][2] -= ev_z;
        }
        #pragma omp parallel for
        for(long long int idx = 0; idx < apply_displacements.size(); idx++)
        {
            apply_displacements[idx][0] -= ev_x;
            apply_displacements[idx][1] -= ev_y;
            apply_displacements[idx][2] -= ev_z;
        }

        return;
    }

    double _vectormagnitude(float *dx, float *dy, float *dz, int shape[3])
    {
        int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
        long long int nslice = nx*ny;
        long long int nstack = nz*nslice;

        double sum = 0.0;

        #pragma omp parallel for reduction(+: sum)
        for(long long int idx = 0; idx < nstack; idx++)
        {
            float valx = dx[idx];
            float valy = dy[idx];
            float valz = dz[idx];
            sum += sqrtf(valx*valx+valy*valy+valz*valz);
        }

        return sum;
    }
    double _vectormagnitude(float *dx, float *dy, float *dz, uint8_t* mask, int shape[3], uint8_t motion_label)
    {
        int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
        long long int nslice = nx*ny;
        long long int nstack = nz*nslice;

        double sum = 0.0;

        #pragma omp parallel for reduction(+: sum)
        for(long long int idx = 0; idx < nstack; idx++)
        {
            if(mask[idx] != 0 && (motion_label == 0 || motion_label == mask[idx]))
            {
                float valx = dx[idx];
                float valy = dy[idx];
                float valz = dz[idx];
                sum += sqrtf(valx*valx+valy*valy+valz*valz);
            }
        }

        return sum;
    }
    double _vectormagnitude(std::vector<std::vector<float>> &displacements, std::vector<mesh_labeltype> &mesh_labels, mesh_labeltype motion_label)
    {
        double sum = 0.0;

        #pragma omp parallel for reduction(+: sum)
        for(long long int idx = 0; idx < displacements.size(); idx++)
        {
            if(motion_label == 0 || motion_label == mesh_labels[idx])
            {
                float valx = displacements[idx][0];
                float valy = displacements[idx][1];
                float valz = displacements[idx][2];
                sum += sqrtf(valx*valx+valy*valy+valz*valz);
            }
        }

        return sum;
    }

    void _prepare_rotation_coefficients_(float *out_coefficients, float yaw, float roll, float pitch)
	{
		//Preprare rotation coefficients
		float phi = yaw*0.01745329252f;
		float theta = pitch*0.01745329252f;
		float psi = roll*0.01745329252f;

		float costheta = cos(theta);
		float sintheta = sin(theta);
		float cospsi = cos(psi);
		float sinpsi = sin(psi);
		float sinpsisintheta = sinpsi*sintheta;
		float cospsisintheta = cospsi*sintheta;
		float cosphi = cos(phi);
		float sinphi = sin(phi);

		//pitch-roll-yaw convention
		float a11 = costheta*cosphi;
		float a12 = costheta*sinphi;
		float a13 = -sintheta;
		float a21 = sinpsisintheta*cosphi-cospsi*sinphi;
		float a22 = sinpsisintheta*sinphi+cospsi*cosphi;
		float a23 = costheta*sinpsi;
		float a31 = cospsisintheta*cosphi+sinpsi*sinphi;
		float a32 = cospsisintheta*sinphi-sinpsi*cosphi;
		float a33 = costheta*cospsi;

		out_coefficients[0] = a11; out_coefficients[1] = a12; out_coefficients[2] = a13;
		out_coefficients[3] = a21; out_coefficients[4] = a22; out_coefficients[5] = a23;
		out_coefficients[6] = a31; out_coefficients[7] = a32; out_coefficients[8] = a33;

		return;
	}

    float _test_activerotation(float *dx, float *dy, float *dz, int shape[3], float yaw, float roll, float pitch)
    {
        float rotcenter[3] = {shape[0]/2.f, shape[1]/2.f, shape[2]/2.f};

        int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
        long long int nslice = nx*ny;
        long long int nstack = nz*nslice;

        float* next_dx = (float*) calloc(nstack, sizeof(*next_dx));
        float* next_dy = (float*) calloc(nstack, sizeof(*next_dy));
        float* next_dz = (float*) calloc(nstack, sizeof(*next_dz));

        float rotation_matrix[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        _prepare_rotation_coefficients_(rotation_matrix,yaw, roll, pitch);

        float a11 = rotation_matrix[0]; float a12 = rotation_matrix[1]; float a13 = rotation_matrix[2];
        float a21 = rotation_matrix[3]; float a22 = rotation_matrix[4]; float a23 = rotation_matrix[5];
        float a31 = rotation_matrix[6]; float a32 = rotation_matrix[7]; float a33 = rotation_matrix[8];

        #pragma omp parallel for
        for(long long int idx = 0; idx < nstack; idx++)
        {
            int z = idx/nslice;
            int y = (idx-z*nslice)/nx;
            int x = idx-z*nslice-y*nx;

            float z0 = z-rotcenter[2]-0.5;
            float y0 = y-rotcenter[1]-0.5;
            float x0 = x-rotcenter[0]-0.5;

            //clockwise
            float x1 = a11*x0 + a12*y0 + a13*z0;
            float y1 = a21*x0 + a22*y0 + a23*z0;
            float z1 = a31*x0 + a32*y0 + a33*z0;

            //counter clockwise
            //float x1 = a11*x0 + a21*y0 + a31*z0;
            //float y1 = a12*x0 + a22*y0 + a32*z0;
            //float z1 = a13*x0 + a23*y0 + a33*z0;

            next_dx[idx] = dx[idx]+x1-x0;
            next_dy[idx] = dy[idx]+y1-y0;
            next_dz[idx] = dz[idx]+z1-z0;
        }

        eliminate_translation(next_dx, next_dy, next_dz, shape, false);

        float next_sum = _vectormagnitude(next_dx, next_dy, next_dz, shape);

        free(next_dx); free(next_dy); free(next_dz);

        return next_sum;
    }
    float _test_activerotation(float *dx, float *dy, float *dz, uint8_t* mask, int shape[3], uint8_t motion_label, float yaw, float roll, float pitch)
    {
        float rotcenter[3] = {shape[0]/2.f, shape[1]/2.f, shape[2]/2.f};

        int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
        long long int nslice = nx*ny;
        long long int nstack = nz*nslice;

        float* next_dx = (float*) calloc(nstack, sizeof(*next_dx));
        float* next_dy = (float*) calloc(nstack, sizeof(*next_dy));
        float* next_dz = (float*) calloc(nstack, sizeof(*next_dz));

        float rotation_matrix[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        _prepare_rotation_coefficients_(rotation_matrix,yaw, roll, pitch);

        float a11 = rotation_matrix[0]; float a12 = rotation_matrix[1]; float a13 = rotation_matrix[2];
        float a21 = rotation_matrix[3]; float a22 = rotation_matrix[4]; float a23 = rotation_matrix[5];
        float a31 = rotation_matrix[6]; float a32 = rotation_matrix[7]; float a33 = rotation_matrix[8];

        #pragma omp parallel for
        for(long long int idx = 0; idx < nstack; idx++)
        {
            if(mask[idx] != 0 && (motion_label == 0 || motion_label == mask[idx]))
            {
                int z = idx/nslice;
                int y = (idx-z*nslice)/nx;
                int x = idx-z*nslice-y*nx;

                float z0 = z-rotcenter[2]-0.5;
                float y0 = y-rotcenter[1]-0.5;
                float x0 = x-rotcenter[0]-0.5;

                //clockwise
                float x1 = a11*x0 + a12*y0 + a13*z0;
                float y1 = a21*x0 + a22*y0 + a23*z0;
                float z1 = a31*x0 + a32*y0 + a33*z0;

                //counter clockwise
                //float x1 = a11*x0 + a21*y0 + a31*z0;
                //float y1 = a12*x0 + a22*y0 + a32*z0;
                //float z1 = a13*x0 + a23*y0 + a33*z0;
                next_dx[idx] = dx[idx]+x1-x0;
                next_dy[idx] = dy[idx]+y1-y0;
                next_dz[idx] = dz[idx]+z1-z0;
            }
        }

        eliminate_translation(next_dx, next_dy, next_dz, mask, shape, motion_label, false);

        float next_sum = _vectormagnitude(next_dx, next_dy, next_dz, mask, shape, motion_label);

        free(next_dx); free(next_dy); free(next_dz);

        return next_sum;
    }
    float _test_activerotation(std::vector<std::vector<float>> &displacements, std::vector<std::vector<float>> &locations, std::vector<mesh_labeltype> &mesh_labels, int shape[3], mesh_labeltype motion_label, float yaw, float roll, float pitch)
    {
        float rotcenter[3] = {shape[0]/2.f, shape[1]/2.f, shape[2]/2.f};

        std::vector<std::vector<float>> next_displacements;
        for (int i = 0; i < displacements.size(); i++)
            next_displacements.push_back({0.0,0.0,0.0});

        float rotation_matrix[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        _prepare_rotation_coefficients_(rotation_matrix,yaw, roll, pitch);

        float a11 = rotation_matrix[0]; float a12 = rotation_matrix[1]; float a13 = rotation_matrix[2];
        float a21 = rotation_matrix[3]; float a22 = rotation_matrix[4]; float a23 = rotation_matrix[5];
        float a31 = rotation_matrix[6]; float a32 = rotation_matrix[7]; float a33 = rotation_matrix[8];

        #pragma omp parallel for
        for(long long int idx = 0; idx < displacements.size(); idx++)
        {
            if(motion_label == 0 || motion_label == mesh_labels[idx])
            {
                float z0 = locations[idx][2]-rotcenter[2]-0.5;
                float y0 = locations[idx][1]-rotcenter[1]-0.5;
                float x0 = locations[idx][0]-rotcenter[0]-0.5;

                //clockwise
                float x1 = a11*x0 + a12*y0 + a13*z0;
                float y1 = a21*x0 + a22*y0 + a23*z0;
                float z1 = a31*x0 + a32*y0 + a33*z0;

                //counter clockwise
                //float x1 = a11*x0 + a21*y0 + a31*z0;
                //float y1 = a12*x0 + a22*y0 + a32*z0;
                //float z1 = a13*x0 + a23*y0 + a33*z0;
                next_displacements[idx][0] = displacements[idx][0]+x1-x0;
                next_displacements[idx][1] = displacements[idx][1]+y1-y0;
                next_displacements[idx][2] = displacements[idx][2]+z1-z0;
            }
        }

        eliminate_translation(next_displacements, mesh_labels, motion_label, false);

        float next_sum = _vectormagnitude(next_displacements, mesh_labels, motion_label);

        return next_sum;
    }

    void eliminate_rigidbody_motion(float *dx, float *dy, float *dz, int shape[3], float target_precision = 0.1, float scaling = 10.)
    {
        int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
        long long int nslice = nx*ny;
        long long int nstack = nz*nslice;

        float rotcenter[3] = {(float) (shape[0]/2.f), (float) (shape[1]/2.f), (float) (shape[2]/2.f)};

        //first remove translation
        eliminate_translation(dx, dy, dz, shape, true);

        //now get the sum of remaining absolute translations and minimize
        float best_yaw = 0.0f;
        float best_roll = 0.0f;
        float best_pitch = 0.0f;
        float best_sum = _vectormagnitude(dx, dy, dz, shape);

        std::cout << "optimizing yaw, roll, pitch: " << std::endl;
        std::cout << best_yaw << " " << best_roll << " " << best_pitch << " " << best_sum << "              \r";
        std::cout << std::endl;
        std::cout.flush();

        float rotation_matrix[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        float active_precision = 10.;

        //minimize the remaining absolute sum of translations
        while(0 == 0)
        {

            active_precision /= scaling;
            std::cout << "precision: " << active_precision << std::endl;

            bool external_change = true;

            while(external_change)
            {
                external_change = false;

                for (int i = 0; i < 6; i++)
                {
                    float increment[3] = {0.0, 0.0, 0.0};
                    if (i == 0) increment[0] = active_precision;
                    else if (i == 1) increment[0] = -active_precision;
                    else if (i == 2) increment[1] = active_precision;
                    else if (i == 3) increment[1] = -active_precision;
                    else if (i == 4) increment[2] = active_precision;
                    else if (i == 5) increment[2] = -active_precision;

                    float active_sum = _test_activerotation(dx, dy, dz, shape, best_yaw+increment[0], best_roll+increment[1], best_pitch+increment[2]);

                    if (active_sum < best_sum)
                    {
                        external_change = true;

                        best_yaw += increment[0];
                        best_roll += increment[1];
                        best_pitch += increment[2];
                        best_sum = active_sum;

                        std::cout << best_yaw << " " << best_roll << " " << best_pitch << " " << best_sum << "              \r";
                        std::cout.flush();
                    }
                }
            }
            std::cout << std::endl;
            if(active_precision <= target_precision) break;
        }

        //apply best solution
        _prepare_rotation_coefficients_(rotation_matrix,best_yaw,best_roll,best_pitch);
        float a11 = rotation_matrix[0]; float a12 = rotation_matrix[1]; float a13 = rotation_matrix[2];
        float a21 = rotation_matrix[3]; float a22 = rotation_matrix[4]; float a23 = rotation_matrix[5];
        float a31 = rotation_matrix[6]; float a32 = rotation_matrix[7]; float a33 = rotation_matrix[8];

        #pragma omp parallel for
        for(long long int idx = 0; idx < nstack; idx++)
        {
            int z = idx/nslice;
            int y = (idx-z*nslice)/nx;
            int x = idx-z*nslice-y*nx;

            float z0 = z-rotcenter[2]-0.5;
            float y0 = y-rotcenter[1]-0.5;
            float x0 = x-rotcenter[0]-0.5;

            //clockwise
            float x1 = a11*x0 + a12*y0 + a13*z0;
            float y1 = a21*x0 + a22*y0 + a23*z0;
            float z1 = a31*x0 + a32*y0 + a33*z0;

            dx[idx] = dx[idx]+x1-x0;
            dy[idx] = dy[idx]+y1-y0;
            dz[idx] = dz[idx]+z1-z0;
        }

        eliminate_translation(dx, dy, dz, shape, true);

        return;
    }
    void eliminate_rigidbody_motion(float *dx, float *dy, float *dz, uint8_t *mask, int shape[3], uint8_t motion_label, float target_precision = 0.1, float scaling = 10.)
    {
        int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
        long long int nslice = nx*ny;
        long long int nstack = nz*nslice;

        float rotcenter[3] = {shape[0]/2.f, shape[1]/2.f, shape[2]/2.f};

        //first remove translation
        eliminate_translation(dx, dy, dz, mask, shape, motion_label, true);

        //now get the sum of remaining absolute translations and minimize
        float best_yaw = 0.0f;
        float best_roll = 0.0f;
        float best_pitch = 0.0f;
        float best_sum = _vectormagnitude(dx, dy, dz, mask, shape, motion_label);

        std::cout << "optimizing yaw, roll, pitch: " << std::endl;
        std::cout << best_yaw << " " << best_roll << " " << best_pitch << " " << best_sum << "              \r";
        std::cout.flush();

        float rotation_matrix[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        float active_precision = 10.;

        //minimize the remaining absolute sum of translations
        while(0 == 0)
        {
            active_precision /= scaling;
            bool external_change = true;

            while(external_change)
            {
                external_change = false;

                for (int i = 0; i < 6; i++)
                {
                    float increment[3] = {0.0, 0.0, 0.0};
                    if (i == 0) increment[0] = active_precision;
                    else if (i == 1) increment[0] = -active_precision;
                    else if (i == 2) increment[1] = active_precision;
                    else if (i == 3) increment[1] = -active_precision;
                    else if (i == 4) increment[2] = active_precision;
                    else if (i == 5) increment[2] = -active_precision;

                    float active_sum = _test_activerotation(dx, dy, dz, mask, shape, motion_label, best_yaw+increment[0], best_roll+increment[1], best_pitch+increment[2]);

                    if (active_sum < best_sum)
                    {
                        external_change = true;

                        best_yaw += increment[0];
                        best_roll += increment[1];
                        best_pitch += increment[2];
                        best_sum = active_sum;

                        std::cout << best_yaw << " " << best_roll << " " << best_pitch << " " << best_sum << "              \r";
                        std::cout.flush();
                    }
                }
            }

            if(active_precision <= target_precision) break;
        }
        std::cout << std::endl;

        //apply best solution
        _prepare_rotation_coefficients_(rotation_matrix,best_yaw,best_roll,best_pitch);
        float a11 = rotation_matrix[0]; float a12 = rotation_matrix[1]; float a13 = rotation_matrix[2];
        float a21 = rotation_matrix[3]; float a22 = rotation_matrix[4]; float a23 = rotation_matrix[5];
        float a31 = rotation_matrix[6]; float a32 = rotation_matrix[7]; float a33 = rotation_matrix[8];

        #pragma omp parallel for
        for(long long int idx = 0; idx < nstack; idx++)
        {
            int z = idx/nslice;
            int y = (idx-z*nslice)/nx;
            int x = idx-z*nslice-y*nx;

            float z0 = z-rotcenter[2]-0.5;
            float y0 = y-rotcenter[1]-0.5;
            float x0 = x-rotcenter[0]-0.5;

            //clockwise
            float x1 = a11*x0 + a12*y0 + a13*z0;
            float y1 = a21*x0 + a22*y0 + a23*z0;
            float z1 = a31*x0 + a32*y0 + a33*z0;

            dx[idx] = dx[idx]+x1-x0;
            dy[idx] = dy[idx]+y1-y0;
            dz[idx] = dz[idx]+z1-z0;
        }

        eliminate_translation(dx, dy, dz, mask, shape, motion_label, true);

        return;
    }
    void eliminate_rigidbody_motion(std::vector<std::vector<float>> &displacements, std::vector<std::vector<float>> &locations, std::vector<mesh_labeltype> &mesh_labels,
            int shape[3], mesh_labeltype motion_label, float target_precision = 0.1, float scaling = 10.)
    {
        int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
        long long int nslice = nx*ny;
        long long int nstack = nz*nslice;

        float rotcenter[3] = {shape[0]/2.f, shape[1]/2.f, shape[2]/2.f};

        //first remove translation
        eliminate_translation(displacements, mesh_labels, motion_label, true);

        //now get the sum of remaining absolute translations and minimize
        float best_yaw = 0.0f;
        float best_roll = 0.0f;
        float best_pitch = 0.0f;
        float best_sum = _vectormagnitude(displacements, mesh_labels, motion_label);

        std::cout << "optimizing yaw, roll, pitch: " << std::endl;
        std::cout << best_yaw << " " << best_roll << " " << best_pitch << " " << best_sum << "              \r";
        std::cout.flush();

        float rotation_matrix[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        float active_precision = 10.;

        //minimize the remaining absolute sum of translations
        while(0 == 0)
        {
            active_precision /= scaling;
            bool external_change = true;

            while(external_change)
            {
                external_change = false;

                for (int i = 0; i < 6; i++)
                {
                    float increment[3] = {0.0, 0.0, 0.0};
                    if (i == 0) increment[0] = active_precision;
                    else if (i == 1) increment[0] = -active_precision;
                    else if (i == 2) increment[1] = active_precision;
                    else if (i == 3) increment[1] = -active_precision;
                    else if (i == 4) increment[2] = active_precision;
                    else if (i == 5) increment[2] = -active_precision;

                    float active_sum = _test_activerotation(displacements, locations, mesh_labels, shape, motion_label, best_yaw+increment[0], best_roll+increment[1], best_pitch+increment[2]);

                    if (active_sum < best_sum)
                    {
                        external_change = true;

                        best_yaw += increment[0];
                        best_roll += increment[1];
                        best_pitch += increment[2];
                        best_sum = active_sum;

                        std::cout << best_yaw << " " << best_roll << " " << best_pitch << " " << best_sum << "              \r";
                        std::cout.flush();
                    }
                }
            }

            if(active_precision <= target_precision) break;
        }
        std::cout << std::endl;

        //apply best solution
        _prepare_rotation_coefficients_(rotation_matrix,best_yaw,best_roll,best_pitch);
        float a11 = rotation_matrix[0]; float a12 = rotation_matrix[1]; float a13 = rotation_matrix[2];
        float a21 = rotation_matrix[3]; float a22 = rotation_matrix[4]; float a23 = rotation_matrix[5];
        float a31 = rotation_matrix[6]; float a32 = rotation_matrix[7]; float a33 = rotation_matrix[8];

        #pragma omp parallel for
        for(long long int idx = 0; idx < displacements.size(); idx++)
        {
            if(motion_label == 0 || motion_label == mesh_labels[idx])
            {
                float z0 = locations[idx][2]-rotcenter[2]-0.5;
                float y0 = locations[idx][1]-rotcenter[1]-0.5;
                float x0 = locations[idx][0]-rotcenter[0]-0.5;

                //clockwise
                float x1 = a11*x0 + a12*y0 + a13*z0;
                float y1 = a21*x0 + a22*y0 + a23*z0;
                float z1 = a31*x0 + a32*y0 + a33*z0;

                displacements[idx][0] = displacements[idx][0]+x1-x0;
                displacements[idx][1] = displacements[idx][1]+y1-y0;
                displacements[idx][2] = displacements[idx][2]+z1-z0;
            }
            else
            {
                //kill the displacement outside the label of interest
                //displacements[idx][0] = 0.0;
                //displacements[idx][1] = 0.0;
                //displacements[idx][2] = 0.0;

                //or apply
                float z0 = locations[idx][2]-rotcenter[2]-0.5;
                float y0 = locations[idx][1]-rotcenter[1]-0.5;
                float x0 = locations[idx][0]-rotcenter[0]-0.5;

                //clockwise
                float x1 = a11*x0 + a12*y0 + a13*z0;
                float y1 = a21*x0 + a22*y0 + a23*z0;
                float z1 = a31*x0 + a32*y0 + a33*z0;

                displacements[idx][0] = displacements[idx][0]+x1-x0;
                displacements[idx][1] = displacements[idx][1]+y1-y0;
                displacements[idx][2] = displacements[idx][2]+z1-z0;
            }
        }

        eliminate_translation(displacements, mesh_labels, motion_label, true);

        return;
    }
    void eliminate_rigidbody_motion(std::vector<std::vector<float>> &eval_displacements, std::vector<std::vector<float>> &eval_locations,
        std::vector<std::vector<float>> &apply_displacements, std::vector<std::vector<float>> &apply_locations,
        int shape[3], float target_precision = 0.1, float scaling = 10.)
    {
        int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
        long long int nslice = nx*ny;
        long long int nstack = nz*nslice;

        float rotcenter[3] = {shape[0]/2.f, shape[1]/2.f, shape[2]/2.f};

        //first remove translation
        eliminate_translation(eval_displacements, apply_displacements, true);

        //now get the sum of remaining absolute translations and minimize
        float best_yaw = 0.0f;
        float best_roll = 0.0f;
        float best_pitch = 0.0f;
        std::vector<mesh_labeltype> empty_labels;
        int motionlabel = 0;
        float best_sum = _vectormagnitude(eval_displacements, empty_labels, motionlabel);

        std::cout << "optimizing yaw, roll, pitch: " << std::endl;
        std::cout << best_yaw << " " << best_roll << " " << best_pitch << " " << best_sum << "              \r";
        std::cout.flush();

        float rotation_matrix[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        float active_precision = 10.;

        //minimize the remaining absolute sum of translations
        while(0 == 0)
        {
            active_precision /= scaling;
            bool external_change = true;

            while(external_change)
            {
                external_change = false;

                for (int i = 0; i < 6; i++)
                {
                    float increment[3] = {0.0, 0.0, 0.0};
                    if (i == 0) increment[0] = active_precision;
                    else if (i == 1) increment[0] = -active_precision;
                    else if (i == 2) increment[1] = active_precision;
                    else if (i == 3) increment[1] = -active_precision;
                    else if (i == 4) increment[2] = active_precision;
                    else if (i == 5) increment[2] = -active_precision;

                    float active_sum = _test_activerotation(eval_displacements, eval_locations, empty_labels, shape, motionlabel, best_yaw+increment[0], best_roll+increment[1], best_pitch+increment[2]);

                    if (active_sum < best_sum)
                    {
                        external_change = true;

                        best_yaw += increment[0];
                        best_roll += increment[1];
                        best_pitch += increment[2];
                        best_sum = active_sum;

                        std::cout << best_yaw << " " << best_roll << " " << best_pitch << " " << best_sum << "              \r";
                        std::cout.flush();
                    }
                }
            }

            if(active_precision <= target_precision) break;
        }
        std::cout << std::endl;

        //apply best solution
        _prepare_rotation_coefficients_(rotation_matrix,best_yaw,best_roll,best_pitch);
        float a11 = rotation_matrix[0]; float a12 = rotation_matrix[1]; float a13 = rotation_matrix[2];
        float a21 = rotation_matrix[3]; float a22 = rotation_matrix[4]; float a23 = rotation_matrix[5];
        float a31 = rotation_matrix[6]; float a32 = rotation_matrix[7]; float a33 = rotation_matrix[8];

        #pragma omp parallel for
        for(long long int idx = 0; idx < apply_displacements.size(); idx++)
        {
            float z0 = apply_locations[idx][2]-rotcenter[2]-0.5;
            float y0 = apply_locations[idx][1]-rotcenter[1]-0.5;
            float x0 = apply_locations[idx][0]-rotcenter[0]-0.5;

            //clockwise
            float x1 = a11*x0 + a12*y0 + a13*z0;
            float y1 = a21*x0 + a22*y0 + a23*z0;
            float z1 = a31*x0 + a32*y0 + a33*z0;

            apply_displacements[idx][0] = apply_displacements[idx][0]+x1-x0;
            apply_displacements[idx][1] = apply_displacements[idx][1]+y1-y0;
            apply_displacements[idx][2] = apply_displacements[idx][2]+z1-z0;
        }
        #pragma omp parallel for
        for(long long int idx = 0; idx < eval_displacements.size(); idx++)
        {
            float z0 = eval_locations[idx][2]-rotcenter[2]-0.5;
            float y0 = eval_locations[idx][1]-rotcenter[1]-0.5;
            float x0 = eval_locations[idx][0]-rotcenter[0]-0.5;

            //clockwise
            float x1 = a11*x0 + a12*y0 + a13*z0;
            float y1 = a21*x0 + a22*y0 + a23*z0;
            float z1 = a31*x0 + a32*y0 + a33*z0;

            eval_displacements[idx][0] = eval_displacements[idx][0]+x1-x0;
            eval_displacements[idx][1] = eval_displacements[idx][1]+y1-y0;
            eval_displacements[idx][2] = eval_displacements[idx][2]+z1-z0;
        }

        eliminate_translation(eval_displacements, apply_displacements, true);

        return;
    }

    std::vector<std::vector<float>> minimize_translation_along_axis(int dim, int n_support, std::vector<std::vector<float>> &displacements,
        std::vector<std::vector<float>> &coordinates, std::vector<std::vector<float>> axis_cog, float sigma, std::vector<float> &out_inplane_deformation)
    {
        ///////////////////////////////////////////////////////////////////////////////////////////////
        //move a Gaussian window along the given axis and minimize the translation along the other two
        //returns a plot
        ///////////////////////////////////////////////////////////////////////////////////////////////

        //adjust dims
        int xdim = 0;
        int ydim = 1;
        int zdim = 2;

        if (dim == 0) {xdim = 2; zdim = 0;}
        else if (dim == 1) {ydim = 2; zdim = 1;}

        out_inplane_deformation.clear(); out_inplane_deformation.assign(displacements.size(), 0.0);
        std::vector<float> deformation_plot(n_support, 0.0);

        //assuming dim = 2, create a uniform z-axis
        ///////////////////////////////////////////////////////////////////////////////////////////////
        float min_z = 1e9; float max_z = -1e9;
        std::vector<float> position_dim2(n_support, 0.0);
        for (long long int idx = 0; idx < coordinates.size(); idx++)
        {
            if (coordinates[idx][zdim] > max_z) max_z = coordinates[idx][zdim];
            if (coordinates[idx][zdim] < min_z) min_z = coordinates[idx][zdim];
        }
        for (int i = 0; i < n_support; i++)
            position_dim2[i] = ((float)i)/(n_support-1)*(max_z-min_z) + min_z;
        ///////////////////////////////////////////////////////////////////////////////////////////////

        //calculate the mean motion for this axis with a moving average
        ///////////////////////////////////////////////////////////////////////////////////////////////
        std::vector<float> motion_dim0(n_support, 0.0);
        std::vector<float> motion_dim1(n_support, 0.0);
        std::vector<float> weights(n_support, 0.0);
        float sigma2 = sigma*sigma;

        #pragma omp parallel for
        for (int i = 0; i < n_support; i++)
        {
            float z1 = position_dim2[i];

            for (long long int idx = 0; idx < displacements.size(); idx++)
            {
                float x0 = displacements[idx][xdim];
                float y0 = displacements[idx][ydim];
                float z0 = coordinates[idx][zdim];

                float this_weight = z0-z1;
                this_weight = exp(-(0.5f*this_weight*this_weight)/sigma2);
                this_weight = sqrtf(this_weight);

                motion_dim0[i] += x0*this_weight;
                motion_dim1[i] += y0*this_weight;
                weights[i] += this_weight;
            }

            motion_dim0[i] /= std::max(weights[i], 1e-9f);
            motion_dim1[i] /= std::max(weights[i], 1e-9f);
        }
        ///////////////////////////////////////////////////////////////////////////////////////////////

        //subtract the mean motion along the axis
        ///////////////////////////////////////////////////////////////////////////////////////////////
        #pragma omp parallel for
        for (long long int p = 0; p < displacements.size(); p++)
        {
            float x = coordinates[p][xdim];
            float y = coordinates[p][ydim];
            float z = coordinates[p][zdim];

            //linear interpolation
            ////////////////////////////////////////////////////////////////////////////
            float zpos = (z-min_z)/(max_z-min_z)*n_support;
            int zpos0 = std::min(std::max((int) floor(zpos), 0), n_support-1);
            int zpos1 = std::min(std::max((int) ceil(zpos), 0), n_support-1);
            float weight = zpos-zpos0;

            float xval = (1.f-weight)*motion_dim0[zpos0] + weight*motion_dim0[zpos1];
            float yval = (1.f-weight)*motion_dim1[zpos0] + weight*motion_dim1[zpos1];
            float cogx = (1.f-weight)*axis_cog[zpos0][0] + weight*axis_cog[zpos1][0];
            float cogy = (1.f-weight)*axis_cog[zpos0][1] + weight*axis_cog[zpos1][1];
            ////////////////////////////////////////////////////////////////////////////

            displacements[p][xdim] -= xval;
            displacements[p][ydim] -= yval;
            //displacements[p][zdim] = 0.0;

            //magnitude of inplane vector pointing inward or outward
            float dispx = displacements[p][xdim];
            float dispy = displacements[p][ydim];
            float dist0 = sqrtf((x-cogx)*(x-cogx)+(y-cogy)*(y-cogy));
            float dist1 = sqrtf((x-cogx+dispx)*(x-cogx+dispx)+(y-cogy+dispy)*(y-cogy+dispy));
            int direction = dist0 > dist1 ? -1 : 1;
            float magnitude = sqrtf(dispx*dispx + dispy*dispy);
            out_inplane_deformation[p] = (direction*magnitude)/dist0;

        }
        for (int i = 0; i < n_support; i++)
        {
            float magnitude = sqrtf(motion_dim0[i]*motion_dim0[i]+motion_dim1[i]*motion_dim1[i]);
            int direction = motion_dim0[i] < 0 ? -1 : 1;
            deformation_plot[i] = direction*magnitude;
        }
        ///////////////////////////////////////////////////////////////////////////////////////////////

        std::vector<std::vector<float>> output;
        output.push_back(position_dim2);
        output.push_back(motion_dim0);
        output.push_back(motion_dim1);
        output.push_back(deformation_plot);

        return output;
    }
}

#endif // RIGID_BODY_MOTION_H

