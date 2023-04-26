#ifndef RIGID_BODY_MOTION_H
#define RIGID_BODY_MOTION_H

#include <iostream>
#include <string.h>
#include <cstdint>
#include <vector>
#include "../Geometry/auxiliary.h"

namespace rbmotion
{
	typedef float img_type;

    void eliminate_translation(img_type *vectorfield, int shape[3], bool console_print)
    {
        int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
        long long int nslice = nx*ny;
        long long int nstack = nz*nslice;

        double ev_x = 0.0; double ev_y = 0.0; double ev_z = 0.0;

        //get average displacements
        #pragma omp parallel for reduction(+: ev_x, ev_y, ev_z)
        for(long long int idx = 0; idx < nstack; idx++)
        {
            ev_x += vectorfield[idx];
            ev_y += vectorfield[nstack+idx];
            ev_z += vectorfield[2*nstack+idx];
        }

        ev_x /= nstack; ev_y /= nstack; ev_z /= nstack;

        //and subtract
        if(console_print)
            std::cout << "average translation: " << round(ev_x*100.)/100. << ", " << round(ev_y*100.)/100. << ", " << round(ev_z*100.)/100. << std::endl;

        #pragma omp parallel for
        for(long long int idx = 0; idx < nstack; idx++)
        {
        	vectorfield[idx] -= ev_x;
        	vectorfield[nstack+idx] -= ev_y;
        	vectorfield[2*nstack+idx] -= ev_z;
        }

        return;
    }
    void eliminate_translation(img_type *vectorfield, img_type *mask, int shape[3], img_type motion_label = 0, bool console_print = false)
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
            	ev_x += vectorfield[idx];
				ev_y += vectorfield[nstack+idx];
				ev_z += vectorfield[2*nstack+idx];
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
        	vectorfield[idx] -= ev_x;
			vectorfield[nstack+idx] -= ev_y;
			vectorfield[2*nstack+idx] -= ev_z;
        }

        return;
    }

    double _vectormagnitude(img_type *vectorfield, int shape[3])
    {
        int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
        long long int nslice = nx*ny;
        long long int nstack = nz*nslice;

        double sum = 0.0;

        #pragma omp parallel for reduction(+: sum)
        for(long long int idx = 0; idx < nstack; idx++)
        {
        	img_type valx = vectorfield[idx];
        	img_type valy = vectorfield[nstack+idx];
        	img_type valz = vectorfield[2*nstack+idx];
            sum += sqrtf(valx*valx+valy*valy+valz*valz);
        }

        return sum;
    }
    double _vectormagnitude(img_type *vectorfield, img_type* mask, int shape[3], img_type motion_label)
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
            	img_type valx = vectorfield[idx];
            	img_type valy = vectorfield[nstack+idx];
            	img_type valz = vectorfield[2*nstack+idx];
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

    float _test_activerotation(img_type *vectorfield, int shape[3], float yaw, float roll, float pitch)
    {
        float rotcenter[3] = {shape[0]/2.f, shape[1]/2.f, shape[2]/2.f};

        int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
        long long int nslice = nx*ny;
        long long int nstack = nz*nslice;

        img_type* next_vectorfield = (img_type*) calloc(3*nstack, sizeof(*next_vectorfield));

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

            next_vectorfield[idx] = vectorfield[idx]+x1-x0;
            next_vectorfield[nstack+idx] = vectorfield[nstack+idx]+y1-y0;
            next_vectorfield[2*nstack+idx] = vectorfield[2*nstack+idx]+z1-z0;
        }

        eliminate_translation(next_vectorfield, shape, false);

        float next_sum = _vectormagnitude(next_vectorfield, shape);

        free(next_vectorfield);

        return next_sum;
    }
    float _test_activerotation(img_type *vectorfield, img_type* mask, int shape[3], img_type motion_label, float yaw, float roll, float pitch)
    {
        float rotcenter[3] = {shape[0]/2.f, shape[1]/2.f, shape[2]/2.f};

        int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
        long long int nslice = nx*ny;
        long long int nstack = nz*nslice;

        img_type* next_vectorfield = (img_type*) calloc(3*nstack, sizeof(*next_vectorfield));

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
                next_vectorfield[idx] = vectorfield[idx]+x1-x0;
                next_vectorfield[nstack+idx] = vectorfield[nstack+idx]+y1-y0;
                next_vectorfield[2*nstack+idx] = vectorfield[2*nstack+idx]+z1-z0;
            }
        }

        eliminate_translation(next_vectorfield, shape, false);

		float next_sum = _vectormagnitude(next_vectorfield, shape);

		free(next_vectorfield);

        return next_sum;
    }

    void eliminate_rigidbody_motion(img_type *vectorfield, int shape[3], float target_precision = 0.1, float scaling = 10.)
    {
        int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
        long long int nslice = nx*ny;
        long long int nstack = nz*nslice;

        float rotcenter[3] = {(float) (shape[0]/2.f), (float) (shape[1]/2.f), (float) (shape[2]/2.f)};

        //first remove translation
        eliminate_translation(vectorfield, shape, true);

        //now get the sum of remaining absolute translations and minimize
        float best_yaw = 0.0f;
        float best_roll = 0.0f;
        float best_pitch = 0.0f;
        float best_sum = _vectormagnitude(vectorfield, shape);

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

                    float active_sum = _test_activerotation(vectorfield, shape, best_yaw+increment[0], best_roll+increment[1], best_pitch+increment[2]);

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

            vectorfield[idx] = vectorfield[idx]+x1-x0;
            vectorfield[nstack+idx] = vectorfield[nstack+idx]+y1-y0;
            vectorfield[2*nstack+idx] = vectorfield[2*nstack+idx]+z1-z0;
        }

        eliminate_translation(vectorfield, shape, true);

        return;
    }
    void eliminate_rigidbody_motion(img_type* vectorfield, img_type *mask, int shape[3], img_type motion_label, float target_precision = 0.1, float scaling = 10.)
    {
        int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
        long long int nslice = nx*ny;
        long long int nstack = nz*nslice;

        float rotcenter[3] = {shape[0]/2.f, shape[1]/2.f, shape[2]/2.f};

        //first remove translation
        eliminate_translation(vectorfield, mask, shape, motion_label, true);

        //now get the sum of remaining absolute translations and minimize
        float best_yaw = 0.0f;
        float best_roll = 0.0f;
        float best_pitch = 0.0f;
        float best_sum = _vectormagnitude(vectorfield, mask, shape, motion_label);

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

                    float active_sum = _test_activerotation(vectorfield, mask, shape, motion_label, best_yaw+increment[0], best_roll+increment[1], best_pitch+increment[2]);

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

            vectorfield[idx] = vectorfield[idx]+x1-x0;
            vectorfield[nstack+idx] = vectorfield[nstack+idx]+y1-y0;
            vectorfield[2*nstack+idx] = vectorfield[2*nstack+idx]+z1-z0;
        }

        eliminate_translation(vectorfield, mask, shape, motion_label, true);

        return;
    }
    void eliminate_rigidbody_motion_definedrotation(img_type* vectorfield, img_type* mask, int shape[3], img_type motion_label, std::vector<float> rotation)
	{
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nz*nslice;

		float rotcenter[3] = {shape[0]/2.f, shape[1]/2.f, shape[2]/2.f};

		//first remove translation
		eliminate_translation(vectorfield, mask, shape, motion_label, true);

		float rotation_matrix[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
		_prepare_rotation_coefficients_(rotation_matrix,rotation[0],rotation[1],rotation[2]);
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

			vectorfield[idx] = vectorfield[idx]+x1-x0;
			vectorfield[nstack+idx] = vectorfield[nstack+idx]+y1-y0;
			vectorfield[2*nstack+idx] = vectorfield[2*nstack+idx]+z1-z0;
		}

		eliminate_translation(vectorfield, mask, shape, motion_label, true);

		return;
	}
}

#endif // RIGID_BODY_MOTION_H
