#ifndef TRANSFORMATION_CPU_H
#define TRANSFORMATION_CPU_H

namespace transform
{
	void prepare_rotation_coefficients_(float *out_coefficients, float jaw, float roll, float pitch)
	{
		//Preprare rotation coefficients
		float phi = jaw*0.01745329252f;
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

	float linearinterpolation_(float *image, float &x, float &y, float &z, int shape[3])
	{
		int nx = shape[0];
		int ny = shape[1];
		long long int nslice = nx*ny;

		int xf = (int) x; int xc = ceil(x);
		int yf = (int) y; int yc = ceil(y);
		int zf = (int) z; int zc = ceil(z);

		float wx = x-xf;
		float wy = y-yf;
		float wz = z-zf;

		float val, vala, valb, valc, vald;

		if (zf != zc && yf != yc && xf != xc)
		{
			float val000 = image[zf*nslice+yf*nx+xf];
			float val001 = image[zf*nslice+yf*nx+xc];
			float val010 = image[zf*nslice+yc*nx+xf];
			float val100 = image[zc*nslice+yf*nx+xf];
			float val011 = image[zf*nslice+yc*nx+xc];
			float val110 = image[zc*nslice+yc*nx+xf];
			float val101 = image[zc*nslice+yf*nx+xc];
			float val111 = image[zc*nslice+yc*nx+xc];

			vala = (1.f-wz)*val000 + wz*val001;
			valb = (1.f-wz)*val100 + wz*val101;
			valc = (1.f-wz)*val010 + wz*val011;
			vald = (1.f-wz)*val110 + wz*val111;

			vala = (1.f-wx)*vala + wx*valb;
			valb = (1.f-wx)*valc + wx*vald;

			val = (1.f-wy)*vala + wy*valb;
		}
		else if (zf != zc && yf != yc)
		{
			float val000 = image[zf*nslice+yf*nx+xf];
			float val001 = image[zf*nslice+yf*nx+xc];
			float val010 = image[zf*nslice+yc*nx+xf];
			float val011 = image[zf*nslice+yc*nx+xc];

			vala = (1.f-wz)*val000 + wz*val001;
			valb = (1.f-wz)*val010 + wz*val011;

			val = (1.f-wy)*vala + wy*valb;
		}
		else if (zf != zc && xf != xc)
		{
			float val000 = image[zf*nslice+yf*nx+xf];
			float val001 = image[zf*nslice+yf*nx+xc];
			float val100 = image[zc*nslice+yf*nx+xf];
			float val101 = image[zc*nslice+yf*nx+xc];

			vala = (1.f-wz)*val000 + wz*val001;
			valb = (1.f-wz)*val100 + wz*val101;

			val = (1.f-wx)*vala + wx*valb;
		}
		else if (zf != zc)
		{
			vala = image[zf*nslice+yf*nx+xf];
			valb = image[zc*nslice+yf*nx+xf];

			val = (1.f-wz)*vala + wz*valb;
		}
		else if (xf != xc && yf != yc)
		{
			vala = image[zf*nslice+yf*nx+xf];
			valb = image[zf*nslice+yf*nx+xc];
			valc = image[zf*nslice+yc*nx+xf];
			vald = image[zf*nslice+yc*nx+xc];

			vala = (1.f-wx)*vala + wx*valb;
			valb = (1.f-wx)*valc + wx*vald;

			val = (1.f-wy)*vala + wy*valb;
		}
		else if (xf != xc)
		{
			vala = image[zf*nslice+yf*nx+xf];
			valb = image[zf*nslice+yf*nx+xc];

			val = (1.f-wx)*vala + wx*valb;
		}
		else if (yf != yc)
		{
			vala = image[zf*nslice+yf*nx+xf];
			valb = image[zf*nslice+yc*nx+xf];

			val = (1.f-wy)*vala + wy*valb;
		}
		else
			val = image[zf*nslice+yf*nx+xf];

		return val;
	}

	float _interpolate_cubic(float &y0, float &y1, float &y2, float &y3, float &mu)
	{
		float mu2 = mu*mu;

		float a0 = y3-y2-y0+y1;
		float a1 = y0-y1-a0;
		float a2 = y2-y0;
		float a3 = y1;

		return a0*mu*mu2+a1*mu2+a2*mu+a3;
	}
	float cubicinterpolation_(float *image, float &x, float &y, float &z, int shape[3])
	{
		int nx = shape[0];
		int ny = shape[1];
		int nz = shape[2];
		long long int nslice = nx*ny;

		int xf = (int) x; int xc = ceil(x);
		int yf = (int) y; int yc = ceil(y);
		int zf = (int) z; int zc = ceil(z);

		float wx = x-xf;
		float wy = y-yf;
		float wz = z-zf;

		float val, vala, valb, valc, vald;

		if (zf != zc && yf != yc && xf != xc)
		{
			//extrapolate with zero-gradient
			int xf2 = std::max(0, xf-1);
			int xc2 = std::min(xc+1, nx-1);
			int yf2 = std::max(0, yf-1);
			int yc2 = std::min(yc+1, ny-1);
			int zf2 = std::max(0, zf-1);
			int zc2 = std::min(zc+1, nz-1);

			float P100 = image[zf2*nslice+yf2*nx + xf];
			float P200 = image[zf2*nslice+yf2*nx + xc];
			float P101 = image[zf*nslice+yf2*nx + xf];
			float P201 = image[zf*nslice+yf2*nx + xc];
			float P102 = image[zc*nslice+yf2*nx + xf];
			float P202 = image[zc*nslice+yf2*nx + xc];
			float P103 = image[zc2*nslice+yf2*nx + xf];
			float P203 = image[zc2*nslice+yf2*nx + xc];

			float P10 = _interpolate_cubic(P100, P101, P102, P103, wz);
			float P20 = _interpolate_cubic(P200, P201, P202, P203, wz);

			float P010 = image[zf2*nslice+yf*nx + xf2];
			float P110 = image[zf2*nslice+yf*nx + xf];
			float P210 = image[zf2*nslice+yf*nx + xc];
			float P310 = image[zf2*nslice+yf*nx + xc2];
			float P011 = image[zf*nslice+yf*nx + xf2];
			float P111 = image[zf*nslice+yf*nx + xf];
			float P211 = image[zf*nslice+yf*nx + xc];
			float P311 = image[zf*nslice+yf*nx + xc2];
			float P012 = image[zc*nslice+yf*nx + xf2];
			float P112 = image[zc*nslice+yf*nx + xf];
			float P212 = image[zc*nslice+yf*nx + xc];
			float P312 = image[zc*nslice+yf*nx + xc2];
			float P013 = image[zc2*nslice+yf*nx + xf2];
			float P113 = image[zc2*nslice+yf*nx + xf];
			float P213 = image[zc2*nslice+yf*nx + xc];
			float P313 = image[zc2*nslice+yf*nx + xc2];

			float P01 = _interpolate_cubic(P010, P011, P012, P013, wz);
			float P11 = _interpolate_cubic(P110, P111, P112, P113, wz);
			float P21 = _interpolate_cubic(P210, P211, P212, P213, wz);
			float P31 = _interpolate_cubic(P310, P311, P312, P313, wz);

			float P020 = image[zf2*nslice+yc*nx + xf2];
			float P120 = image[zf2*nslice+yc*nx + xf];
			float P220 = image[zf2*nslice+yc*nx + xc];
			float P320 = image[zf2*nslice+yc*nx + xc2];
			float P021 = image[zf*nslice+yc*nx + xf2];
			float P121 = image[zf*nslice+yc*nx + xf];
			float P221 = image[zf*nslice+yc*nx + xc];
			float P321 = image[zf*nslice+yc*nx + xc2];
			float P022 = image[zc*nslice+yc*nx + xf2];
			float P122 = image[zc*nslice+yc*nx + xf];
			float P222 = image[zc*nslice+yc*nx + xc];
			float P322 = image[zc*nslice+yc*nx + xc2];
			float P023 = image[zc2*nslice+yc*nx + xf2];
			float P123 = image[zc2*nslice+yc*nx + xf];
			float P223 = image[zc2*nslice+yc*nx + xc];
			float P323 = image[zc2*nslice+yc*nx + xc2];

			float P02 = _interpolate_cubic(P020, P021, P022, P023, wz);
			float P12 = _interpolate_cubic(P120, P121, P122, P123, wz);
			float P22 = _interpolate_cubic(P220, P221, P222, P223, wz);
			float P32 = _interpolate_cubic(P320, P321, P322, P323, wz);

			float P130 = image[zf2*nslice+yc2*nx + xf];
			float P230 = image[zf2*nslice+yc2*nx + xc];
			float P131 = image[zf*nslice+yc2*nx + xf];
			float P231 = image[zf*nslice+yc2*nx + xc];
			float P132 = image[zc*nslice+yc2*nx + xf];
			float P232 = image[zc*nslice+yc2*nx + xc];
			float P133 = image[zc2*nslice+yc2*nx + xf];
			float P233 = image[zc2*nslice+yc2*nx + xc];

			float P13 = _interpolate_cubic(P130, P131, P132, P133, wz);
			float P23 = _interpolate_cubic(P230, P231, P232, P233, wz);

			float gtu = _interpolate_cubic(P01,P11,P21,P31,wx);
			float gbu = _interpolate_cubic(P02,P12,P22,P32,wx);

			float glv = _interpolate_cubic(P10,P11,P12,P13,wy);
			float grv = _interpolate_cubic(P20,P21,P22,P23,wy);

			float sigma_lr = (1.-wx)*glv + wx*grv;
			float sigma_bt = (1.-wy)*gtu + wy*gbu;
			float corr_lrbt = P11*(1.-wy)*(1.-wx) + P12*wy*(1.-wx) + P21*(1.-wy)*wx + P22*wx*wy;

			val = sigma_lr+sigma_bt-corr_lrbt;
		}
		else if (zf != zc && yf != yc)
		{
			//extrapolate with zero-gradient
			int zf2 = std::max(0, zf-1);
			int zc2 = std::min(zc+1, nz-1);
			int yf2 = std::max(0, yf-1);
			int yc2 = std::min(yc+1, ny-1);

			float P10 = image[zf*nslice+yf2*nx + xf];
			float P20 = image[zc*nslice+yf2*nx + xf];

			float P01 = image[zf2*nslice+yf*nx + xf];
			float P11 = image[zf *nslice+yf*nx + xf];
			float P21 = image[zc *nslice+yf*nx + xf];
			float P31 = image[zc2*nslice+yf*nx + xf];

			float P02 = image[zf2*nslice+yc*nx + xf];
			float P12 = image[zf *nslice+yc*nx + xf];
			float P22 = image[zc *nslice+yc*nx + xf];
			float P32 = image[zc2*nslice+yc*nx + xf];

			float P13 = image[zf*nslice+yc2*nx + xf];
			float P23 = image[zc*nslice+yc2*nx + xf];

			float gtu = _interpolate_cubic(P01,P11,P21,P31,wz);
			float gbu = _interpolate_cubic(P02,P12,P22,P32,wz);

			float glv = _interpolate_cubic(P10,P11,P12,P13,wy);
			float grv = _interpolate_cubic(P20,P21,P22,P23,wy);

			float sigma_lr = (1.-wz)*glv + wz*grv;
			float sigma_bt = (1.-wy)*gtu + wy*gbu;
			float corr_lrbt = P11*(1.-wy)*(1.-wz) + P12*wy*(1.-wz) + P21*(1.-wy)*wx + P22*wz*wy;

			val = sigma_lr+sigma_bt-corr_lrbt;
		}
		else if (zf != zc && xf != xc)
		{
			//extrapolate with zero-gradient
			int zf2 = std::max(0, zf-1);
			int zc2 = std::min(zc+1, nz-1);
			int xf2 = std::max(0, xf-1);
			int xc2 = std::min(xc+1, nx-1);

			float P10 = image[zf*nslice+yf*nx + xf2];
			float P20 = image[zc*nslice+yf*nx + xf2];

			float P01 = image[zf2*nslice+yf*nx + xf];
			float P11 = image[zf *nslice+yf*nx + xf];
			float P21 = image[zc *nslice+yf*nx + xf];
			float P31 = image[zc2*nslice+yf*nx + xf];

			float P02 = image[zf2*nslice+yf*nx + xc];
			float P12 = image[zf *nslice+yf*nx + xc];
			float P22 = image[zc *nslice+yf*nx + xc];
			float P32 = image[zc2*nslice+yf*nx + xc];

			float P13 = image[zf*nslice+yf*nx + xc2];
			float P23 = image[zc*nslice+yf*nx + xc2];

			float gtu = _interpolate_cubic(P01,P11,P21,P31,wz);
			float gbu = _interpolate_cubic(P02,P12,P22,P32,wz);

			float glv = _interpolate_cubic(P10,P11,P12,P13,wy);
			float grv = _interpolate_cubic(P20,P21,P22,P23,wy);

			float sigma_lr = (1.-wz)*glv + wz*grv;
			float sigma_bt = (1.-wx)*gtu + wx*gbu;
			float corr_lrbt = P11*(1.-wx)*(1.-wz) + P12*wx*(1.-wz) + P21*(1.-wx)*wx + P22*wz*wx;

			val = sigma_lr+sigma_bt-corr_lrbt;
		}
		else if (zf != zc)
		{
			int zf2 = std::max(0, zf-1);
			int zc2 = std::min(zc+1, nz-1);

			float P0 = image[zf2*nslice+yf*nx + xf];
			float P1 = image[zf *nslice+yf*nx + xf];
			float P2 = image[zc *nslice+yf*nx + xf];
			float P3 = image[zc2*nslice+yf*nx + xf];

			val = _interpolate_cubic(P0,P1,P2,P3,wz);
		}
		else if (yf != yc && xf != xc)
		{
			//extrapolate with zero-gradient
			int xf2 = std::max(0, xf-1);
			int xc2 = std::min(xc+1, nx-1);
			int yf2 = std::max(0, yf-1);
			int yc2 = std::min(yc+1, ny-1);

			float P10 = image[zf*nslice+yf2*nx + xf];
			float P20 = image[zf*nslice+yf2*nx + xc];

			float P01 = image[zf*nslice+yf*nx + xf2];
			float P11 = image[zf*nslice+yf*nx + xf];
			float P21 = image[zf*nslice+yf*nx + xc];
			float P31 = image[zf*nslice+yf*nx + xc2];

			float P02 = image[zf*nslice+yc*nx + xf2];
			float P12 = image[zf*nslice+yc*nx + xf];
			float P22 = image[zf*nslice+yc*nx + xc];
			float P32 = image[zf*nslice+yc*nx + xc2];

			float P13 = image[zf*nslice+yc2*nx + xf];
			float P23 = image[zf*nslice+yc2*nx + xc];

			float gtu = _interpolate_cubic(P01,P11,P21,P31,wx);
			float gbu = _interpolate_cubic(P02,P12,P22,P32,wx);

			float glv = _interpolate_cubic(P10,P11,P12,P13,wy);
			float grv = _interpolate_cubic(P20,P21,P22,P23,wy);

			float sigma_lr = (1.-wx)*glv + wx*grv;
			float sigma_bt = (1.-wy)*gtu + wy*gbu;
			float corr_lrbt = P11*(1.-wy)*(1.-wx) + P12*wy*(1.-wx) + P21*(1.-wy)*wx + P22*wx*wy;

			val = sigma_lr+sigma_bt-corr_lrbt;
		}
		else if (xf != xc)
		{
			int xf2 = std::max(0, xf-1);
			int xc2 = std::min(xc+1, nx-1);

			float P0 = image[zf *nslice+yf*nx + xf2];
			float P1 = image[zf *nslice+yf*nx + xf];
			float P2 = image[zf *nslice+yf*nx + xc];
			float P3 = image[zf *nslice+yf*nx + xc2];

			val = _interpolate_cubic(P0,P1,P2,P3,wx);
		}
		else if (yf != yc)
		{
			int yf2 = std::max(0, yf-1);
			int yc2 = std::min(yc+1, ny-1);

			float P0 = image[zf *nslice+yf2*nx + xf];
			float P1 = image[zf *nslice+yf*nx + xf];
			float P2 = image[zf *nslice+yc*nx + xf];
			float P3 = image[zf *nslice+yc2*nx + xf];

			val = _interpolate_cubic(P0,P1,P2,P3,wy);
		}
		else val = image[zf*nslice+yf*nx + xf];

		return val;
	}

    float* apply_transformation(float *image, int shape[3], float transform_params[6], float rotcenter[3], int interpolation_order)
    {
    	int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
    	long long int nslice = shape[0]*shape[1];
    	long long int nstack = shape[2]*nslice;

    	float rotation_matrix[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    	prepare_rotation_coefficients_(rotation_matrix, transform_params[3], transform_params[5], transform_params[4]);

    	float a11 = rotation_matrix[0]; float a12 = rotation_matrix[1]; float a13 = rotation_matrix[2];
    	float a21 = rotation_matrix[3]; float a22 = rotation_matrix[4]; float a23 = rotation_matrix[5];
    	float a31 = rotation_matrix[6]; float a32 = rotation_matrix[7]; float a33 = rotation_matrix[8];

    	float dx = transform_params[0]; float dy = transform_params[1]; float dz = transform_params[2];

    	float *output = (float*) calloc(nstack, sizeof(*output));

		#pragma omp parallel for
    	for (long long int idx = 0; idx < nstack; idx++)
    	{
    		int z0 = idx/nslice;
    		int y0 = (idx-z0*nslice)/nx;
    		int x0 = idx-z0*nslice-y0*nx;

    		float x1 = a11*(x0-rotcenter[0]) + a12*(y0-rotcenter[1]) + a13*(z0-rotcenter[2]) + rotcenter[0] - dx;
			float y1 = a21*(x0-rotcenter[0]) + a22*(y0-rotcenter[1]) + a23*(z0-rotcenter[2]) + rotcenter[1] - dy;
			float z1 = a31*(x0-rotcenter[0]) + a32*(y0-rotcenter[1]) + a33*(z0-rotcenter[2]) + rotcenter[2] - dz;

			int z1f = (int) z1; int z1c = ceil(z1);
			int y1f = (int) y1; int y1c = ceil(y1);
			int x1f = (int) x1; int x1c = ceil(x1);

			if(z1f < 0 || y1f < 0 || x1f < 0 || z1c >= nz || y1c >= ny || x1c >= nx)
				continue;

			if (interpolation_order == 1) output[idx] = linearinterpolation_(image, x1, y1, z1, shape);
			else output[idx] = cubicinterpolation_(image, x1, y1, z1, shape);
    	}

    	return output;
    }

}

#endif //TRANSFORMATION_CPU_H
