//
//  Copyright (c) 2014 richards-tech
//
//  This file is part of RTGPULib
//
//  RTGPULib is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  RTGPULib is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with RTGPULib.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef __RTGPUDEFS_H_
#define __RTGPUDEFS_H_

#include "assert.h"

#include <helper_cuda.h>   
#include <helper_math.h>
#include <cuda_runtime_api.h>

#define	GPU_PITCH_UNIT	32								// memory for a row is always a multiple of this

//	GPU image slot defs

#define	USERGPU_MAX_SLOTS	6				// max number of image slots in GPU available to app
#define	INTERNAL_GPU_SLOT0	6				
#define	INTERNAL_GPU_SLOT1	7				
#define	MAX_GPU_SLOTS		8

//	Gaussian kernel defs

#define	MAX_GAUSSIAN_KERNEL_RADIUS	6
#define	MAX_GAUSSIAN_KERNEL_LENGTH	(2 * MAX_GAUSSIAN_KERNEL_RADIUS + 1)

//	Threshold types

#define GPU_THRESH_BINARY		0
#define GPU_THRESH_BINARY_INV	1
#define	GPU_THRESH_TRUNC		2
#define	GPU_THRESH_TOZERO		3
#define	GPU_THRESH_TOZERO_INV	4

//	ArrayOPs tyeps

#define GPU_ARRAYOPS_MINOPT		0					// minimum option
#define	GPU_ARRAYOPS_ABSDIFF	0					// cvAbsDiff equivalent
#define	GPU_ARRAYOPS_CMPEQ		1					// cvCmp with CV_CMP_EQ equivalent
#define	GPU_ARRAYOPS_CMPGT		2					// cvCmp with CV_CMP_GT equivalent
#define	GPU_ARRAYOPS_CMPGE		3					// cvCmp with CV_CMP_GE equivalent
#define	GPU_ARRAYOPS_CMPLT		4					// cvCmp with CV_CMP_LT equivalent
#define	GPU_ARRAYOPS_CMPLE		5					// cvCmp with CV_CMP_LE equivalent
#define	GPU_ARRAYOPS_CMPNE		6					// cvCmp with CV_CMP_NE equivalent
#define	GPU_ARRAYOPS_OR			7					// cvOr equivalent
#define GPU_ARRAYOPS_MAXOPT		7					// maximum option

//	Scalar functions

#define GPU_SCALAROPS_MINOPT	0					// minimum option
#define	GPU_SCALAROPS_SUB		0					// cvSubS equivalent
#define	GPU_SCALAROPS_ADD		1					// cvAdd equivalent
#define	GPU_SCALAROPS_NOT		2					// cvNot equivalent
#define	GPU_SCALAROPS_LUT		3					// cvLUT equivalent (use RTGPUSetLUT to set the table values)
#define GPU_SCALAROPS_MAXOPT	3					// maximum option

//	Morphology functions

#define GPU_MORPHOPS_MINOPT		0					// minimum option
#define	GPU_MORPHOPS_DILATE		0					// cvDilate equivalent
#define	GPU_MORPHOPS_ERODE		1					// cvErode equivalent
#define GPU_MORPHOPS_MAXOPT	3					// maximum option

enum
{
	DISPLAY_GPU,
	DISPLAY_CPU,
	DISPLAY_DIFF,
};

typedef struct
{
	uchar4 *image;											// the image array
	int *inter;												// intermediate image
	int width;												// image width
	int height;												// image height
	bool color;												// true for 3 channels, false for 1
} RTGPU_IMAGE;

#define	RTGPU_SLOTPTR(slot, slotPtr) {assert((slot < MAX_GPU_SLOTS) && (slot >= 0)); slotPtr = g_images + slot;}

extern	RTGPU_IMAGE	g_images[MAX_GPU_SLOTS];

#define RTGPUSafeCall(err)				__RTGPUSafeCall(err, __FILE__, __LINE__)
#define RTGPUError(emsg)				__RTGPUError(emsg, __FILE__, __LINE__)

#ifdef	_DEBUG
#define RTGPUTrace(tmsg)		//		__RTGPUTrace(tmsg, __FILE__, __LINE__)
#else
#define	RTGPUTrace(tmsg)
#endif

#define	RTGPU_MIN(x, min) {if (x < min) min = x;}
#define	RTGPU_MIN4(val, min) {RTGPU_MIN(val.x, min.x); RTGPU_MIN(val.y, min.y); RTGPU_MIN(val.z, min.z); RTGPU_MIN(val.w, min.w); }
#define	RTGPU_MAX(x, max) {if (x > max) max = x;}
#define	RTGPU_MAX4(val, max) {RTGPU_MAX(val.x, max.x); RTGPU_MAX(val.y, max.y); RTGPU_MAX(val.z, max.z); RTGPU_MAX(val.w, max.w); }
#define	RTGPU_MINMAX(x, min, max) {if (x < min) min = x; if (x > max) max = x;}
#define	RTGPU_MINMAX4(val, min, max) {RTGPU_MINMAX(val.x,min.x,max.x);RTGPU_MINMAX(val.y,min.y,max.y);RTGPU_MINMAX(val.z,min.z,max.z);RTGPU_MINMAX(val.w,min.w,max.w);}

void __RTGPUCUDAError( cudaError err, const char *file, const int line );
void __RTGPUError(const char *emsg, const char *file, const int line );
void __RTGPUTrace(const char *tmsg, const char *file, const int line );

inline void __RTGPUSafeCall( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) 
	{
		__RTGPUCUDAError(err, file, line);
    }
}


class RTGPUScalar
{
public:

	RTGPUScalar(float val0, float val1=0, float val2=0, float val3=0)
	{
		val[0] = val0; val[1] = val1;
		val[2] = val2; val[3] = val3;
	}

	RTGPUScalar(float val0123)
	{
		val[0] = val0123;
		val[1] = val0123;
		val[2] = val0123;
		val[3] = val0123;
	}

    float val[4];
};

extern "C"	void _RTGPUInit(int maxWidth, int maxHeight);
extern "C"	void _RTGPUClose();
extern "C"	void _RTGPUSetupSlot(RTGPU_IMAGE *rI, int w, int h, bool color);

extern "C"	int _RTGPUPutImage(int slot, unsigned char *image, int w, int h, bool color);
extern "C"	int _RTGPUGetImage(int slot, unsigned char *image);
extern "C"	int _RTGPUGetImageWidth(int slot);
extern "C"	int _RTGPUGetImageHeight(int slot);
extern "C"	int _RTGPUGetImageChannels(int slot);

extern "C"	int _RTGPUSobel(int srcSlot, int destSlot);
extern "C"	int _RTGPUMedianBlur(int srcSlot, int destSlot, int rad);
extern "C"	int _RTGPUMedianBlur3x3(int srcSlot, int destSlot);
extern "C"	int _RTGPUThreshold(int srcSlot, int destSlot, int thresh, int maxVal, int type); 
extern "C"	int _RTGPUAdaptiveThreshold(int srcSlot, int destSlot, int maxVal, int method, int type, int blockSize, int delta); 
extern "C"	int _RTGPUSetGaussian(int rad);
extern "C"	int _RTGPUBoxFilter(int srcSlot, int destSlot, int rad, bool normalize);
extern "C"	int _RTGPURGB2GRAY(int srcSlot, int destSlot);
extern "C"	int _RTGPUGRAY2RGB(int srcSlot, int destSlot);
extern "C"	int _RTGPUCreateConvTex(int rad); 
extern "C"	int _RTGPUConvTex(int srcSlot, int sestSlot, int rad); 
extern "C" void _RTGPUSetConvKernel(float *kernel, int rad);

extern "C" void _RTGPUSetGaussianKernel(float *kernel, int rad);
extern "C"	int _RTGPUGaussian(int srcSlot, int destSlot, int rad);
extern "C" void _RTGPUConvolutionRowsGPU(uchar4 *src, float *inter, int imageW, int imageH, bool color);
extern "C" void _RTGPUConvolutionColumnsGPU(float *inter, uchar4 *dst, int imageW, int imageH, bool color);

extern "C" int _RTGPUArrayOps(int srcSlotA, int srcSlotB, int destSlot, int maskSlot, int type);
extern "C" int _RTGPUScalarOps(int srcSlotA, RTGPUScalar scalar, int destSlot, int maskSlot, int type);
extern "C" int _RTGPUCreateLUT(unsigned char *LUT); 

extern "C" int _RTGPUMorphOps3x3(int srcSlot, int destSlot, int mode); 

extern "C" bool _RTGPUJPEGCompress(int srcSlot, unsigned char *outBuf, int& outputBytes);
extern "C" void _RTGPUJPEGInitEncoder( );

extern "C" bool _RTGPUReadJPEGHeader(unsigned char *inBuf, int cbInBuf, int& width,
								  int& height, int& headSize);
extern "C" bool _RTGPUJPEGDecompress(unsigned char *inBuf, int destSlot);

#endif

