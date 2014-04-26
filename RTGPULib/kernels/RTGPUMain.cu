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

#include "RTGPUDefs.h"

texture<unsigned char, 2> g_tex0;						// the char texture
texture<uchar4, 2> g_tex1;								// the texture
texture<uchar4, 2> g_tex2;								// if there's more than one source image

RTGPU_IMAGE	g_images[MAX_GPU_SLOTS];


//-----------------------------------------------------------------------------
//
//	RTGPU Utility and Control Functions


extern "C" void _RTGPUInit(int maxWidth, int maxHeight)
{
	int		i;
	RTGPU_IMAGE	*RI;
	
	RI = g_images;
	
	RTGPUTrace("RTGPUInit");
	for (i = 0; i < MAX_GPU_SLOTS; i++, RI++)
	{
		RTGPUSafeCall(cudaMalloc(&(RI->image), maxWidth * maxHeight * sizeof(uchar4)));
		RTGPUSafeCall(cudaMalloc(&(RI->inter), maxWidth * maxHeight * 4 * sizeof(int)));
		RI->width = maxWidth;
		RI->height = maxHeight;
		RI->color = false;
	}
}

extern "C" void _RTGPUClose()
{
	int		i;
	RTGPU_IMAGE	*RI;
	
	RTGPUTrace("RTGPUClose");
	RI = g_images;
	
	for (i = 0; i < MAX_GPU_SLOTS; i++, RI++)
	{
		if (RI->image != NULL)
		{
			RTGPUSafeCall(cudaFree(RI->image));
			RI->image = NULL;
		}
		if (RI->inter != NULL)
		{
			RTGPUSafeCall(cudaFree(RI->inter));
			RI->inter = NULL;
		}
		RI->width = 0;
		RI->height = 0;
	}
}

extern "C" void _RTGPUSetupSlot(RTGPU_IMAGE *RI, int w, int h, bool color)
{
 	RTGPUTrace("RTGPUSetupSlot");

	RI->width = w;
	RI->height = h;
	RI->color = color;
}

//-----------------------------------------------------------------------------
//
//	Format conversion functions

__global__ void kernelRGB2GRAY( uchar4 *input, unsigned char *output, int w)
{ 
	uchar4	val;
	float	res;
	
    unsigned char *out = output + blockIdx.x * w;
    for ( int i = threadIdx.x; i < w; i += blockDim.x) 
    {
		val =	 tex2D(g_tex1, i, blockIdx.x);
		res =	(float)0.11 * (float)(val.x) +			// red
				(float)0.59 * (float)(val.y) +			// green
				(float)0.3 * (float)(val.z);			// blue
				
		if (res > (float)255.0)
			res = (float)255.0;
		out[i] = (unsigned char)res;
    }
}


extern "C" int _RTGPURGB2GRAY(int srcSlot, int destSlot) 
{
	RTGPU_IMAGE	*SI, *DI;
    cudaChannelFormatDesc desc;

	RTGPU_SLOTPTR(srcSlot, SI);
	RTGPU_SLOTPTR(destSlot, DI);

	RTGPUTrace("RTGPURGB2GRAY");
	assert (SI->color);
	_RTGPUSetupSlot(DI, SI->width, SI->height, false);

	desc = cudaCreateChannelDesc<uchar4>();
	RTGPUSafeCall(cudaBindTexture2D(NULL, g_tex1, SI->image, desc, SI->width, SI->height, SI->width * 4));
	
	kernelRGB2GRAY<<<SI->height, 32>>>(SI->image, (unsigned char *)DI->image, SI->width);

 	RTGPUSafeCall(cudaUnbindTexture(g_tex1));

	return 1;
}

__global__ void kernelGRAY2RGB( unsigned char *input, uchar4 *output, int w)
{ 
	unsigned char	val;
    uchar4	*out = output + blockIdx.x * w;
    
    for ( int i = threadIdx.x; i < w; i += blockDim.x) 
    {
		val = tex2D(g_tex0, i, blockIdx.x);
		out[i].x = val;
		out[i].y = val;
		out[i].z = val;
    }
}

extern "C" int _RTGPUGRAY2RGB(int srcSlot, int destSlot) 
{
	RTGPU_IMAGE	*SI, *DI;
    cudaChannelFormatDesc desc;

	RTGPU_SLOTPTR(srcSlot, SI);
	RTGPU_SLOTPTR(destSlot, DI);
	
	RTGPUTrace("RTGPUGRAY2RGB");
	
	assert(!SI->color);
	_RTGPUSetupSlot(DI, SI->width, SI->height, true);

	desc = cudaCreateChannelDesc<unsigned char>();
	RTGPUSafeCall(cudaBindTexture2D(NULL, g_tex0, SI->image, desc, SI->width, SI->height, SI->width));

	kernelGRAY2RGB<<<SI->height, 32>>>((unsigned char *)SI->image, DI->image, SI->width);

  	RTGPUSafeCall(cudaUnbindTexture(g_tex0));
	return 1;
}

//-----------------------------------------------------------------------------
//
//	Image put and get functions


__global__ void kernelPutRGB(uchar4 *output, int w)
{
    uchar4	*out = output + blockIdx.x * w;
    uchar4	res;
    int		i, j;

    for (i = threadIdx.x, j = 3 * threadIdx.x; i < w; i += blockDim.x, j += 3 * blockDim.x) {
		res.x = tex2D(g_tex0, j, blockIdx.x);
		res.y = tex2D(g_tex0, j + 1, blockIdx.x);
		res.z = tex2D(g_tex0, j + 2, blockIdx.x);
		out[i] = res;
    }
}

extern "C" int _RTGPUPutImage(int slot, unsigned char *image, int w, int h, bool color)
{
	RTGPU_IMAGE	*slotPtr;
	cudaChannelFormatDesc desc;
	cudaArray *arrayTemp;

	
	RTGPUTrace("RTGPUPutImage");
	RTGPU_SLOTPTR(slot, slotPtr);
	_RTGPUSetupSlot(slotPtr, w, h, color);

	if (!color) {	
		RTGPUSafeCall(cudaMemcpy(slotPtr->image, image, w * h, cudaMemcpyHostToDevice));
		return 1;
	}

	desc = cudaCreateChannelDesc<unsigned char>();
	RTGPUSafeCall(cudaMallocArray(&arrayTemp, &desc, w * 3, h));
	RTGPUSafeCall(cudaMemcpyToArray(arrayTemp, 0, 0, image, w * h * 3, cudaMemcpyHostToDevice));
	RTGPUSafeCall(cudaBindTextureToArray(g_tex0, arrayTemp));
	kernelPutRGB<<<h, 32>>>((uchar4 *)slotPtr->image, w);
	RTGPUSafeCall(cudaUnbindTexture(g_tex0));
	RTGPUSafeCall(cudaFreeArray(arrayTemp));

	return 1;
}

__global__ void kernelGetRGB(uchar4 *input, unsigned char *output, int w)
{
    unsigned char *out = output + blockIdx.x * w * 3;
    uchar4	*in = input + blockIdx.x * w;
    uchar4	val;
    int		i, j;
    
    for (i = threadIdx.x, j = 3 * threadIdx.x; i < w; i += blockDim.x, j += 3 * blockDim.x) 
    {
		val = in[i];
		out[j + 0] = val.x;
		out[j + 1] = val.y;
		out[j + 2] = val.z;
    }
}

extern "C" int _RTGPUGetImage(int slot, unsigned char *image)
{
	RTGPU_IMAGE	*slotPtr;
	int		w, h;

	RTGPUTrace("RTGPUGetImage");
	slotPtr = g_images + slot;
	
	if (slotPtr->image == NULL)
		return 0;
		
	w = slotPtr->width;
	h = slotPtr->height;

	if (!slotPtr->color) {
		RTGPUSafeCall(cudaMemcpy(image, slotPtr->image, w * h, cudaMemcpyDeviceToHost));
		return 1;
	}
	kernelGetRGB<<<h, 32>>>((uchar4 *)slotPtr->image, (unsigned char *)slotPtr->inter, w);
	RTGPUSafeCall(cudaMemcpy(image, slotPtr->inter, 3 * w * h, cudaMemcpyDeviceToHost));
    return 1;
}
