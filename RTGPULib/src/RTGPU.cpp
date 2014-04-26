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

#include "RTGPU.h"
#include <qdebug.h>
#include <qstring.h>

void __RTGPUCUDAError( cudaError err, const char *file, const int line )
{
	qDebug() << QString("%1(%2) : kernelSafeCall() Runtime API error : %3")
		.arg(file).arg(line).arg(cudaGetErrorString(err));
	exit(1);
}

void __RTGPUError(const char *emsg, const char *file, const int line )
{
	qDebug() << QString("%1(%2) : kernelError() Runtime error : %3") 
		.arg(file).arg(line).arg(emsg);
	exit(1);
}

void __RTGPUTrace(const char *tmsg, const char *file, const int line )
{
	qDebug() << QString("%1(%2) : %3").arg(file).arg(line).arg(tmsg);
}

void RTGPUCreateImage(Mat& image, CvSize size, bool color, bool pinned)
{
	unsigned char *data;
	int widthStep;
	int dataType;

	if (color)
		dataType = CV_8UC3;
	else
		dataType = CV_8UC1;

	if (!pinned) {
		image = Mat(size, dataType);
		return;
	}
	if (color)
		widthStep = 3 * size.width;
	else
		widthStep = size.width;

	RTGPUSafeCall(cudaHostAlloc((void **)(&data), widthStep * size.height, cudaHostAllocDefault));
	image = Mat(size, dataType, data);
}

void RTGPUReleaseImage(Mat& image, bool pinned)
{
	if (pinned)
		RTGPUSafeCall(cudaFreeHost(image.data));
}

void RTGPUCloneImage(const Mat& src, Mat& dst, bool pinned)
{
	if (!pinned) {
		src.copyTo(dst);
		return;
	}
	
	RTGPUCreateImage(dst, cvSize(src.cols, src.rows), src.type() == CV_8UC3, pinned);
	src.copyTo(dst);
	return;
}

void RTGPUPutImage(int slot, const Mat& image)
{
	_RTGPUPutImage(slot, (unsigned char *)image.data, image.cols, image.rows, image.channels() == 3);
}

void RTGPUGetImage(int slot, Mat& image)
{
	RTGPUSafeCall( cudaThreadSynchronize() );
	_RTGPUGetImage(slot, (unsigned char *)image.data);
}
