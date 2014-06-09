#//
#//  Copyright (c) 2014 richards-tech.
#//
#//  This file is part of RTGPULib
#//
#//  RTGPULib is free software: you can redistribute it and/or modify
#//  it under the terms of the GNU General Public License as published by
#//  the Free Software Foundation, either version 3 of the License, or
#//  (at your option) any later version.
#//
#//  RTGPULib is distributed in the hope that it will be useful,
#//  but WITHOUT ANY WARRANTY; without even the implied warranty of
#//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#//  GNU General Public License for more details.
#//
#//  You should have received a copy of the GNU General Public License
#//  along with RTGPULib.  If not, see <http://www.gnu.org/licenses/>.
#//

INCLUDEPATH += $$PWD/header
DEPENDPATH += $$PWD/header $$PWD/kernels $$PWD/src

HEADERS += $$PWD/header/helper_cuda.h \
	$$PWD/header/helper_math.h \
        $$PWD/header/helper_string.h \
        $$PWD/header/RTGPU.h \
	$$PWD/header/RTGPUDefs.h

SOURCES += $$PWD/src/RTGPU.cpp

CUDA_SOURCES += $$PWD/kernels/arrayOps.cu \
        $$PWD/kernels/boxFilter.cu \
        $$PWD/kernels/convKernel.cu \
        $$PWD/kernels/convolutionSeparable.cu \
        $$PWD/kernels/edge.cu \
        $$PWD/kernels/JPEGDecoder.cu \
        $$PWD/kernels/JPEGEncoder.cu \
        $$PWD/kernels/median.cu \
        $$PWD/kernels/median3x3.cu \
        $$PWD/kernels/morphOps.cu \
        $$PWD/kernels/RTGPUMain.cu \
        $$PWD/kernels/scalarOps.cu \
        $$PWD/kernels/thresholdKernel.cu

SOURCES += CUDA_SOURCES
SOURCES -= CUDA_SOURCES

