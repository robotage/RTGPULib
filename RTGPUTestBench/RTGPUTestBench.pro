#
#  Copyright (c) 2014 richards-tech
#	
#  This file is part of RTGPULib
#
#  RTGPULib is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  RTGPULib is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with RTGPULib.  If not, see <http://www.gnu.org/licenses/>.
#

greaterThan(QT_MAJOR_VERSION, 4): cache()

TEMPLATE = app
TARGET = RTGPUTestBench

DESTDIR = Output 

QT += core network gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += debug_and_release

INSTALLS += target
target.path = /usr/bin
CONFIG += link_pkgconfig
PKGCONFIG += opencv cuda

DEFINES += QT_NETWORK_LIB

INCLUDEPATH += ./../QRDecode \
   ./../RTGPULib/header \
    ./GeneratedFiles \
    . \
    ./GeneratedFiles/Debug

MOC_DIR += GeneratedFiles/moc

OBJECTS_DIR += objects

UI_DIR += GeneratedFiles

RCC_DIR += GeneratedFiles

CUDA_DIR = /usr/local/cuda
INCLUDEPATH += $$CUDA_DIR/include
contains(QMAKE_HOST.arch, x86_64) {
    QMAKE_LIBDIR += $$CUDA_DIR/lib64
} else {
    QMAKE_LIBDIR += $$CUDA_DIR/lib
}
LIBS += -lcudart -lcuda
CUDA_ARCH = sm_20
NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

contains(QMAKE_HOST.arch, x86_64) {
    cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
                $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
# nvcc error printout format ever so slightly different from gcc
# http://forums.nvidia.com/index.php?showtopic=171651
                2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
} else {
    cuda.commands = $$CUDA_DIR/bin/nvcc -m32 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
                $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
# nvcc error printout format ever so slightly different from gcc
# http://forums.nvidia.com/index.php?showtopic=171651
                2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
}

cuda.dependency_type = TYPE_C # there was a typo here. Thanks workmate!
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}

cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda

include(../RTGPULib/RTGPULib.pri)
include(RTGPUTestBench.pri)
