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

HEADERS += ./CameraIF.h \
    ./RTGPUTestBench.h \
    ./RTGPUTestThread.h \
    ../RTGPULib/header/RTGPU.h \
    ../RTGPULib/header/RTGPUDefs.h

SOURCES += ./CameraIF.cpp \
    ./main.cpp \
    ./RTGPUTestBench.cpp \
    ./RTGPUTestThread.cpp

FORMS += ./RTGPUTestBench.ui
RESOURCES +=
