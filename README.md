# RTGPULib and RTGPUTestBench

RTGPULib is a convenient library for implementing image processing functions using NVIDIA's CUDA technology. RTGPUTestBench demonstrates how to use the library and also 
provides an easy way of comparing the performance of a CPU implementation of an algorithm against a GPU implementation of the same algorithm. RTGPUTestBench implements a 
number of common image processing functions in CUDA kernels and compares the performance against the OpenCV CPU implementation of the same function.

Currently RTGPUTestBench is only supported on Windows (VS2010), Linux (Ubuntu) and Jetson TK1 but RTGPULib itself is not platform-dependent.

Check out www.richards-tech.com for more details.

## Windows 

### Dependencies

1. Qt4 or Qt5 development libraries and headers
2. QT plugin for VS
3. OpenCV V2.4.8 (http://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.8/opencv-2.4.8.exe/download)
4. CUDA 6.0 for Windows (https://developer.nvidia.com/cuda-downloads)
5. A webcam to act as the source of images.

### Build

The repo includes a VS2010 solution file that can be used to build RTGPUTestBench. It assumes that the environment contains a symbol OPENCVDIR that 
points to where the prebuilt OpenCV implements resides. For example, if the download was decompressed into c:\opencv-2.4.8 then use:

	OPENCVDIR = c:\opencv-2.4.8\build

If it is desired to use a different version of OpenCV, the project file must be changed to link the appropriate libraries (unfortunately OpenCV 
codes the version number into the library name).

There is a similar issue with CUDA. If a version other that 6.0 is used, the project file will need to be edited as the version number is coded into the props data. Just 
change the 6.0 to whatever the correct number happens to be.

RTGPUTestBench can be built in Debug and Release modes.

## Ubuntu

### Dependencies

build-essential, Qt, QtCreator and opencv are needed:

	sudo apt-get install build-essential qt5-default qtcreator libopencv-dev

The CUDA SDK is also needed. The code was tested with CUDA 6.0 which can be downloaded from https://developer.nvidia.com/cuda-downloads.

### Build

To just build, navigate to the RTGPUTestBench directory and enter:

	qmake
	make
	sudo make install

To build using QtCreator, start QtCreator and open the RTGPUTestBench.pro file. This can be used to set breakpoints and other thinsg as needed.

## NVIDIA Jetson TK1

### Dependencies

build-essential, Qt and QtCreator are needed:

	sudo apt-get install build-essential qt5-default qtcreator

The CUDA toolkit and Tegra OpenCV should be downloaded from https://developer.nvidia.com/jetson-tk1-support.

A soft link is need to make the cuda-6.0.pc file available. Enter:

	sudo ln -s /usr/lib/pkgconfig/cuda-6.0.pc /usr/lib/pkgconfig/cuda.pc

There also appears to be an error in cuda-6.0.pc. The cudaroot variable needs to be:

	cudaroot=/usr/local/cuda-6.0

Use any editor to make this change.

### Build

To just build, navigate to the RTGPUTestBench directory and enter:

	qmake
	make
	sudo make install

To build using QtCreator, start QtCreator and open the RTGPUTestBench.pro file. This can be used to set breakpoints and other things as needed.

## Run

By default, the software will use the first webcam it finds (OpenCV device 0). If this is not the correct device, CameraIF.cpp can be modified to choose a different source. If there's no webcam, the program will crash! You should see the "Camera available" message appear in the status line if everything is ok.

The general process is to select a test from the drop down box, enter any parameters that might be required (such as a kernel width) and then press the "Run once" button to get a single execution 
or "Run continuously" to run the test at a maximum of around 10 tests per second. The average execution time is accumulated in continuous mode.

## Some details of RTGPULib

The general concept revolves around image slots in device memory. By default, there are six user slots and two internal slots. Kernels generally take one or two slot 
numbers as parameters. The idea is that images can remain in device memory across multiple kernel calls in a standardized way. This simplifies building up complex sequences 
up functions from primitives.

For example:

	RTGPUPutImage(0, m_imageInGray);

would copy the contents of the (openCV Mat) image in m_imageRGB in to GPU slot 0. Then:

	_RTGPUThreshold(0, 1, thresh, 255, GPU_THRESH_BINARY);

can be used to create a threholded version of the image in slot 0 and place the results in slot 2. Then:

	RTGPUGetImage(1, m_gpuImageOutGray);

would copy the result from device memory to host memory for display.

Check out RTGPUTestThread.cpp for more examples of how this all works.
