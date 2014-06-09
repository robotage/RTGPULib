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

#include "RTGPUTestThread.h"
#include "CameraIF.h"

#include <QDebug>
#include <qbuffer.h>

#define	GSTART		{m_gpuStartTime = m_elapsedTimer.nsecsElapsed(); }
#define	GSTOP		{RTGPUSafeCall(cudaThreadSynchronize()); m_gpuStopTime = m_elapsedTimer.nsecsElapsed(); }
#define	CSTART		m_cpuStartTime = m_elapsedTimer.nsecsElapsed();
#define	CSTOP		m_cpuStopTime = m_elapsedTimer.nsecsElapsed();

//	When adding a new function, create another entry in this array and update testFunctionCount

TESTFUNCTION RTGPUTestThread::testArray[] = {
									{"2D Sobel", &RTGPUTestThread::testSobel, "No params"},
									{"AdaptiveThreshold", &RTGPUTestThread::testAdaptiveThreshold, "<Kernel width (odd number, default 3)>"},
									{"Array ops", &RTGPUTestThread::testArrayOps, "<op code (0-7> <1 = use mask>"},
									{"Box Filter", &RTGPUTestThread::testBoxFilter, "<Kernel width (odd number, default 3)>"},
									{"Gaussian blur using shared memory", &RTGPUTestThread::testGaussianShared, "<Kernel width (odd number, default 3)>"},
									{"Gaussian blur using texture", &RTGPUTestThread::testGaussianTexture, "<Kernel width (odd number, default 3)>"},
									{"GPU image put", &RTGPUTestThread::testImagePut, "No params"},
									{"GPU image get", &RTGPUTestThread::testImageGet, "No params"},
									{"JPEG Encoder", &RTGPUTestThread::testJPEGEncoder, "No params"},
									{"JPEG Decoder", &RTGPUTestThread::testJPEGDecoder, "No params"},
									{"Median blur", &RTGPUTestThread::testMedianBlur, "<Kernel width (odd number, default 3)>"},
									{"Morph ops", &RTGPUTestThread::testMorphOps, "<0 for dilate, 1 for erode>"},
									{"RGB->GRAY->RGB", &RTGPUTestThread::testRGB2GRAY, "No params"},
									{"Scalar ops", &RTGPUTestThread::testScalarOps, "<op code (0-3> <val0> <val1> <val2>"},
									{"Threshold", &RTGPUTestThread::testThreshold, "<threshold (0-255)>"},
								};

int RTGPUTestThread::testFunctionCount = 15;			// number of entries in the gpTestArray

RTGPUTestThread::RTGPUTestThread() : QObject()
{
    m_timer = -1;
	m_frameWidth = 640;
	m_frameHeight = 480;
}

RTGPUTestThread::~RTGPUTestThread()
{

}

void RTGPUTestThread::initThread()
{

	m_testRun = false;
	m_runInProgress = false;
	m_runContFirst = false;

	if (m_camera.open(m_frameWidth, m_frameHeight))
		emit displayStatus("Camera available");
	else
		emit displayStatus("Camera not found");

    //  up the priority in case it's helpful

    m_thread->setPriority(QThread::TimeCriticalPriority);

	m_elapsedTimer.start();

	_RTGPUInit(m_frameWidth, m_frameHeight);
}

void RTGPUTestThread::finishThread()
{
	qDebug() << "RTGPUTestThread exit";
    if (m_timer != -1)
        killTimer(m_timer);

    m_timer = -1;

	_RTGPUClose();
}

void RTGPUTestThread::timerEvent(QTimerEvent * /* event */)
{
	if (!m_testRun)
		return;
	executeTest();
}

void RTGPUTestThread::startTest(RTGPU_SETTINGS *settings)
{
	m_settings = *settings;

	if (!m_settings.continuous) {
		executeTest();
		return;
	} 
	//	continuous mode

	m_iterations = 0;
	m_firstFrame = true;
	m_gpuTotalTime = 0;
	m_cpuTotalTime = 0;
	m_testRun = true;

    m_timer = startTimer(100);
}

void RTGPUTestThread::stopTest()
{
	m_testRun = false;
    if (m_timer != -1)
        killTimer(m_timer);

    m_timer = -1;
}

//
//	This neat macro came from http://www.parashift.com/c++-faq-lite/pointers-to-members.html
//	Section 33.7

#define	CALL_TESTFUNC(object, member) ((object).*(member))

void RTGPUTestThread::executeTest()
{
	processFrame();
	m_runInProgress = true;
	CALL_TESTFUNC(*this, RTGPUTestThread::testArray[m_settings.testIndex].func)();
	m_runInProgress = false;
}


void RTGPUTestThread::processFrame()
{
	if (m_runInProgress)
		return;

	if (!m_camera.isOpen())
		return;

	if (!m_firstFrame)
			deleteImageSet();
	m_firstFrame = false;
	storeImageSet(m_camera.getFrame());
}

void RTGPUTestThread::storeImageSet(unsigned char *image)
{
	Mat temp;

	m_wasPinned = m_settings.usePinned;
	m_width = m_frameWidth;
	m_height = m_frameHeight;
	RTGPUCreateImage(m_imageInRGB, cvSize(m_width, m_height), true, m_settings.usePinned);
	memcpy(m_imageInRGB.data, image, m_width * m_height * 3);
	RTGPUCreateImage(m_imageInGray, cvSize(m_width, m_height), false, m_settings.usePinned);
	cvtColor(m_imageInRGB, m_imageInGray, CV_RGB2GRAY);

	RTGPUCreateImage(m_imageInterRGB, cvSize(m_width, m_height), true, m_settings.usePinned);
	RTGPUCreateImage(m_imageInterGray, cvSize(m_width, m_height), false, m_settings.usePinned);

	RTGPUCreateImage(m_imageDiffRGB, cvSize(m_width, m_height), true, m_settings.usePinned);
	RTGPUCreateImage(m_imageDiffGray, cvSize(m_width, m_height), false, m_settings.usePinned);

	//	Flip these images to make the array ops more interesting

	flip(m_imageInRGB, m_imageInterRGB, 0);
	flip(m_imageInGray, m_imageInterGray, 0);

	RTGPUCreateImage(m_gpuImageOutRGB, cvSize(m_width, m_height), true, m_settings.usePinned);
	RTGPUCreateImage(m_gpuImageOutGray, cvSize(m_width, m_height), false, m_settings.usePinned);
	RTGPUCreateImage(m_cpuImageOutRGB, cvSize(m_width, m_height), true, m_settings.usePinned);
	RTGPUCreateImage(m_cpuImageOutGray, cvSize(m_width, m_height), false, m_settings.usePinned);

	RTGPUCreateImage(m_imageMask, cvSize(m_width, m_height), false, m_settings.usePinned);

	for (int y = 0; y < m_height; y++) {
		for (int x = 0; x < m_width; x++) {
			m_imageMask.data[y * m_width + x] = 0;

			if ((x > m_width / 5) && (x < (m_width * 4) / 5) && (y > m_height / 5) && (y < (m_height * 4) / 5))
			m_imageMask.data[y * m_width + x] = 128;
		}
	}
}

void RTGPUTestThread::deleteImageSet()
{
	RTGPUReleaseImage(m_imageInRGB, m_wasPinned);
	RTGPUReleaseImage(m_imageInGray, m_wasPinned);

	RTGPUReleaseImage(m_imageInterRGB, m_wasPinned);
	RTGPUReleaseImage(m_imageInterGray, m_wasPinned);

	RTGPUReleaseImage(m_imageDiffRGB, m_wasPinned);
	RTGPUReleaseImage(m_imageDiffGray, m_wasPinned);

	RTGPUReleaseImage(m_gpuImageOutRGB, m_wasPinned);
	RTGPUReleaseImage(m_gpuImageOutGray, m_wasPinned);
	RTGPUReleaseImage(m_cpuImageOutRGB, m_wasPinned);
	RTGPUReleaseImage(m_cpuImageOutGray, m_wasPinned);

	RTGPUReleaseImage(m_imageMask, m_wasPinned);
}


void RTGPUTestThread::displayImage(int mode)
{
	Mat temp1;
	QImage img;
	unsigned char *data;

	switch (mode)
	{
		case DISPLAY_GPU:
			if (!m_settings.color)
				cvtColor(m_gpuImageOutGray, m_gpuImageOutRGB, CV_GRAY2RGB);

			data = (unsigned char *)malloc(m_gpuImageOutRGB.step * m_gpuImageOutRGB.rows);
			memcpy(data, m_gpuImageOutRGB.data, m_gpuImageOutRGB.step * m_gpuImageOutRGB.rows);
            img = QImage(data, m_gpuImageOutRGB.cols, m_gpuImageOutRGB.rows, 
				m_gpuImageOutRGB.step, QImage::Format_RGB888, free, data);
			emit displayImage(img, DISPLAY_GPU);
			break;

		case DISPLAY_CPU:
			if (!m_settings.color)
				cvtColor(m_cpuImageOutGray, m_cpuImageOutRGB, CV_GRAY2RGB);

			data = (unsigned char *)malloc(m_cpuImageOutRGB.step * m_cpuImageOutRGB.rows);
			memcpy(data, m_cpuImageOutRGB.data, m_cpuImageOutRGB.step * m_cpuImageOutRGB.rows);
            img = QImage(data, m_cpuImageOutRGB.cols, m_cpuImageOutRGB.rows, 
				m_cpuImageOutRGB.step, QImage::Format_RGB888, free, data);
			emit displayImage(img, DISPLAY_CPU);
			break;

		case DISPLAY_DIFF:
			if (!m_settings.color)
				cvtColor(m_imageDiffGray, m_imageDiffRGB, CV_GRAY2RGB);

			data = (unsigned char *)malloc(m_imageDiffRGB.step * m_imageDiffRGB.rows);
			memcpy(data, m_imageDiffRGB.data, m_imageDiffRGB.step * m_imageDiffRGB.rows);
            img = QImage(data, m_imageDiffRGB.cols, m_imageDiffRGB.rows, 
				m_imageDiffRGB.step, QImage::Format_RGB888, free, data);
			emit displayImage(img, DISPLAY_DIFF);
			break;
	}
}

void RTGPUTestThread::displayResult()
{
	float gpuTime, cpuTime, gpurate, cpurate;
	QString msg;

//	Compute difference image

	if (m_settings.color) {
		absdiff(m_gpuImageOutRGB, m_cpuImageOutRGB, m_imageDiffRGB);
		m_imageDiffRGB *= 5.0;
	} else {
		absdiff(m_gpuImageOutGray, m_cpuImageOutGray, m_imageDiffGray);
	}

	displayImage(DISPLAY_DIFF);
	if (m_settings.continuous) {
		if (m_runContFirst)
		{
			m_runContFirst = false;		// ignore first run
			return;
		}
		m_iterations++;
		m_gpuTotalTime += m_gpuStopTime - m_gpuStartTime;
		m_cpuTotalTime += m_cpuStopTime - m_cpuStartTime;
		gpuTime = (float)m_gpuTotalTime / (float)m_iterations;
		cpuTime = (float)m_cpuTotalTime / (float)m_iterations;
	} else {
		gpuTime = m_gpuStopTime - m_gpuStartTime;
		cpuTime = m_cpuStopTime - m_cpuStartTime;
	}
	gpurate = (float)(m_width * m_height) * 1.0e9 / gpuTime;
	cpurate = (float)(m_width * m_height) * 1.0e9 / cpuTime;

	msg = QString("%1 GPU time = %2mS (%3 MPixels/s), CPU time = %4mS (%5 MPixels/s)")
		.arg(m_settings.testName).arg(gpuTime / 1.0e6, 0, 'g', 3).arg(gpurate, 0, 'g', 3)
		.arg(cpuTime / 1.0e6, 0, 'g', 3).arg(cpurate, 0, 'g', 3);
	emit displayStatus(msg);
}




//----------------------------------------------------------
//
//	The tests

void RTGPUTestThread::testImagePut()
{

//	Do GPU version

	GSTART;
	RTGPUPutImage(0, m_settings.color ? m_imageInRGB : m_imageInGray);
	GSTOP;	
	RTGPUGetImage(0, m_settings.color ? m_gpuImageOutRGB : m_gpuImageOutGray);

	displayImage(DISPLAY_GPU);

//	Do CPU version

	CSTART;
	if (m_settings.color)
		m_cpuImageOutRGB = m_imageInRGB.clone();
	else
		m_cpuImageOutGray = m_imageInGray.clone();
	CSTOP;
	displayImage(DISPLAY_CPU);
	displayResult();
}

void RTGPUTestThread::testImageGet()
{

//	Do GPU version

	RTGPUPutImage(0, m_settings.color ? m_imageInRGB : m_imageInGray);
	GSTART;
	RTGPUGetImage(0, m_settings.color ? m_gpuImageOutRGB : m_gpuImageOutGray);
	GSTOP;	

	displayImage(DISPLAY_GPU);

//	Do CPU version

	CSTART;
	if (m_settings.color)
		m_cpuImageOutRGB = m_imageInRGB.clone();
	else
		m_cpuImageOutGray = m_imageInGray.clone();
	CSTOP;
	displayImage(DISPLAY_CPU);
	displayResult();
}


void RTGPUTestThread::testBoxFilter()
{
	int width;

	width = m_settings.params.toInt();
	if (width < 3)
		width = 3;
	
	//	make sure odd

	width |= 1;

//	Do GPU version

	RTGPUPutImage(0, m_settings.color ? m_imageInRGB : m_imageInGray);

	GSTART
	_RTGPUBoxFilter(0, 1, width / 2, true);
	GSTOP

	if (m_settings.color)
		RTGPUGetImage(1, m_gpuImageOutRGB);
	else
		RTGPUGetImage(1, m_gpuImageOutGray);

	displayImage(DISPLAY_GPU);

//	Do CPU version

	Mat mi;
	Mat mo;

	if (m_settings.color)
	{
		mi = m_imageInRGB;
		mo = m_cpuImageOutRGB;
	}
	else
	{
		mi = m_imageInGray;
		mo = m_cpuImageOutGray;
	}
	CSTART;
	boxFilter(mi, mo, mi.type(), cvSize(width, width));
	CSTOP;

	displayImage(DISPLAY_CPU);

	displayResult();
}

void RTGPUTestThread::testRGB2GRAY()
{

	m_settings.color = true;

//	Do GPU version

	RTGPUPutImage(0, m_imageInRGB);
	GSTART;
	_RTGPURGB2GRAY(0, 1);
	_RTGPUGRAY2RGB(1, 2);
	GSTOP;	
	RTGPUGetImage(2, m_gpuImageOutRGB);

	displayImage(DISPLAY_GPU);

//	Do CPU version

	CSTART;
	cvtColor(m_imageInRGB, m_imageInterGray, CV_RGB2GRAY);
	cvtColor(m_imageInterGray, m_cpuImageOutRGB, CV_GRAY2RGB);
	CSTOP;
	displayImage(DISPLAY_CPU);

	displayResult();
}


void RTGPUTestThread::testGaussianShared()
{
	testGaussian(false);
}

void RTGPUTestThread::testGaussianTexture()
{
	testGaussian(true);
}

void RTGPUTestThread::testGaussian(bool tex)
{
	int width;

	width = m_settings.params.toInt();
	if (width < 3)
		width = 3;
	
	//	make sure odd

	width |= 1;

	if ((width / 2) > MAX_GAUSSIAN_KERNEL_RADIUS)
	{
		emit displayStatus("Unsupported kernel size");
		return;
	}

//	Do GPU version

	RTGPUPutImage(0, m_settings.color ? m_imageInRGB : m_imageInGray);

	if (tex) {
		_RTGPUCreateConvTex(width / 2);
		GSTART;
		_RTGPUConvTex(0, 1, width / 2);
		GSTOP;
	} else {
		_RTGPUSetGaussian(width / 2);			// precompute the kernel
		GSTART;
		_RTGPUGaussian(0, 1, width / 2);
		GSTOP;
	}
 
	if (m_settings.color)
		RTGPUGetImage(1, m_gpuImageOutRGB);
	else
		RTGPUGetImage(1, m_gpuImageOutGray);

	displayImage(DISPLAY_GPU);

//	Do CPU version

	if (m_settings.color) {
		CSTART;
		GaussianBlur(m_imageInRGB, m_cpuImageOutRGB, Size(width, width), 0, 0);
		CSTOP;
	} else {
		CSTART;
		GaussianBlur(m_imageInGray, m_cpuImageOutGray, Size(width, width), 0, 0);
		CSTOP;
	}

	displayImage(DISPLAY_CPU);

	displayResult();
}



void RTGPUTestThread::testThreshold()
{
	int thresh;

	thresh = m_settings.params.toInt();						// get threshold level
	if (thresh == 0)
		thresh = 128;										// default;

	if ((thresh < 0) || (thresh > 255)) {
		emit displayStatus("Invalid threshold (must be between 0 and 255)");
		return;
	}

	m_settings.color = false;

//	Do GPU version

	RTGPUPutImage(0, m_imageInGray);

	GSTART;
	_RTGPUThreshold(0, 1, thresh, 255, GPU_THRESH_BINARY);
	GSTOP;

	RTGPUGetImage(1, m_gpuImageOutGray);

	displayImage(DISPLAY_GPU);

//	Do CPU version

	CSTART;
	threshold(m_imageInGray, m_cpuImageOutGray, thresh, 255, CV_THRESH_BINARY);
	CSTOP;

	displayImage(DISPLAY_CPU);

	displayResult();
}

void RTGPUTestThread::testAdaptiveThreshold()
{
	int width;

	width = m_settings.params.toInt();
	if (width < 3)
		width = 3;
	
	//	make sure odd

	width |= 1;

	m_settings.color = false;

//	Do GPU version

	RTGPUPutImage(0, m_imageInGray);

	GSTART;
	_RTGPUAdaptiveThreshold(0, 1, 128, 0, GPU_THRESH_BINARY_INV, width / 2, 10);
	GSTOP;

	RTGPUGetImage(1, m_gpuImageOutGray);

	displayImage(DISPLAY_GPU);

//	Do CPU version

	CSTART;
	adaptiveThreshold(m_imageInGray, m_cpuImageOutGray, 128, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, width, 10);
	CSTOP;

	displayImage(DISPLAY_CPU);

	displayResult();
}

void RTGPUTestThread::testSobel()
{
	Mat	opl16;

	m_settings.color = false;

//	Do GPU version

	RTGPUPutImage(0, m_imageInGray);
	GSTART;
	_RTGPUSobel(0, 1);
	GSTOP;
	RTGPUGetImage(1, m_gpuImageOutGray);

	displayImage(DISPLAY_GPU);

//	Do CPU version

	opl16 = Mat(Size(m_width, m_height), CV_16SC1);

	CSTART;
	Sobel(m_imageInGray, opl16, opl16.type(), 1, 1, 3, 8);
	convertScaleAbs(opl16, m_cpuImageOutGray);
	CSTOP;

	displayImage(DISPLAY_CPU);

	displayResult();
}


void RTGPUTestThread::testMedianBlur()
{
	int width;

	width = m_settings.params.toInt();
	if (width < 3)
		width = 3;
	
	//	make sure odd

	width |= 1;

	m_settings.color = false;

//	Do GPU version

	RTGPUPutImage(0, m_imageInGray);

	GSTART;
	_RTGPUMedianBlur(0, 1, width / 2);
	GSTOP;

	RTGPUGetImage(1, m_gpuImageOutGray);

	displayImage(DISPLAY_GPU);

//	Do CPU version

	CSTART;
	medianBlur(m_imageInGray, m_cpuImageOutGray, width);
	CSTOP;

	displayImage(DISPLAY_CPU);

	displayResult();
}

void RTGPUTestThread::testArrayOps()
{
	int type, mask;
	bool useMask;

	type = mask = 0;
	QStringList list = m_settings.params.split(" ");
	if (list.count() > 0) {
		type = list.at(0).toInt();
		if (list.count() > 1) {
			mask = list.at(1).toInt();
		}
	}
	useMask = mask != 0;

	if ((type < GPU_ARRAYOPS_MINOPT) || (type > GPU_ARRAYOPS_MAXOPT)) {
		emit displayStatus("Invalid array op code");
		return;
	}

	m_settings.color = false;

//	Do GPU version

	RTGPUPutImage(0, m_imageInGray);
	RTGPUPutImage(1, m_imageInterGray);
	RTGPUPutImage(2, m_imageInGray);
	RTGPUPutImage(3, m_imageMask);

	GSTART;
	_RTGPUArrayOps(0, 1, 2, useMask ? 3 : -1, type);
	GSTOP;

	RTGPUGetImage(2, m_gpuImageOutGray);

	displayImage(DISPLAY_GPU);

//	Do CPU version

	m_imageInGray.copyTo(m_cpuImageOutGray);

	CSTART;
	switch(type)
	{
		case GPU_ARRAYOPS_ABSDIFF:
			absdiff(m_imageInGray, m_imageInterGray, m_cpuImageOutGray);
			break;

		case GPU_ARRAYOPS_CMPEQ:
			compare(m_imageInGray, m_imageInterGray, m_cpuImageOutGray, CMP_EQ);
			break;

		case GPU_ARRAYOPS_CMPGE:
			compare(m_imageInGray, m_imageInterGray, m_cpuImageOutGray, CMP_GE);
			break;

		case GPU_ARRAYOPS_CMPGT:
			compare(m_imageInGray, m_imageInterGray, m_cpuImageOutGray, CMP_GT);
			break;

		case GPU_ARRAYOPS_CMPLT:
			compare(m_imageInGray, m_imageInterGray, m_cpuImageOutGray, CMP_LT);
			break;

		case GPU_ARRAYOPS_CMPLE:
			compare(m_imageInGray, m_imageInterGray, m_cpuImageOutGray, CMP_LE);
			break;

		case GPU_ARRAYOPS_CMPNE:
			compare(m_imageInGray, m_imageInterGray, m_cpuImageOutGray, CMP_NE);
			break;

		case GPU_ARRAYOPS_OR:
			bitwise_or(m_imageInGray, m_imageInterGray, m_cpuImageOutGray, useMask ? m_imageMask : noArray());
			break;

	}
	CSTOP;
	displayImage(DISPLAY_CPU);

	displayResult();
}

void RTGPUTestThread::testScalarOps()
{
	int type, val0, val1, val2;
	bool useMask;
	unsigned char lut[256];
	int i;

	type = val0 = val1 = val2 = 0;

	QStringList list = m_settings.params.split(" ");
	if (list.count() > 0) {
		type = list.at(0).toInt();
		if (list.count() > 1) {
			val0 = list.at(1).toInt();
			if (list.count() > 2) {
				val1 = list.at(2).toInt();
				if (list.count() > 3) {
					val2 = list.at(3).toInt();
				}
			}
		}
	}
	if ((type < GPU_SCALAROPS_MINOPT) || (type > GPU_SCALAROPS_MAXOPT)) {
		emit displayStatus("Invalid scalar op code");
		return;
	}

	if ((val0 == 0.0f) && (val1 == 0.0f) && (val2 == 0.0f))
		useMask = false;
	else
		useMask = true;

	for (i = 0; i < 256; i++)
		lut[i] = 255-i;

	_RTGPUCreateLUT(lut);
    Mat mlut= Mat(1,256, CV_8UC1, lut);

//	Do GPU version

	RTGPUPutImage(0, m_settings.color ? m_imageInRGB : m_imageInGray);
	RTGPUPutImage(1, m_settings.color ? m_imageInRGB : m_imageInGray);
	RTGPUPutImage(2, m_imageMask);

	GSTART;
	_RTGPUScalarOps(0, RTGPUScalar((float)val0, (float)val1, (float)val2, 0), 1, useMask ? 2 : -1, type);
	GSTOP;

	RTGPUGetImage(1, m_settings.color ? m_gpuImageOutRGB : m_gpuImageOutGray);

	displayImage(DISPLAY_GPU);

//	Do CPU version

	m_imageInRGB.copyTo(m_cpuImageOutRGB);
	m_imageInGray.copyTo(m_cpuImageOutGray);
	CSTART;
	switch (type)
	{
		case GPU_SCALAROPS_SUB:
			subtract(m_settings.color ? m_imageInRGB : m_imageInGray, Scalar(val0, val1, val2, 0),
				m_settings.color ? m_cpuImageOutRGB : m_cpuImageOutGray, useMask ? m_imageMask : noArray());
			break;

		case GPU_SCALAROPS_ADD:
			add(m_settings.color ? m_imageInRGB : m_imageInGray, Scalar(val0, val1, val2, 0), 
				m_settings.color ? m_cpuImageOutRGB : m_cpuImageOutGray, useMask ? m_imageMask : noArray());
			break;

		case GPU_SCALAROPS_NOT:
			bitwise_not(m_settings.color ? m_imageInRGB : m_imageInGray, m_settings.color ? m_cpuImageOutRGB : m_cpuImageOutGray, 
					useMask ? m_imageMask : noArray());
			break;

		case GPU_SCALAROPS_LUT:
			LUT(m_settings.color ? m_imageInRGB : m_imageInGray, mlut, m_settings.color ? m_cpuImageOutRGB : m_cpuImageOutGray);
			break;
	}
	CSTOP;

	displayImage(DISPLAY_CPU);

	displayResult();
}

void RTGPUTestThread::testMorphOps()
{
	int type;

	type = 0;
	type = m_settings.params.toInt();

	if ((type < GPU_MORPHOPS_MINOPT) || (type > GPU_MORPHOPS_MAXOPT)) {
		emit displayStatus("Invalid morphology op code");
		return;
	}

//	Do GPU version

	RTGPUPutImage(0, m_settings.color ? m_imageInRGB : m_imageInGray);

	GSTART;
	_RTGPUMorphOps3x3(0, 1, type);
	GSTOP;

	RTGPUGetImage(1, m_settings.color ? m_gpuImageOutRGB : m_gpuImageOutGray);

	displayImage(DISPLAY_GPU);

//	Do CPU version

	m_imageInRGB.copyTo(m_cpuImageOutRGB);
	m_imageInGray.copyTo(m_cpuImageOutGray);
	CSTART;
	switch (type)
	{
		case GPU_MORPHOPS_DILATE:
			dilate(m_settings.color ? m_imageInRGB : m_imageInGray, m_settings.color ? m_cpuImageOutRGB : m_cpuImageOutGray, Mat());
			break;

		case GPU_MORPHOPS_ERODE:
			erode(m_settings.color ? m_imageInRGB : m_imageInGray, m_settings.color ? m_cpuImageOutRGB : m_cpuImageOutGray, Mat());
			break;

	}
	CSTOP;

	displayImage(DISPLAY_CPU);

	displayResult();
}

void RTGPUTestThread::testJPEGEncoder()
{
	int jpegBytes;
	QImage img;
	QImage imgRGB;
	QByteArray frame;
	QByteArray jpeg;

	m_settings.color = true;

//	Do GPU version

	RTGPUPutImage(0, m_imageInRGB);
	_RTGPUJPEGInitEncoder();
	GSTART
	_RTGPUJPEGCompress(0, (unsigned char *)m_imageInterRGB.data, jpegBytes);
	GSTOP;

	frame = QByteArray((const char *)m_imageInterRGB.data, jpegBytes);
	img.loadFromData(frame, "JPEG");
	imgRGB = img.convertToFormat(QImage::Format_RGB888);
	memcpy(m_gpuImageOutRGB.data, imgRGB.constBits(), imgRGB.byteCount());

	displayImage(DISPLAY_GPU);

//	Do CPU version

	CSTART;
	img = QImage(m_imageInRGB.data, m_width, m_height, QImage::Format_RGB888);
    QBuffer buffer(&jpeg);
    buffer.open(QIODevice::WriteOnly);
    img.save(&buffer, "JPEG");	
	CSTOP;

	img.loadFromData(jpeg, "JPEG");
	imgRGB = img.convertToFormat(QImage::Format_RGB888);
	memcpy(m_cpuImageOutRGB.data, imgRGB.constBits(), imgRGB.byteCount());

	displayImage(DISPLAY_CPU);

	displayResult();

}

void RTGPUTestThread::testJPEGDecoder()
{
    int w, h, size;
	QImage img;
	QImage imgRGB;
	QByteArray frame;
	QByteArray jpeg;

	m_settings.color = true;

	//	prepare compressed image

	img = QImage(m_imageInRGB.data, m_width, m_height, QImage::Format_RGB888);
    QBuffer buffer(&jpeg);
    buffer.open(QIODevice::WriteOnly);
    img.save(&buffer, "JPEG");	

//	Do GPU version

	if (!_RTGPUReadJPEGHeader((unsigned char *)jpeg.constData(), jpeg.count(), w, h, size)) {
		qDebug() << "Failed to read jpeg header";
		return;
	}

	if ((w != m_width) || (h != m_height)) {
		qDebug() << "jpeg image size mismatch";
		return;
	}

	GSTART
	_RTGPUJPEGDecompress((unsigned char *)jpeg.constData() + size, 0);
	GSTOP;

	RTGPUGetImage(0, m_gpuImageOutRGB);

	displayImage(DISPLAY_GPU);

//	Do CPU version

	CSTART;
	img.loadFromData(jpeg, "JPEG");
	imgRGB = img.convertToFormat(QImage::Format_RGB888);
	memcpy(m_cpuImageOutRGB.data, imgRGB.constBits(), imgRGB.byteCount());
	CSTOP;

	displayImage(DISPLAY_CPU);

	displayResult();
}


//----------------------------------------------------------
//
//  The following is some Qt threading stuff

void RTGPUTestThread::resumeThread()
{
    m_thread = new QThread();
    moveToThread(m_thread);
    connect(m_thread, SIGNAL(started()), this, SLOT(internalRunLoop()));
	connect(this, SIGNAL(internalEndThread()), this, SLOT(cleanup()));
	connect(this, SIGNAL(internalKillThread()), m_thread, SLOT(quit()), Qt::DirectConnection);
    connect(m_thread, SIGNAL(finished()), m_thread, SLOT(deleteLater()));
    connect(m_thread, SIGNAL(finished()), this, SLOT(deleteLater()));
    m_thread->start();
}

