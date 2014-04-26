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

#ifndef _RTGPUTESTTHREAD_H
#define	_RTGPUTESTTHREAD_H

#include <QThread>
#include <qstringlist.h>
#include <qimage.h>
#include <qelapsedtimer.h>

#include "opencv2/opencv.hpp"
using namespace cv;

#include "RTGPU.h"
#include "CameraIF.h"

class RTGPUTestThread;

//	The TESTFUNCTION structure

#define	TESTFUNCTION_NAME_MAX	256			// maximum length of test name

typedef void (RTGPUTestThread::*TESTFUNCPTR)();

typedef struct
{
	char *name;												// name of test for the dialog box
	TESTFUNCPTR	func;										// the actual function
	char *prompt;											// the params prompt
} TESTFUNCTION;


typedef struct
{
	QString testName;										// name of the test
	int testIndex;											// index in test array
	QString params;											// Test parameters
	bool color;												// true if color, false if monochrome
	bool usePinned;											// use pinned memory
	bool continuous;										// if run continuously
} RTGPU_SETTINGS;

class RTGPUTestThread : public QObject
{
    Q_OBJECT

public:
    RTGPUTestThread();
    virtual ~RTGPUTestThread();

    //  resumeThread() is called when init is complete

    void resumeThread();

    //  exitThread is called to terminate and delete the thread

    void exitThread() { emit internalEndThread(); }


	static TESTFUNCTION testArray[];						// array of test functions
	static int testFunctionCount;							// number of entries in the array

	//	getThread gets a pointer to the thread itself

	QThread *getThread() { return m_thread; }

public slots:

	void startTest(RTGPU_SETTINGS *);
	void stopTest();

	//	Qt threading stuff

	void internalRunLoop() { initThread(); emit running();}
    void cleanup() {finishThread(); emit internalKillThread(); }

signals:
	void displayImage(QImage image, int destWindow);		// emitted to display an image
	void displayStatus(QString status);						// emitted to display results

    void running();											// emitted when everything set up and thread active
    void internalEndThread();								// this to end thread
    void internalKillThread();								// tells the QThread to quit

protected:
    void initThread();
    void finishThread();
    void timerEvent(QTimerEvent *event);

private:
	void executeTest();
	void processFrame();
	void storeImageSet(unsigned char *image);
	void deleteImageSet();
	void displayImage(int mode);
	void displayResult();

    int m_timer;
 
    QThread *m_thread;

	QStringList m_testNames;
	RTGPU_SETTINGS m_settings;

//	The test functions

	void testImagePut();
	void testImageGet();
	void testRGB2GRAY();
	void testGaussianShared();
	void testGaussianTexture();
	void testThreshold();
	void testAdaptiveThreshold();
	void testSobel();
	void testBoxFilter();
	void testMedianBlur();
	void testArrayOps();
	void testScalarOps();
	void testMorphOps();
	void testJPEGEncoder();
	void testJPEGDecoder();

	void testGaussian(bool tex);

	Mat m_imageInRGB;
	Mat m_imageInGray;
	Mat m_imageMask;
	Mat m_imageInterRGB;
	Mat m_imageInterGray;
	Mat m_gpuImageOutRGB;
	Mat	m_gpuImageOutGray;
	Mat m_cpuImageOutRGB;
	Mat m_cpuImageOutGray;
	Mat m_imageDiffRGB;
	Mat m_imageDiffGray;

	QElapsedTimer m_elapsedTimer;
	qint64 m_gpuStartTime;
	qint64 m_gpuStopTime;
	qint64 m_cpuStartTime;
	qint64 m_cpuStopTime;

	bool m_testRun;
	bool m_runInProgress;
	bool m_runContFirst;

	int m_iterations;
	qint64 m_gpuTotalTime;
	qint64 m_cpuTotalTime;
	//	camera frame width

	int m_frameWidth;
	int m_frameHeight;

	//	processed image width (may be different if odd size)
	int m_width;
	int m_height;

	CameraIF m_camera;

	bool m_firstFrame;
	bool m_wasPinned;
};


#endif // _RTGPUTESTTHREAD_H
