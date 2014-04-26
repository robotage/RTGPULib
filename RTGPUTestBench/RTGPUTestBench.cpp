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

#include "RTGPUTestBench.h"
#include "ui_RTGPUTestBench.h"
#include "RTGPU.h"

#include <qboxlayout.h>
#include <qsize.h>

RTGPUTestBench::RTGPUTestBench(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	m_thread = new RTGPUTestThread();
	m_thread->resumeThread();

	layoutWindow();

	connect(this, SIGNAL(startTest(RTGPU_SETTINGS *)), m_thread, SLOT(startTest(RTGPU_SETTINGS *)));
	connect(m_thread, SIGNAL(displayImage(QImage, int)), this, SLOT(displayImage(QImage, int)));
	connect(m_thread, SIGNAL(displayStatus(QString)), this, SLOT(displayStatus(QString)));

	connect(m_runOnce, SIGNAL(clicked()), this, SLOT(onRunOnce()));
	connect(m_runCont, SIGNAL(clicked()), this, SLOT(onRunCont()));
	connect(m_stopTest, SIGNAL(clicked()), this, SLOT(onStopTest()));
	connect(this, SIGNAL(stopTest()), m_thread, SLOT(stopTest()));
	
	connect(m_testSelect, SIGNAL(currentIndexChanged(int)), this, SLOT(onTestChanged(int)));
}

RTGPUTestBench::~RTGPUTestBench()
{

}

void RTGPUTestBench::closeEvent(QCloseEvent *)
{
    m_thread->exitThread();
	m_thread = NULL;
}

void RTGPUTestBench::onRunOnce()
{
	m_settings.testName = m_testSelect->currentText();
	m_settings.testIndex = m_testSelect->currentIndex();
	m_settings.params = m_params->text();
	m_settings.color = m_RGBColor->checkState() == Qt::Checked;
	m_settings.usePinned = m_usePinned->checkState() == Qt::Checked;
	m_settings.continuous = false;
	emit startTest(&m_settings);
}

void RTGPUTestBench::onRunCont()
{
	m_settings.testName = m_testSelect->currentText();
	m_settings.testIndex = m_testSelect->currentIndex();
	m_settings.params = m_params->text();
	m_settings.color = m_RGBColor->checkState() == Qt::Checked;
	m_settings.usePinned = m_usePinned->checkState() == Qt::Checked;
	m_settings.continuous = true;
	emit startTest(&m_settings);
	m_runCont->setDisabled(true);
	m_stopTest->setDisabled(false);
}

void RTGPUTestBench::onStopTest()
{
	emit stopTest();
	m_runCont->setDisabled(false);
	m_stopTest->setDisabled(true);
}

void RTGPUTestBench::onTestChanged(int test)
{
	if (!m_runCont->isEnabled())
		onStopTest();

	m_paramsPrompt->setText(RTGPUTestThread::testArray[test].prompt);
}

void RTGPUTestBench::displayImage(QImage image, int destWindow)
{
	QImage scaledImg;

	switch (destWindow) {
	case DISPLAY_CPU:
		scaledImg = image.scaled(m_cpuImage->size(), Qt::KeepAspectRatio);
		m_cpuImage->setPixmap(QPixmap::fromImage(scaledImg));
		break;

	case DISPLAY_GPU:
		scaledImg = image.scaled(m_gpuImage->size(), Qt::KeepAspectRatio);
		m_gpuImage->setPixmap(QPixmap::fromImage(scaledImg));
		break;

	case DISPLAY_DIFF:
		scaledImg = image.scaled(m_diffImage->size(), Qt::KeepAspectRatio);
		m_diffImage->setPixmap(QPixmap::fromImage(scaledImg));
		break;
	}
}

void RTGPUTestBench::displayStatus(QString status)
{
	m_runStatus->setText(status);
}

void RTGPUTestBench::layoutWindow()
{
	QLabel *label;

	// create the main layouts

	setCentralWidget(ui.centralWidget);
	QVBoxLayout *mainLayout = new QVBoxLayout();
	QHBoxLayout *testLayout = new QHBoxLayout();
	QHBoxLayout *runLayout = new QHBoxLayout();
	QHBoxLayout *imageLayout = new QHBoxLayout();
	QVBoxLayout *cpuImageLayout = new QVBoxLayout();
	QVBoxLayout *gpuImageLayout = new QVBoxLayout();
	QVBoxLayout *diffImageLayout = new QVBoxLayout();

	ui.centralWidget->setLayout(mainLayout);
	mainLayout->addLayout(testLayout);
	mainLayout->addSpacing(20);
	mainLayout->addLayout(runLayout);
	mainLayout->addSpacing(20);
	mainLayout->addLayout(imageLayout);
	mainLayout->addStretch(1);

	imageLayout->addLayout(cpuImageLayout);
	imageLayout->addSpacing(20);
	imageLayout->addLayout(gpuImageLayout);
	imageLayout->addSpacing(20);
	imageLayout->addLayout(diffImageLayout);

	//	set up test layout

	testLayout->addWidget(new QLabel("Test: "));
	m_testSelect = new QComboBox();
	testLayout->addWidget(m_testSelect);
	testLayout->addSpacing(40);
	testLayout->addWidget(new QLabel("Params: "));
	m_params = new QLineEdit();
	m_params->setMinimumWidth(200);
	testLayout->addWidget(m_params);
	testLayout->addWidget(new QLabel(" Hint: "));
	m_paramsPrompt = new QLabel(); 
	testLayout->addWidget(m_paramsPrompt);
	m_paramsPrompt->setMinimumWidth(200);
	testLayout->addStretch(1);

	for (int i = 0; i < RTGPUTestThread::testFunctionCount; i++)
		m_testSelect->addItem(RTGPUTestThread::testArray[i].name);

	m_paramsPrompt->setText(RTGPUTestThread::testArray[0].prompt);

	m_RGBColor = new QCheckBox("RGB Color");
	testLayout->addWidget(m_RGBColor);
	testLayout->addSpacing(20);
	m_usePinned = new QCheckBox("Use pinned memory");
	testLayout->addWidget(m_usePinned);
	testLayout->addStretch(1);

	//	set up the run layout

	m_runOnce = new QPushButton("Run once");
	runLayout->addWidget(m_runOnce);
	runLayout->addSpacing(20);
	m_runCont = new QPushButton("Run continuously");
	runLayout->addWidget(m_runCont);
	runLayout->addSpacing(20);
	m_stopTest = new QPushButton("Stop test");
	runLayout->addWidget(m_stopTest);
	m_stopTest->setDisabled(true);
	runLayout->addSpacing(20);
	m_runStatus = new QLabel("...");
	m_runStatus->setFrameStyle(QFrame::Panel | QFrame::Sunken);
	runLayout->addWidget(m_runStatus, 1);

	//	set up the image layout

	label = new QLabel("CPU/OpenCV Image");
	label->setAlignment(Qt::AlignCenter);
	cpuImageLayout->addWidget(label);
	m_cpuImage = new QLabel();
	m_cpuImage->setMinimumSize(QSize(320, 240));
	cpuImageLayout->addWidget(m_cpuImage);

	label = new QLabel("RTGPULib Image");
	label->setAlignment(Qt::AlignCenter);
	gpuImageLayout->addWidget(label);
	m_gpuImage = new QLabel();
	m_gpuImage->setMinimumSize(QSize(320, 240));
	gpuImageLayout->addWidget(m_gpuImage);

	label = new QLabel("Difference Image");
	label->setAlignment(Qt::AlignCenter);
	diffImageLayout->addWidget(label);
	m_diffImage = new QLabel();
	m_diffImage->setMinimumSize(QSize(320, 240));
	diffImageLayout->addWidget(m_diffImage);

}