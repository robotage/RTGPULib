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

#ifndef RTGPUTESTBENCH_H
#define RTGPUTESTBENCH_H

#include <QtWidgets/QMainWindow>
#include <QLabel>
#include <qcombobox.h>
#include <qpushbutton.h>
#include <qcheckbox.h>	
#include <qimage.h>
#include <qlineedit.h>

#include "ui_RTGPUTestBench.h"
#include "RTGPUTestThread.h"

class RTGPUTestBench : public QMainWindow
{
	Q_OBJECT

public:
	RTGPUTestBench(QWidget *parent = 0);
	~RTGPUTestBench();

public slots:
	void onRunOnce();
	void onRunCont();
	void onStopTest();
	void onTestChanged(int);
	void displayStatus(QString status);					
	void displayImage(QImage image, int destWindow);

signals:
	void startTest(RTGPU_SETTINGS *);
	void stopTest();

protected:
	void closeEvent(QCloseEvent *);

private:
	Ui::RTGPUTestBenchClass ui;

	void layoutWindow();

	QComboBox *m_testSelect;
	QLabel *m_paramsPrompt;
	QLineEdit *m_params;
	QPushButton *m_stopTest;
	QPushButton *m_runOnce;
	QPushButton *m_runCont;
	QCheckBox *m_RGBColor;
	QCheckBox *m_usePinned;
	QLabel *m_runStatus;

	QLabel *m_cpuImage;
	QLabel *m_gpuImage;
	QLabel *m_diffImage;

	RTGPUTestThread *m_thread;
	RTGPU_SETTINGS m_settings;

};

#endif // RTGPUTESTBENCH_H
