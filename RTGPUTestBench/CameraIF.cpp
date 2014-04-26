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
//  along with R

#include "CameraIF.h"

static VideoCapture g_capture;

CameraIF::CameraIF()
{
}

CameraIF::~CameraIF()
{
}

bool CameraIF::open(int width, int height)
{
	g_capture.open(0);
	if (!g_capture.isOpened()) {
		return false;
	}
	g_capture.set(CV_CAP_PROP_FRAME_WIDTH, width);
	g_capture.set(CV_CAP_PROP_FRAME_HEIGHT, height);
	return true;
}

bool CameraIF::isOpen()
{ 
	return g_capture.isOpened(); 
}

unsigned char *CameraIF::getFrame()
{

	if (!g_capture.isOpened())
		return NULL;
	g_capture >> m_frame;
	cvtColor(m_frame, m_frame, CV_BGR2RGB);
	return m_frame.data;
}