#ifndef MYBLENDING_H
#define MYBLENDING_H

#include "CImg.h"
#include <iostream>
using namespace cimg_library;
using namespace std;

struct TransVector {
	int dx;
	int dy;
	TransVector() : dx(-1), dy(-1) {}
	TransVector(int _dx, int _dy) : dx(_dx), dy(_dy) {}
};

class MyBlending
{
public:
	MyBlending();
	~MyBlending();
	MyBlending(int sx, int sy);

	void blendingMainProcess(char* _filenameA, char* _filenameB);
	void saveBlendedImg(char* blendedImgAddr);

private:
	TransVector matchVec; 
	CImg<int> srcImgA, srcImgB;
	CImg<int> blendedImg;
};


#endif