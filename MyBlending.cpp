#include "MyBlending.h"

MyBlending::MyBlending() {
}

MyBlending::~MyBlending() {
}

MyBlending::MyBlending(int sx, int sy) {
	matchVec.dx = sx;
	matchVec.dy = sy;
}

void MyBlending::blendingMainProcess(char* _filenameA, char* _filenameB) {
	srcImgA.load_bmp(_filenameA);
	srcImgB.load_bmp(_filenameB);

	blendedImg = CImg<int>(srcImgA._width + srcImgB._width - matchVec.dx, 
		srcImgA._height + abs(matchVec.dy), 1, 3, 0);

	cimg_forXY(blendedImg, x, y) {
		if (matchVec.dy <= 0) { 
			if (x < srcImgA._width && y < srcImgA._height) {
				if (x >= (srcImgA._width - matchVec.dx) && y >= (0 - matchVec.dy)) { 
					blendedImg(x, y, 0, 0) = (float)srcImgA(x, y, 0, 0)
						* (float)(srcImgA._width - x) / (float)abs(matchVec.dx)
						+ (float)srcImgB(x - (srcImgA._width - matchVec.dx), y - (0 - matchVec.dy), 0, 0)
						* (float)(x - (srcImgA._width - matchVec.dx)) / (float)abs(matchVec.dx);
					blendedImg(x, y, 0, 1) = (float)srcImgA(x, y, 0, 1)
						* (float)(srcImgA._width - x) / (float)abs(matchVec.dx)
						+ (float)srcImgB(x - (srcImgA._width - matchVec.dx), y - (0 - matchVec.dy), 0, 1)
						* (float)(x - (srcImgA._width - matchVec.dx)) / (float)abs(matchVec.dx);
					blendedImg(x, y, 0, 2) = (float)srcImgA(x, y, 0, 2)
						* (float)(srcImgA._width - x) / (float)abs(matchVec.dx)
						+ (float)srcImgB(x - (srcImgA._width - matchVec.dx), y - (0 - matchVec.dy), 0, 2)
						* (float)(x - (srcImgA._width - matchVec.dx)) / (float)abs(matchVec.dx);
				}
				else {   
					blendedImg(x, y, 0, 0) = srcImgA(x, y, 0, 0);
					blendedImg(x, y, 0, 1) = srcImgA(x, y, 0, 1);
					blendedImg(x, y, 0, 2) = srcImgA(x, y, 0, 2);
				}
			}
			else if (x >= (srcImgA._width - matchVec.dx) 
				&& y >= (0 - matchVec.dy) && y < (0 - matchVec.dy) + srcImgB._height) { 
				blendedImg(x, y, 0, 0) = srcImgB(x - (srcImgA._width - matchVec.dx), y - (0 - matchVec.dy), 0, 0);
				blendedImg(x, y, 0, 1) = srcImgB(x - (srcImgA._width - matchVec.dx), y - (0 - matchVec.dy), 0, 1);
				blendedImg(x, y, 0, 2) = srcImgB(x - (srcImgA._width - matchVec.dx), y - (0 - matchVec.dy), 0, 2);
			}
			else { 
				blendedImg(x, y, 0, 0) = 0;
				blendedImg(x, y, 0, 1) = 0;
				blendedImg(x, y, 0, 2) = 0;
			}
		}
		else { 
			if (x < srcImgA._width && y >= matchVec.dy) {
				if (x >= (srcImgA._width - matchVec.dx) && y < srcImgB._height) {   
					blendedImg(x, y, 0, 0) = (float)srcImgA(x, y - matchVec.dy, 0, 0)
						* (float)(srcImgA._width - x) / (float)abs(matchVec.dx)
						+ (float)srcImgB(x - (srcImgA._width - matchVec.dx), y, 0, 0)
						* (float)(x - (srcImgA._width - matchVec.dx)) / (float)abs(matchVec.dx);
					blendedImg(x, y, 0, 1) = (float)srcImgA(x, y - matchVec.dy, 0, 1)
						* (float)(srcImgA._width - x) / (float)abs(matchVec.dx)
						+ (float)srcImgB(x - (srcImgA._width - matchVec.dx), y, 0, 1)
						* (float)(x - (srcImgA._width - matchVec.dx)) / (float)abs(matchVec.dx);
					blendedImg(x, y, 0, 2) = (float)srcImgA(x, y - matchVec.dy, 0, 2)
						* (float)(srcImgA._width - x) / (float)abs(matchVec.dx)
						+ (float)srcImgB(x - (srcImgA._width - matchVec.dx), y, 0, 2)
						* (float)(x - (srcImgA._width - matchVec.dx)) / (float)abs(matchVec.dx);
				}
				else {    
					blendedImg(x, y, 0, 0) = srcImgA(x, y - matchVec.dy, 0, 0);
					blendedImg(x, y, 0, 1) = srcImgA(x, y - matchVec.dy, 0, 1);
					blendedImg(x, y, 0, 2) = srcImgA(x, y - matchVec.dy, 0, 2);
				}
			}
			else if (x >= (srcImgA._width - matchVec.dx) && y < srcImgB._height) {  
				blendedImg(x, y, 0, 0) = srcImgB(x - (srcImgA._width - matchVec.dx), y, 0, 0);
				blendedImg(x, y, 0, 1) = srcImgB(x - (srcImgA._width - matchVec.dx), y, 0, 1);
				blendedImg(x, y, 0, 2) = srcImgB(x - (srcImgA._width - matchVec.dx), y, 0, 2);
			}
			else {  
				blendedImg(x, y, 0, 0) = 0;
				blendedImg(x, y, 0, 1) = 0;
				blendedImg(x, y, 0, 2) = 0;
			}
		}
	}
	blendedImg.display("blendedImg");
}


void MyBlending::saveBlendedImg(char* blendedImgAddr) {
	blendedImg.save(blendedImgAddr);
}