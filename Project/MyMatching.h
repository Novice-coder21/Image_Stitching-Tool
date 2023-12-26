#ifndef MYMATCHING_H
#define MYMATCHING_H

#include "MySift.h"
#include "CImg.h"
#include <vector>
using namespace cimg_library;

#define FeatureDescGap 1.0
#define InliersGap 500.0

struct Point {
	int col;    //x
	int row;    //y
	Point() : col(-1), row(-1) {}
	Point(int _col, int _row) : col(_col), row(_row) {}
};

struct MatchedPair {
	Point keyPointA;
	Point keyPointB;
	float minDis;
	MatchedPair(Point _pa, Point _pb, float _minDis) : keyPointA(_pa), keyPointB(_pb), minDis(_minDis) {}
};

class MyMatching
{
public:
	MyMatching();
	~MyMatching();
	MyMatching(int _kp_count_A, Keypoint _firstKeyDesc_A, int _kp_count_B, Keypoint _firstKeyDesc_B);

	void featureMatchMainProcess();

	void drawOriKeypointOnImg(char* _filenameA, char* _filenameB, char* _saveAddrA, char* _saveAddrB);
	
	void mixImageAndDrawPairLine(char* mixImgAddr, char* mixImgWithLineAddr);

	void myRANSACtoFindKpTransAndDrawOut(char* _filename);
	
	void drawRealKeypointOnImg(char* _filename, int maxIndex);

	Point getMatchVec();

private:
	int keypoint_count_A, keypoint_count_B;
	Keypoint firstKeyDescriptor_A, firstKeyDescriptor_B;

	vector <MatchedPair> matchedPairSet;
	Point matchVec;

	CImg<int> srcImgA, srcImgB;
	CImg<int> srcImgWithKpA, srcImgWithKpB;
	CImg<int> mixImg;
	CImg<int> fixedMatchedImg;
};

#endif

