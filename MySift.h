#ifdef _CH_  
package <opencv>  
#endif  

#ifndef _EiC  
#include <stdio.h>  


#include "stdlib.h"  
#include "string.h"   
#include "malloc.h"   
#include "math.h"   
#include <assert.h>  
#include <ctype.h>  
#include <time.h>  
#include <cv.h>
#include <cxcore.h>  
#include <highgui.h>
#include <vector>  
#include <iostream>

using namespace std;

#endif  

#ifdef _EiC  
#define WIN32  
#endif

#define NUMSIZE 2  
#define GAUSSKERN 3.5  
#define PI 3.14159265358979323846  

//Sigma of base image
#define INITSIGMA 0.5  
//Sigma of each octave 
#define SIGMA sqrt(3)//1.6//  

//Number of scales per octave. 
#define SCALESPEROCTAVE 2  
#define MAXOCTAVES 4  

#define CONTRAST_THRESHOLD   0.02  
#define CURVATURE_THRESHOLD  10.0  
#define DOUBLE_BASE_IMAGE_SIZE 1  
#define peakRelThresh 0.8  
#define LEN 128 
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#define max(a,b)            (((a) > (b)) ? (a) : (b))
    
#define GridSpacing 4

//Data structure for a float image.  
typedef struct ImageSt { 
	float levelsigma;
	int levelsigmalength;
	float absolute_sigma;
	CvMat *Level;          
} ImageLevels;

typedef struct ImageSt1 {   
	int row, col;          //Dimensions of image.   
	float subsample;
	ImageLevels *Octave;
} ImageOctaves;

typedef struct KeypointSt {
	float row, col;
	float sx, sy;       
	int octave, level;  
	float scale, ori, mag;    
	float *descrip;           
	struct KeypointSt *next;  /* Pointer to next keypoint in list. */
} *Keypoint;

class MySift {
public:
	MySift();
	~MySift();

	MySift(char* _filename, int _isColor);

	CvMat * halfSizeImage(CvMat * im);       
	CvMat * doubleSizeImage(CvMat * im);    
	CvMat * doubleSizeImage2(CvMat * im);   
	float getPixelBI(CvMat * im, float col, float row);  
	void normalizeVec(float* vec, int dim);    
	CvMat* GaussianKernel2D(float sigma);  
	void normalizeMat(CvMat* mat);  
	float* GaussianKernel1D(float sigma, int dim);  

	float ConvolveLocWidth(float* kernel, int dim, CvMat * src, int x, int y);
	void Convolve1DWidth(float* kern, int dim, CvMat * src, CvMat * dst);  
	float ConvolveLocHeight(float* kernel, int dim, CvMat * src, int x, int y);
	void Convolve1DHeight(float* kern, int dim, CvMat * src, CvMat * dst);
	int BlurImage(CvMat * src, CvMat * dst, float sigma);


	CvMat *ScaleInitImage(CvMat * im);                   

	ImageOctaves* BuildGaussianOctaves(CvMat * image);    
  
	int DetectKeypoint(int numoctaves, ImageOctaves *GaussianPyr);
	void DisplayKeypointLocation(IplImage* image, ImageOctaves *GaussianPyr);

	void ComputeGrad_DirecandMag(int numoctaves, ImageOctaves *GaussianPyr);

	int FindClosestRotationBin(int binCount, float angle);    
	void AverageWeakBins(double* bins, int binCount);         
	bool InterpolateOrientation(double left, double middle, double right, double *degreeCorrection, double *peakValue);  
	void AssignTheMainOrientation(int numoctaves, ImageOctaves *GaussianPyr, ImageOctaves *mag_pyr, ImageOctaves *grad_pyr);  
	void DisplayOrientation(IplImage* image, ImageOctaves *GaussianPyr);
  
	void ExtractFeatureDescriptors(int numoctaves, ImageOctaves *GaussianPyr);
	CvMat* MosaicHorizen(CvMat* im1, CvMat* im2);
	CvMat* MosaicVertical(CvMat* im1, CvMat* im2);

	void SiftMainProcess();
	int getKeyPointsCount();            
	Keypoint getFirstKeyDescriptors();  

	void saveImgWithKeypoint(char* filename);

private:
	char* filename;
	int isColor;

	int numoctaves;

	ImageOctaves *DOGoctaves;
	ImageOctaves *mag_thresh;
	ImageOctaves *mag_pyr;
	ImageOctaves *grad_pyr;

	int keypoint_count = 0;

	Keypoint keypoints = NULL;        
	Keypoint keyDescriptors = NULL; 


	IplImage* src = NULL;
	IplImage* image_kp = NULL;
	IplImage* image_featDir = NULL;
	IplImage* grey_im1 = NULL;

	IplImage* mosaic1 = NULL;
	IplImage* mosaic2 = NULL;

	CvMat* mosaicHorizen1 = NULL;
	CvMat* mosaicHorizen2 = NULL;
	CvMat* mosaicVertical1 = NULL;

	CvMat* image1Mat = NULL;
	CvMat* tempMat = NULL;

	ImageOctaves *Gaussianpyr;
};