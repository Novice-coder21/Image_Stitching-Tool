# Image_Stitching-Tool
A unique software tool that allows you to create panoramas by attaching images with overlapping areas.

## Software required
* An IDE such as VS code or Visual Studio
* A C++ compiler (MS VC is optimum)
* [CMake](https://cmake.org/download/)
* [OpenCV library version 3.2.0](https://opencv.org/releases/page/6/)
* [CImg library](http://www.cimg.eu/)

## Set up
* Download the required libraries and link them to your compilers and IDEs.
* Using CMake make a CMakelists.txt file that configures the local build files.
* Create and exe file and place accordingly. Use the exe to run the program.
* The program will then create a folder named build which then runs the code.
* Update the paths according to your image locations.

## Main Procedure
1. Image **feature extraction** with `SIFT` algorithm
2. Image feature points **matching** with `RANSAC` algorithm
3. Image **blending** with matched feature points

#### Image feature extraction with `SIFT` algorithm
> relevant code: `MySift.h` and `MySift.cpp`
- results of key feature points (each with a feature descriptor of 128 dimention) of two images:

![Image text](https://github.com/Novice-coder21/Image_Stitching-Tool/blob/main/Images/kps.png)

#### Image feature points **matching** with `RANSAC` algorithm
> relevant code: `MyMatching.h` and `MyMatching.cpp`
- First do a *coarse-grained* feature points matching by calculating the distance of two feature descriptors, and regard the two points as matched if the distance is lower than some threshold. The matched points are lined together as shown below:

![Image text](https://github.com/Novice-coder21/Image_Stitching-Tool/blob/main/Images/kps_real.png)

![Image text](https://github.com/Novice-coder21/Image_Stitching-Tool/blob/main/Images/mixImgWithLine.png)

- Clearly there exist many outliers, which can be removed by `RANSAC` algorithm as shown below. The algorithm works on selecting the main transforming direction with most inliers:

![Image text](https://github.com/Novice-coder21/Image_Stitching-Tool/blob/main/Images/ransac.png)

- Removed the outliers which are conflicted with the selected transforming direction:

![Image text](https://github.com/Novice-coder21/Image_Stitching-Tool/blob/main/Images/mixImgWithLine_fixed.png)

#### Image **blending** with matched feature points
> relevant code: `MyBlending.h` and `MyBlending.cpp`
- First use a simple translation method:

![Image text](https://github.com/Novice-coder21/Image_Stitching-Tool/blob/main/Images/mixImg.png)

becomes

![Image text](https://github.com/Novice-coder21/Image_Stitching-Tool/blob/main/Images/blended.png)

- Then apply a RGB interpolation at fusion region `A/B`:

![Image text](https://github.com/Novice-coder21/Image_Stitching-Tool/blob/main/Images/blend.png)

- Stitched Result of two images

![Image text](https://github.com/Novice-coder21/Image_Stitching-Tool/blob/main/Images/blendedImg.png)

- Repeat this procedure and get the stitched Result of all images

![Image text](https://github.com/Novice-coder21/Image_Stitching-Tool/blob/main/Images/blendedImg0.png)
