# Image_Stitching-Tool
A unique software tool that allows you to create panoramas by attaching images with overlapping areas.

## Software required
* An IDE such as VS code or Visual Studio
* A C++ compiler (MS VC is optimum)
* CMake
* OpenCV library version 3.2.0
* CImg library : http://www.cimg.eu/

## set up
* Download the required libraries and link them to your compilers and IDEs.
* Using CMake make a CMakelists.txt file that configures the local build files.
* Create and exe file and place accordingly. Use the exe to run the program.

## Main Procedure
1. Image **feature extraction** with `SIFT` algorithm
2. Image feature points **matching** with `RANSAC` algorithm
3. Image **blending** with matched feature points

