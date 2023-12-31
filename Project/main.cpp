#include "stdafx.h"
#include "MyMatching.h"
#include "MyBlending.h"

/* function to convert file types to bmp.
or make other types sucn as .png .jpg .jpeg */

void Success(const char* success_message) {
     MessageBox(nullptr, success_message, "Successful Stitching", MB_ICONINFORMATION);
}

struct ImageInfo {
    char* path;
};

int main(){
    // Create a vector to store information for each image
    vector <ImageInfo> imageList;
    // Add information for each input image
    imageList.push_back({ "C:\\Users\\PMLS\\Desktop\\OOP_Project_Images\\Inputs\\1.bmp"});
    imageList.push_back({ "C:\\Users\\PMLS\\Desktop\\OOP_Project_Images\\Inputs\\2.bmp"});
    imageList.push_back({ "C:\\Users\\PMLS\\Desktop\\OOP_Project_Images\\Inputs\\3.bmp"});
    // Add more images as needed
    char* resultaddr = "C:\\Users\\PMLS\\Desktop\\OOP_Project_Images\\Outputs\\result.bmp";
    char* x;
    char* y;

    MySift mySift1;
    MySift mySift2;

    for (int i = 0; i < imageList.size()-1 ; i++)
    {
        if(i==0){
            mySift1 = MySift(imageList[i].path, 1);
            mySift1.SiftMainProcess();
            mySift1.saveImgWithKeypoint("C:\\Users\\PMLS\\Desktop\\OOP_Project_Images\\Outputs\\1-2\\1_kp.bmp");
            x = imageList[i].path;

            mySift2 = MySift(imageList[i+1].path, 1);
            mySift2.SiftMainProcess();
            mySift2.saveImgWithKeypoint("C:\\Users\\PMLS\\Desktop\\OOP_Project_Images\\Outputs\\1-2\\2_kp.bmp");
            y = imageList[i+1].path;
        }

        else{
            mySift1 = MySift(resultaddr, 1);
            mySift1.SiftMainProcess();
            mySift1.saveImgWithKeypoint("C:\\Users\\PMLS\\Desktop\\OOP_Project_Images\\Outputs\\1-2\\1_kp.bmp");
            x = resultaddr;

            mySift2 = MySift(imageList[i+1].path, 1);
            mySift2.SiftMainProcess();
            mySift2.saveImgWithKeypoint("C:\\Users\\PMLS\\Desktop\\OOP_Project_Images\\Outputs\\1-2\\2_kp.bmp");
            y = imageList[i+1].path;
        }

    MyMatching myMatching(mySift1.getKeyPointsCount(), mySift1.getFirstKeyDescriptors(),
    mySift2.getKeyPointsCount(), mySift2.getFirstKeyDescriptors());
    myMatching.featureMatchMainProcess();
    myMatching.drawOriKeypointOnImg(x , y,"C:\\Users\\PMLS\\Desktop\\OOP_Project_Images\\Outputs\\1-2\\1_kp_real.bmp", "C:\\Users\\PMLS\\Desktop\\OOP_Project_Images\\Outputs\\1-2\\2_kp_real.bmp");
    myMatching.mixImageAndDrawPairLine("C:\\Users\\PMLS\\Desktop\\OOP_Project_Images\\Outputs\\1-2/mixImg.bmp", "C:\\Users\\PMLS\\Desktop\\OOP_Project_Images\\Outputs\\1-2\\mixImgWithLine.bmp");
    myMatching.myRANSACtoFindKpTransAndDrawOut("C:\\Users\\PMLS\\Desktop\\OOP_Project_Images\\Outputs\\1-2\\mixImgWithLine_fixed.bmp");

    MyBlending myBlending(myMatching.getMatchVec().col, myMatching.getMatchVec().row);
    myBlending.blendingMainProcess(x, y);
    myBlending.saveBlendedImg("C:\\Users\\PMLS\\Desktop\\OOP_Project_Images\\Outputs\\result.bmp");
    }
    Success("Image Stitched together successfully and saved at provided location.");
    return 0;
    }
