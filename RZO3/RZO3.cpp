#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    Mat img = imread("markers.png");

    if (img.empty()) {
        cout << "Ошибка загрузки изображения" << endl;
        return -1;
    }

    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    Mat mask_rose1, mask_rose2, mask_rose;

    inRange(hsv, Scalar(160, 50, 100), Scalar(180, 255, 255), mask_rose1);
    inRange(hsv, Scalar(0, 50, 100), Scalar(10, 255, 255), mask_rose2);

    mask_rose = mask_rose1 | mask_rose2;

    imshow("image_png.png", img);
    imshow("Rose Mask", mask_rose);

    Mat result;
    img.copyTo(result, mask_rose);
    imshow("Rose Objects", result);

    waitKey(0);
    return 0;
}