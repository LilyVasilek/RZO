#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

string detectShape(const vector<Point>& approx, double area, double perimeter) {
    int vertices = (int)approx.size();
    if (vertices == 3) return "Triangle";
    if (vertices == 4) {
        Rect rect = boundingRect(approx);
        double aspectRatio = (double)rect.width / rect.height;
        if (aspectRatio >= 0.95 && aspectRatio <= 1.05)
            return "Square";
        else
            return "Rectangle";
    }
    if (vertices == 5) return "Pentagon";
    if (vertices == 6) return "Hexagon";
    double circularity = 4 * CV_PI * area / (perimeter * perimeter);
    if (circularity > 0.8) return "Circle";
    return "Unknown";
}

int main() {
    VideoCapture cap("Motion abstract geometric shapes.mkv");
    if (!cap.isOpened()) {
        cerr << "Error opening video file!\n";
        return -1;
    }

    double fps = cap.get(CAP_PROP_FPS);
    int width = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);

    VideoWriter writer("result.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height));
    if (!writer.isOpened()) {
        cerr << "Error creating output video file!\n";
        return -1;
    }

    Mat frame, gray, binary;
    vector<vector<Point>> contours;

    while (cap.read(frame)) {
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        threshold(gray, binary, 240, 255, THRESH_BINARY_INV);
        findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            double perimeter = arcLength(contour, true);
            double area = contourArea(contour);
            if (area < 50) continue;

            vector<Point> approx;
            double epsilon = 0.02 * perimeter;
            approxPolyDP(contour, approx, epsilon, true);

            string shapeName = detectShape(approx, area, perimeter);

            Scalar color;
            if (shapeName == "Triangle") color = Scalar(255, 0, 0);
            else if (shapeName == "Square") color = Scalar(0, 255, 0);
            else if (shapeName == "Rectangle") color = Scalar(0, 255, 255);
            else if (shapeName == "Circle") color = Scalar(0, 165, 255);
            else if (shapeName == "Pentagon") color = Scalar(255, 255, 0);
            else if (shapeName == "Hexagon") color = Scalar(255, 0, 255);
            else color = Scalar(255, 255, 255);

            Rect bbox = boundingRect(contour);
            rectangle(frame, bbox, color, 2);

            string text = shapeName + " | " + to_string((int)area);
            int fontFace = FONT_HERSHEY_DUPLEX;
            double fontScale = 0.5;
            int baseline = 0;
            Size textSize = getTextSize(text, fontFace, fontScale, 1, &baseline);
            Point textPos(bbox.x + 5, bbox.y - 5);
            if (textPos.y - textSize.height < 0) {
                textPos = Point(bbox.x + 5, bbox.y + textSize.height + 5);
            }
            putText(frame, text, textPos, fontFace, fontScale, Scalar(0, 0, 0), 1);
        }

        string info = "Shapes: " + to_string(contours.size());
        putText(frame, info, Point(10, 30), FONT_HERSHEY_DUPLEX, 0.7, Scalar(0, 0, 0), 1);

        writer.write(frame);

        imshow("Detected Shapes", frame);
        imshow("Binary Mask", binary);

        if (waitKey(30) == 27) break;
    }

    cap.release();
    writer.release();
    destroyAllWindows();
    return 0;
}
