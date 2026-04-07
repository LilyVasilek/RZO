#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

struct TrackedObject {
    int id;
    string shapeClass;
    Point2f center;
    Rect bbox;
    int lostCounter;
    static int nextId;
    TrackedObject(const Point2f& c, const Rect& b, const string& cls)
        : center(c), bbox(b), shapeClass(cls), lostCounter(0) {
        id = nextId++;
    }
};
int TrackedObject::nextId = 1;

string classifyShape(const vector<Point>& contour) {
    double area = contourArea(contour);
    if (area < 200) return "Unknown";
    double perimeter = arcLength(contour, true);
    if (perimeter <= 0) return "Unknown";
    vector<Point> approx;
    approxPolyDP(contour, approx, 0.02 * perimeter, true);
    int vertices = approx.size();
    if (vertices == 3) return "Triangle";
    if (vertices == 4) {
        Rect r = boundingRect(contour);
        double aspect = (double)r.width / r.height;
        if (aspect > 0.85 && aspect < 1.15) return "Square";
        else return "Rectangle";
    }
    double circularity = 4 * CV_PI * area / (perimeter * perimeter);
    if (circularity > 0.8 && vertices >= 6) return "Circle";
    return "Unknown";
}

int main() {
    VideoCapture cap("Motion abstract geometric shapes.mkv");
    if (!cap.isOpened()) { cerr << "Error opening video" << endl; return -1; }
    double fps = cap.get(CAP_PROP_FPS);
    Size frameSize(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT));
    VideoWriter writerContours("contours_output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frameSize);
    VideoWriter writerLabels("labeled_output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frameSize);
    if (!writerContours.isOpened() || !writerLabels.isOpened()) { cerr << "Error creating video writers" << endl; return -1; }

    Mat frame, gray, binary;
    vector<TrackedObject> trackedObjects;
    const float MAX_DIST = 50.0f;

    while (cap.read(frame)) {
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        threshold(gray, binary, 240, 255, THRESH_BINARY_INV);
        vector<vector<Point>> contours;
        findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<pair<Point2f, Rect>> newDetections;
        for (const auto& c : contours) {
            if (contourArea(c) < 200) continue;
            Rect bbox = boundingRect(c);
            Point2f center(bbox.x + bbox.width / 2.0f, bbox.y + bbox.height / 2.0f);
            newDetections.emplace_back(center, bbox);
        }

        vector<bool> usedNew(newDetections.size(), false);
        for (auto& obj : trackedObjects) {
            obj.lostCounter++;
            int bestIdx = -1;
            float bestDist = MAX_DIST;
            for (size_t i = 0; i < newDetections.size(); ++i) {
                if (usedNew[i]) continue;
                float dist = norm(obj.center - newDetections[i].first);
                if (dist < bestDist) { bestDist = dist; bestIdx = i; }
            }
            if (bestIdx != -1) {
                obj.center = newDetections[bestIdx].first;
                obj.bbox = newDetections[bestIdx].second;
                obj.lostCounter = 0;
                usedNew[bestIdx] = true;
            }
        }

        for (size_t i = 0; i < newDetections.size(); ++i) {
            if (!usedNew[i]) {
                for (const auto& c : contours) {
                    Rect b = boundingRect(c);
                    if (abs(b.x - newDetections[i].second.x) < 5 && abs(b.y - newDetections[i].second.y) < 5) {
                        string shape = classifyShape(c);
                        if (shape != "Unknown")
                            trackedObjects.emplace_back(newDetections[i].first, newDetections[i].second, shape);
                        break;
                    }
                }
            }
        }

        trackedObjects.erase(remove_if(trackedObjects.begin(), trackedObjects.end(),
            [](const TrackedObject& obj) { return obj.lostCounter > 10; }), trackedObjects.end());

        Mat frameContours = frame.clone();
        for (const auto& c : contours) {
            if (contourArea(c) >= 200)
                drawContours(frameContours, vector<vector<Point>>{c}, -1, Scalar(0, 255, 0), 2);
        }
        writerContours.write(frameContours);

        for (const auto& obj : trackedObjects) {
            rectangle(frame, obj.bbox, Scalar(0, 255, 0), 2);
            Point textCenter(obj.bbox.x + obj.bbox.width / 2, obj.bbox.y + obj.bbox.height / 2);
            putText(frame, obj.shapeClass, textCenter, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        }
        writerLabels.write(frame);

        imshow("Shape Tracking", frame);
        if (waitKey(30) == 27) break;
    }
    destroyAllWindows();
    return 0;
}