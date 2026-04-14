#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main() {
    string facePath = "C:\\Users\\lagun\\Programs\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml";
    string eyePath = "C:\\Users\\lagun\\Programs\\opencv\\sources\\data\\haarcascades\\haarcascade_eye.xml";
    string smilePath = "C:\\Users\\lagun\\Programs\\opencv\\sources\\data\\haarcascades\\haarcascade_smile.xml";

    CascadeClassifier faceCascade, eyeCascade, smileCascade;
    if (!faceCascade.load(facePath) || !eyeCascade.load(eyePath) || !smileCascade.load(smilePath)) {
        cerr << "Ошибка загрузки каскадов" << endl;
        return -1;
    }

    VideoCapture cap("ZUA.mp4");
    if (!cap.isOpened()) {
        cerr << "Не удалось открыть видео" << endl;
        return -1;
    }

    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;

    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    if (width <= 0 || height <= 0) {
        cerr << "Не удалось получить размер кадра" << endl;
        return -1;
    }

    VideoWriter writer("output_detected.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height));
    if (!writer.isOpened()) {
        cerr << "Предупреждение: не удалось открыть VideoWriter. Сохранение видео невозможно." << endl;
    }

    const double resizeFactor = 0.5;
    Mat frame, smallGray;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        Mat smallFrame;
        resize(frame, smallFrame, Size(), resizeFactor, resizeFactor);
        cvtColor(smallFrame, smallGray, COLOR_BGR2GRAY);
        equalizeHist(smallGray, smallGray);

        vector<Rect> faces;
        faceCascade.detectMultiScale(smallGray, faces, 1.3, 7, 0, Size(40, 40));

        for (auto& face : faces) {
            face.x = cvRound(face.x / resizeFactor);
            face.y = cvRound(face.y / resizeFactor);
            face.width = cvRound(face.width / resizeFactor);
            face.height = cvRound(face.height / resizeFactor);

            rectangle(frame, face, Scalar(0, 255, 0), 2);

            Mat grayFace;
            cvtColor(frame(face), grayFace, COLOR_BGR2GRAY);
            equalizeHist(grayFace, grayFace);

            vector<Rect> eyes;
            eyeCascade.detectMultiScale(grayFace, eyes, 1.1, 8, 0, Size(12, 12), Size(face.width / 2, face.height / 3));
            vector<Rect> validEyes;
            for (const auto& e : eyes) {
                if (e.y + e.height > face.height / 2) continue;
                if (e.width < e.height * 0.8) continue;
                validEyes.push_back(e);
            }
            if (validEyes.size() > 2) {
                sort(validEyes.begin(), validEyes.end(),
                    [](const Rect& a, const Rect& b) { return a.area() > b.area(); });
                validEyes.resize(2);
            }
            for (const auto& e : validEyes) {
                Point center(face.x + e.x + e.width / 2, face.y + e.y + e.height / 2);
                int radius = cvRound((e.width + e.height) * 0.25);
                circle(frame, center, radius, Scalar(255, 0, 0), 2);
            }

            vector<Rect> smiles;
            smileCascade.detectMultiScale(grayFace, smiles, 1.1, 25, 0, Size(25, 25), Size(face.width * 0.8, face.height * 0.5));

            vector<Rect> validSmiles;
            for (const auto& s : smiles) {
                if (s.y < face.height / 2) continue;
                if (s.width <= s.height) continue;
                if (s.width < face.width * 0.25) continue;
                if (s.area() < 200) continue;
                if (s.height > face.height * 0.35) continue;
                validSmiles.push_back(s);
            }

            if (!validSmiles.empty()) {
                auto bestSmile = *max_element(validSmiles.begin(), validSmiles.end(),
                    [](const Rect& a, const Rect& b) { return a.area() < b.area(); });
                Rect smileGlobal(face.x + bestSmile.x, face.y + bestSmile.y,
                    bestSmile.width, bestSmile.height);
                rectangle(frame, smileGlobal, Scalar(0, 0, 255), 2);
            }
        }

        imshow("Детекция", frame);

        if (writer.isOpened()) {
            writer.write(frame);
        }

        if (waitKey(30) == 27) break;
    }

    cap.release();
    writer.release();
    destroyAllWindows();
    return 0;
}