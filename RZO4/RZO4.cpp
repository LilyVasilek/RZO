#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

std::string identifyShape(const std::vector<cv::Point>& approx, double area, double perimeter) {
    int vertices = approx.size();
    double circularity = 4 * CV_PI * area / (perimeter * perimeter);

    if (vertices == 3) return "Triangle";
    else if (vertices == 4) {
        // Вычисляем внутренние углы через скалярное произведение
        std::vector<double> angles;
        for (int i = 0; i < 4; i++) {
            cv::Point p1 = approx[i];
            cv::Point p2 = approx[(i + 1) % 4];
            cv::Point p3 = approx[(i + 2) % 4];

            cv::Point v1 = p2 - p1;
            cv::Point v2 = p3 - p2;

            double dot = v1.x * v2.x + v1.y * v2.y;
            double len1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
            double len2 = std::sqrt(v2.x * v2.x + v2.y * v2.y);
            double cos_angle = dot / (len1 * len2);
            // Защита от выхода за диапазон из-за погрешностей
            cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
            double angle = std::acos(cos_angle) * 180 / CV_PI;
            angles.push_back(angle);
        }

        // Считаем углы, близкие к 90° (допуск ±15°)
        int rightAngleCount = 0;
        for (double angle : angles) {
            if (std::abs(angle - 90.0) < 15.0) rightAngleCount++;
        }

        // Для прямоугольника нужно хотя бы 3 угла по 90° (или все 4)
        if (rightAngleCount >= 3) {
            // Повёрнутый прямоугольник для учёта ориентации
            cv::RotatedRect rect = cv::minAreaRect(approx);
            double width = rect.size.width;
            double height = rect.size.height;
            if (width < height) std::swap(width, height);
            double aspectRatio = width / height;

            if (std::abs(aspectRatio - 1.0) < 0.1) return "Square";
            else return "Rectangle";
        }
        else {
            return "Rhombus";
        }
    }
    else if (vertices == 5) return "Pentagon";
    else if (vertices == 6) return "Hexagon";
    else if (vertices >= 8) {
        if (circularity > 0.85) return "Circle";
        else return "Oval";
    }
    else return "Shape";
}

int main() {
    setlocale(LC_ALL, "rus");

    cv::Mat img = cv::imread("shapes.jpg");
    if (img.empty()) {
        std::cout << "Не удалось открыть изображение!" << std::endl;
        return -1;
    }

    cv::Mat result = img.clone();

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::cout << "\nРЕЗУЛЬТАТЫ РАСПОЗНАВАНИЯ\n" << std::endl;

    int figureNumber = 0;

    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area < 500) continue; // игнорируем мелкий шум

        figureNumber++;

        double perimeter = cv::arcLength(contours[i], true);

        std::vector<cv::Point> approx;
        cv::approxPolyDP(contours[i], approx, 0.02 * perimeter, true);

        std::string shapeName = identifyShape(approx, area, perimeter);

        // Центр масс через моменты
        cv::Moments m = cv::moments(contours[i]);
        cv::Point2f center(m.m10 / m.m00, m.m01 / m.m00);

        // Рисуем контур и центр
        cv::drawContours(result, std::vector<std::vector<cv::Point>>{contours[i]}, -1, cv::Scalar(0, 255, 0), 2);
        cv::circle(result, center, 5, cv::Scalar(0, 0, 255), -1);

        // Подпись на английском (избегаем проблем с кириллицей в OpenCV)
        std::string label = shapeName + " P:" + std::to_string((int)perimeter);
        cv::putText(result, label, cv::Point(center.x - 30, center.y - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        // Вывод в консоль (русский)
        std::cout << "Фигура " << figureNumber << ":" << std::endl;
        std::cout << "  Название: " << shapeName << std::endl;
        std::cout << "  Периметр: " << (int)perimeter << std::endl;
        std::cout << "  Центр масс: (" << center.x << ", " << center.y << ")" << std::endl;
        std::cout << "  Количество вершин: " << approx.size() << std::endl;
        std::cout << "------------------------" << std::endl;
    }

    cv::imshow("Границы Canny", edges);
    cv::imshow("Распознанные фигуры", result);

    cv::imwrite("result_with_labels.jpg", result);
    std::cout << "\nРезультат сохранён как 'result_with_labels.jpg'" << std::endl;

    cv::waitKey(0);
    return 0;
}