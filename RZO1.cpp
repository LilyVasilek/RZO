#include <opencv2/opencv.hpp>
#include <iostream>
#include <clocale>
#include <filesystem>
#include <string>

int main(int argc, char** argv) {
    setlocale(LC_ALL, "");

    namespace fs = std::filesystem;

    // Запуск:
    //   app.exe [imagePath] [outDir]
    const std::string usedPath = (argc >= 2) ? std::string(argv[1]) : std::string("C:/Users/lagun/Pictures/forest.jpg");
    const std::string outDirArg = (argc >= 3) ? std::string(argv[2]) : std::string("out");

    cv::Mat image = cv::imread(usedPath);
    if (image.empty()) {
        std::cerr << "Ошибка: Не удалось загрузить изображение: " << usedPath << std::endl;
        return -1;
    }

    std::cout << "Загружено изображение: " << usedPath << std::endl;
    cv::imshow("Original image", image);

    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    cv::imshow("Gray image", grayImage);

    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
    cv::imshow("Resized image", resizedImage);

    cv::Mat rotatedImage;
    cv::rotate(image, rotatedImage, cv::ROTATE_90_CLOCKWISE);
    cv::imshow("Rotated image", rotatedImage);

    cv::Mat blurredImage;
    cv::GaussianBlur(image, blurredImage, cv::Size(15, 15), 0);
    cv::imshow("Blurred image", blurredImage);

    cv::Mat edges;
    cv::Canny(grayImage, edges, 50, 150);
    cv::imshow("Edges image", edges);

    // Сохранение результатов в папку
    try {
        fs::path outDir(outDirArg);
        if (outDir.empty()) outDir = fs::path("out");
        fs::create_directories(outDir);

        const std::string base = fs::path(usedPath).stem().string();
        const std::string safeBase = base.empty() ? std::string("image") : base;

        const fs::path p0 = outDir / (safeBase + "_orig.png");
        const fs::path p1 = outDir / (safeBase + "_gray.png");
        const fs::path p2 = outDir / (safeBase + "_resized.png");
        const fs::path p3 = outDir / (safeBase + "_rotated.png");
        const fs::path p4 = outDir / (safeBase + "_blurred.png");
        const fs::path p5 = outDir / (safeBase + "_edges.png");

        cv::imwrite(p0.string(), image);
        cv::imwrite(p1.string(), grayImage);
        cv::imwrite(p2.string(), resizedImage);
        cv::imwrite(p3.string(), rotatedImage);
        cv::imwrite(p4.string(), blurredImage);
        cv::imwrite(p5.string(), edges);

        std::cout << "Сохранено в папку: " << outDir.string() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Ошибка при сохранении: " << e.what() << std::endl;
    }

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}