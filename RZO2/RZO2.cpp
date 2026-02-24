#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int main() {
    setlocale(LC_ALL, “ru”);
    Mat canvas(600, 800, CV_8UC3, Scalar(0, 0, 0));
    line(canvas, Point(0, 0), Point(799, 599), Scalar(255, 0, 0), 3); 
    rectangle(canvas, Point(100, 100), Point(300, 250), Scalar(0, 255, 0), 2); 
    circle(canvas, Point(400, 300), 100, Scalar(0, 0, 255), -1); 
    ellipse(canvas, Point(400, 300), Size(150, 70), 30, 0, 360, Scalar(0, 255, 255), 2); // Желтый эллипс

    // 5. Рисуем произвольный многоугольник
    vector<Point> pts = { Point(500, 100), Point(600, 150), Point(650, 250),
                         Point(550, 300), Point(450, 200) };
    polylines(canvas, pts, true, Scalar(255, 255, 0), 2); // Голубой многоугольник, замкнутый

    // 6. Добавляем текстовую подпись (фамилия и номер группы) в правом верхнем углу
    string text = "Лагунова Л.А. Группа 22-ПМ-1";
    int fontFace = FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.0;
    int thickness = 2;
    Scalar textColor = Scalar(255, 255, 255); // Белый цвет текста

    // Получаем размер текста для правильного позиционирования
    int baseline = 0;
    Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);

    // Позиция в правом верхнем углу (с отступом 10 пикселей от краев)
    Point textOrg(canvas.cols - textSize.width - 10, textSize.height + 10);

    putText(canvas, text, textOrg, fontFace, fontScale, textColor, thickness);

    // Отображаем результат на экране
    namedWindow("Результат", WINDOW_NORMAL);
    imshow("Результат", canvas);

    // Сохраняем итоговое изображение на диск
    imwrite("result.png", canvas);
    cout << "Изображение сохранено как result.png" << endl;

    // Ожидаем нажатия клавиши для закрытия окна
    cout << "Нажмите любую клавишу для выхода..." << endl;
    waitKey(0);

    return 0;
}