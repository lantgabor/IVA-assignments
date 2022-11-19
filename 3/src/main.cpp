#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

#include "helper.h"

using namespace cv;
using namespace std;

int main(int argc, char const* argv[])
{
    // LOAD IMAGE
    cv::String keys = "{@input |../data/blood.png|input image}"
                      "{@min |3 |min range of diameters}"
                      "{@max |32 |max range of diameters}";

    cv::CommandLineParser parser(argc, argv, keys);

    String imageName = parser.get<String>("@input");
    std::string image_path = samples::findFile(imageName);
    Mat img = imread(image_path, IMREAD_COLOR);
    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    imshow("original", img);

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    medianBlur(gray, gray, 5);

    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
        img.rows / 16, // change this value to detect circles with different distances to each other
        100, 30, parser.get<int>("@min"), parser.get<int>("@max") // change the last two parameters
        // (min_radius & max_radius) to detect larger circles
        );
    for (size_t i = 0; i < circles.size(); i++) {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        // circle center
        drawMarker(img, center, Scalar(0, 255, 0), MARKER_CROSS, 8, 1, LINE_AA);
        // circle outline
        int radius = c[2];
        circle(img, center, radius, Scalar(255, 0, 255), 1, LINE_AA);
    }
    imshow("detected circles", img);
    int k = waitKey(0); // Wait for a keystroke in the window
    // if (k == 's') {
    //     imwrite(imageName + "-" + std::to_string(val) + "-save.png", img_bw);
    // }
    return 0;
}
