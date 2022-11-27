#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

#include "helper.h"

using namespace cv;
using namespace std;

void voteWithBoundaries(Mat& A, int x, int y, int r, int rows, int cols, int min, int max)
{
    if (x < 0 || x >= rows || y < 0 || y >= cols)
        return;
    else
        A.at<uchar>(x, y, r - min) += 1;
}

void voteCircle(Mat& A, int x, int y, int r, float angle, int rows, int cols, int min, int max)
{
    voteWithBoundaries(A, x, y, r, rows, cols, min, max);
    int a, b;
    for (int t = 0; t < 360; t++) {
        a = x + r * cos(t * CV_PI / 180);
        b = y + r * sin(t * CV_PI / 180);
        voteWithBoundaries(A, a, b, r, rows, cols, min, max);
    }
}

// https://en.wikipedia.org/wiki/Circle_Hough_Transform
void HC(Mat& gray, vector<Vec3f>& circles, int min, int max)
{

    // Edge detection
    Mat edges;
    Canny(gray, edges, 50, 200, 3);

    imshow("canny", edges);

    Mat dx, dy;
    Sobel(gray, dx, CV_32FC1, 1, 0);
    Sobel(gray, dy, CV_32FC1, 0, 1);
    Mat angles;

    phase(dx, dy, angles, false);
    // angles = angles / 360 * 255;
    // angles.convertTo(angles, CV_8U);

    // 3D accumulator A[a,b,r] - cones
    // Init zero
    int sizes[] = { gray.rows, gray.cols, max - min };
    Mat A(3, sizes, CV_8UC1, Scalar(0));

    for (int r = min; r < max; r++) {

        for (int x = 0; x < edges.rows; x++) {
            for (int y = 0; y < edges.cols; y++) {
                if (edges.at<uchar>(x, y) > 0) {
                    voteCircle(A, x, y, r, angles.at<float>(x, y), edges.rows, edges.cols, min, max);
                }
            }
        }

        for (int x = 0; x < edges.rows; x++) {
            for (int y = 0; y < edges.cols; y++) {
                if (A.at<uchar>(x, y, r - min) > 155) {
                    circles.push_back(Vec3f(y, x, r));
                }
            }
        }
    }

    for (size_t r = min; r < max; r += 10) {
        Mat test = Mat::zeros(gray.rows, gray.cols, CV_8UC1);
        for (int x = 0; x < gray.rows; x++) {
            for (int y = 0; y < gray.cols; y++) {
                test.at<uchar>(x, y) = A.at<uchar>(x, y, r - min);
            }
        }
        imshow("test" + r, test);
    }
}

int main(int argc, char const* argv[])
{
    // LOAD IMAGE
    cv::String keys = "{@input |../data/cable.png|input image}"
                      "{@min |32 |min range of diameters}"
                      "{@max |128 |max range of diameters}";

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
    // medianBlur(gray, gray, 5);
    vector<Vec3f> circles;

    // HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
    //     img.rows / 16, // change this value to detect circles with different distances to each other
    //     100, 30, parser.get<int>("@min"), parser.get<int>("@max") // change the last two parameters
    //     // (min_radius & max_radius) to detect larger circles
    //     );

    HC(gray, circles, parser.get<int>("@min"), parser.get<int>("@max"));

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
