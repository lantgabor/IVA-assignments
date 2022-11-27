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

    int a, b;

    // Try with perpendicular points only
    voteWithBoundaries(A, x, y, r, rows, cols, min, max);
    angle = angle - 90;
    a = x + r * cos(angle * CV_PI / 180);
    b = y - r * sin(angle * CV_PI / 180);
    voteWithBoundaries(A, a, b, r, rows, cols, min, max);

    a = x - r * cos(angle * CV_PI / 180);
    b = y + r * sin(angle * CV_PI / 180);
    voteWithBoundaries(A, a, b, r, rows, cols, min, max);

    // Try with full circles
    // for (int t = 0; t < 360; t++) {
    //     a = x + r * cos(t * CV_PI / 180);
    //     b = y + r * sin(t * CV_PI / 180);
    //     voteWithBoundaries(A, a, b, r, rows, cols, min, max);
    // }
}

// https://en.wikipedia.org/wiki/Circle_Hough_Transform
void HC(Mat& gray, Mat& accumulatorImg, vector<Vec3f>& circles, int minDist, int param1, int param2, int min, int max)
{

    // Edge detection
    Mat edges;
    Canny(gray, edges, param1 / 2, param1, 3);

    // Get angles using Sobel
    Mat dx, dy;
    Sobel(gray, dx, CV_32FC1, 1, 0);
    Sobel(gray, dy, CV_32FC1, 0, 1);
    Mat angles;
    phase(dx, dy, angles, true);

    // 3D accumulator A[a,b,r] - cones[centerX, centerY, radius]
    int sizes[] = { gray.rows, gray.cols, max - min };
    Mat A(3, sizes, CV_8UC1, Scalar(0));

    for (int r = min; r < max; r += minDist) { // radii in range [min-max]

        for (int x = 0; x < edges.rows; x++) {
            for (int y = 0; y < edges.cols; y++) {
                if (edges.at<uchar>(x, y) > 0) { // if edges
                    voteCircle(A, x, y, r, angles.at<float>(x, y), edges.rows, edges.cols, min, max);
                }
            }
        }

        // Add center to cirlces if for all above the threshold
        for (int x = 0; x < edges.rows; x++) {
            for (int y = 0; y < edges.cols; y++) {
                if (A.at<uchar>(x, y, r - min) > param2) {
                    circles.push_back(Vec3f(y, x, r));
                }
            }
        }
    }

    // Display some accumulators
    for (size_t r = 0; r < max - min; r++) {
        for (int x = 0; x < gray.rows; x++) {
            for (int y = 0; y < gray.cols; y++) {
                accumulatorImg.at<uchar>(x, y) = A.at<uchar>(x, y, r);
            }
        }
    }
}

int main(int argc, char const* argv[])
{
    // LOAD IMAGE
    cv::String keys = "{@input |../data/circles.png|input image}"
                      "{@min |5 |min range of diameters}"
                      "{@max |15 |max range of diameters}"
                      "{@param1 |155 |threshold of the Canny edge detector}"
                      "{@param2 |15 | accumulator threshold for the circle centers}";

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
    Mat accumulatorImg = Mat::zeros(gray.rows, gray.cols, CV_8UC1);

    // OpenCV Reference
    // HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
    //     img.rows / 16, // change this value to detect circles with different distances to each other
    //     100, 30, parser.get<int>("@min"), parser.get<int>("@max") // change the last two parameters
    //     // (min_radius & max_radius) to detect larger circles
    //     );

    HC(gray, accumulatorImg, circles, 1, parser.get<int>("@param1"), parser.get<int>("@param2"), parser.get<int>("@min"), parser.get<int>("@max"));

    // Display accumulator image
    accumulatorImg = accumulatorImg * 10; // scale for more visibility
    imshow("accumulatorImg", accumulatorImg);

    // Display cirlces and centers on original img
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
    if (k == 's') {
        imwrite(imageName + "_accumulator.png", accumulatorImg);
        imwrite(imageName + "_detected.png", img);
    }
    return 0;
}
