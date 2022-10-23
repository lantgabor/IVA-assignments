#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

#include "helper.h"

using namespace cv;

double Otsu(Mat& in, Mat& out, int a, int b, int c)
{
    double thresh = 0;
    out = Mat::zeros(in.rows, in.cols, CV_8UC1);
    for (int i = 0; i < in.rows; i++) {
        for (int j = 0; j < in.cols; j++) {
            out.at<uchar>(i, j) = 128;
        }
    }

    return thresh;
}

int main(int argc, char const* argv[])
{
    // LOAD IMAGE
    cv::CommandLineParser parser(argc, argv, "{@input |../data/finger.png|input image}");
    String imageName = parser.get<String>("@input");
    std::string image_path
        = samples::findFile(imageName);
    Mat img = imread(image_path, 0);
    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    imshow("original", img);
    check_data(img, "img");

    Mat img_bw;
    double val = 0;
    val = threshold(img, img_bw, 0, 255, THRESH_OTSU);
    val = Otsu(img, img_bw, 0, 255, THRESH_OTSU);

    imshow("img_bw", img_bw);
    std::cout << val << std::endl;

    int k = waitKey(0); // Wait for a keystroke in the window

    return 0;
}
