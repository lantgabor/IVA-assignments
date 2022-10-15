#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "helper.h"
using namespace cv;

void filterPrewitt(Mat& in, Mat& magnitude, Mat& theta, Mat& Kx, Mat& Ky)
{
    const auto W = in.cols;
    const auto H = in.rows;

    const auto W_kern = Kx.cols / 2;
    const auto H_kern = Kx.rows / 2;

    // over the input image
    for (int i = H_kern; i < H - H_kern; i++) {
        for (int j = W_kern; j < W - W_kern; j++) {

            // over the filter
            float magX = 0;
            float magY = 0;

            for (int k = -H_kern; k <= H_kern; k++) {
                for (int l = -W_kern; l <= W_kern; l++) {
                    magX += in.at<uchar>(i + k, j + l) * Kx.at<float>(H_kern + k, W_kern + l);
                    magY += in.at<uchar>(i + k, j + l) * Ky.at<float>(H_kern + k, W_kern + l);
                }
            }

            magnitude.at<float>(i, j) = sqrtf(powf(magX, 2) + powf(magY, 2));
            theta.at<float>(i, j) = atanf(magY / magX);
        }
    }

    // normalise
    double min, max;
    cv::minMaxLoc(magnitude, &min, &max);
    magnitude = magnitude / max;
}

void NMS(Mat& in, Mat& out, Mat& angles)
{
    angles = angles * 180 / CV_PI;
    for (int i = 2; i < in.rows - 1; i++) {
        for (int j = 2; j < in.cols - 1; j++) {
            if (angles.at<float>(i, j) >= -22.5 && angles.at<float>(i, j) <= 22.5) {
                if ((in.at<float>(i, j) >= in.at<float>(i, j + 1)) && (in.at<float>(i, j) >= in.at<float>(i, j - 1)))
                    out.at<float>(i, j) = in.at<float>(i, j);
            }
            else if (angles.at<float>(i, j) >= 22.5 && angles.at<float>(i, j) <= 67.5) {
                if ((in.at<float>(i, j) >= in.at<float>(i + 1, j + 1)) && (in.at<float>(i, j) >= in.at<float>(i - 1, j - 1)))
                    out.at<float>(i, j) = in.at<float>(i, j);
            }
            else if (
                (angles.at<float>(i, j) < -22.5 && angles.at<float>(i, j) >= -67.5)) {
                if ((in.at<float>(i, j) >= in.at<float>(i + 1, j - 1)) && (in.at<float>(i, j) >= in.at<float>(i - 1, j + 1)))
                    out.at<float>(i, j) = in.at<float>(i, j);
            }
            else if (
                (angles.at<float>(i, j) >= 67.5 && angles.at<float>(i, j) <= 90) || (angles.at<float>(i, j) < -67.5 && angles.at<float>(i, j) >= -90)) {
                if (
                    (in.at<float>(i, j) >= in.at<float>(i + 1, j)) && (in.at<float>(i, j) >= in.at<float>(i - 1, j)))
                    out.at<float>(i, j) = in.at<float>(i, j);
            }
        }
    }
}

int main(int argc, char** argv)
{
    // LOAD IMAGE
    cv::CommandLineParser parser(argc, argv, "{@input |../data/julia.png|input image}");
    String imageName = parser.get<String>("@input");
    std::string image_path
        = samples::findFile(imageName);
    Mat img = imread(image_path, 0);
    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    imshow("original", img);

    Mat img_blur;
    GaussianBlur(img, img_blur, Size(3, 3), 0);

    // Prewitt horizontal
    float p_kernelY[9] = { 1, 1, 1, 0, 0, 0, -1, -1, -1 };
    cv::Mat prewittY = cv::Mat(3, 3, CV_32F, p_kernelY);
    // std::cout << prewittY << std::endl;

    // Prewitt vertical
    float p_kernelX[9] = { 1, 0, -1, 1, 0, -1, 1, 0, -1 };
    cv::Mat prewittX = cv::Mat(3, 3, CV_32F, p_kernelX);
    // std::cout << prewittX << std::endl;

    cv::Mat magnitude = Mat::zeros(img.rows, img.cols, CV_32F);
    cv::Mat theta = Mat::zeros(img.rows, img.cols, CV_32F);
    cv::Mat nms = Mat::zeros(img.rows, img.cols, CV_32F);

    filterPrewitt(img, magnitude, theta, prewittX, prewittY);
    // check_data(magnitude, "magnitude");
    // check_data(theta, "theta");
    imshow("magnitude", magnitude);

    NMS(magnitude, nms, theta);
    // check_data(nms, "nms");

    magnitude.convertTo(magnitude, CV_8UC3, 255.0);
    imwrite("magnitude.png", magnitude);
    nms.convertTo(nms, CV_8UC3, 255.0);
    imshow("nms", nms);
    imwrite("nms.png", nms);

    int k = waitKey(0); // Wait for a keystroke in the window
    // if (k == 's') {
    //     imwrite("save.png", img);
    // }

    // if (true) {
    //     std::cerr << "Usage: " << argv[0] << "input" << std::endl;
    //     return 1;
    // }

    return 0;
}