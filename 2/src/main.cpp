#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

#include "helper.h"

using namespace cv;
using namespace std;

/* src: https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html */
void historgram(Mat& src, Mat& hists)
{

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange[] = { range };

    calcHist(&src, 1, 0, Mat(), hists, 1, &histSize, histRange, true, false);

    int hist_w = 512, hist_h = 512;
    int bin_w = cvRound((double)hist_w / histSize);

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0));
    normalize(hists, hists, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hists.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(hists.at<float>(i))),
            Scalar(255, 0, 0), 1, 8, 0);
    }

    imshow("calcHist Demo", histImage);
}

double Otsu(Mat& src, Mat& dst, int threshold = 128)
{
    int nb_pixels = src.total();
    int nb_pixels1 = 0;
    dst = Mat::zeros(src.rows, src.cols, CV_8UC1);

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (src.at<uchar>(i, j) >= threshold) {
                dst.at<uchar>(i, j) = 255;
                nb_pixels1++;
            }
            else {
                dst.at<uchar>(i, j) = 0;
            }
        }
    }

    cout << nb_pixels << endl;
    cout << nb_pixels1 << endl;

    return 0;
}

int main(int argc, char const* argv[])
{
    // LOAD IMAGE
    cv::CommandLineParser parser(argc, argv, "{@input |../data/lena.png|input image}");
    String imageName = parser.get<String>("@input");
    std::string image_path = samples::findFile(imageName);
    Mat img = imread(image_path, 0);
    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    imshow("original", img);

    Mat hists;
    historgram(img, hists);

    Mat img_bw;
    double val = 0;
    // val = threshold(img, img_bw, 0, 255, THRESH_OTSU);
    val = Otsu(img, img_bw);

    imshow("img_bw", img_bw);

    int k = waitKey(0); // Wait for a keystroke in the window
    return 0;
}
