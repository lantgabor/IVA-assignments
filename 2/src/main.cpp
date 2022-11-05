#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

#include "helper.h"

using namespace cv;
using namespace std;

void histogram(Mat& src, Mat& hists)
{
    /* Calc histogram */
    int histSize = 256;

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            hists.at<float>(src.at<uchar>(i, j)) += 1;
        }
    }
}

int Otsu(Mat& src, Mat& dst, Mat& hists, Mat& vars)
{

    // reset dst image
    dst = Mat::zeros(src.rows, src.cols, CV_8UC1);
    int threshold = 0;

    size_t total = src.total();
    int histSize = 256;
    float varMax = 0, sum = 0, sumB = 0, q1 = 0, q2 = 0;

    for (int i = 0; i < histSize; i++) {
        sum += i * hists.at<float>(i);
    }

    // Algorithm explained: http://www.ipol.im/pub/art/2016/158/article.pdf
    for (int i = 0; i < histSize; i++) {
        q1 += hists.at<float>(i); // q1(t) = Sum_0^t { P(i) }

        if (q1 == 0) // skip early zeros
            continue;

        q2 = total - q1; // 1 - q1 or q2(t) = Sum_t+1^255 { P(i) }

        if (q2 == 0) // skip late zeros
            break;

        sumB += i * hists.at<float>(i); // i * P(i)
        float u1 = sumB / q1;
        float u2 = (sum - sumB) / q2;

        float betweenVariance = q1 * q2 * (u1 - u2) * (u1 - u2); // q1(t)[1-q1(t)][u1(t)-u2(t)]^2

        vars.at<float>(i) = betweenVariance;

        if (betweenVariance > varMax) { // Find maximum variance
            varMax = betweenVariance;
            threshold = i;
        }
    }

    /* display thresholded image */
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (src.at<uchar>(i, j) >= threshold) {
                dst.at<uchar>(i, j) = 255;
            }
        }
    }

    return threshold;
}

int main(int argc, char const* argv[])
{
    // LOAD IMAGE
    cv::CommandLineParser parser(argc, argv, "{@input |../data/julia.png|input image}");
    String imageName = parser.get<String>("@input");
    std::string image_path = samples::findFile(imageName);
    Mat img = imread(image_path, 0);
    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    imshow("original", img);

    Mat hists = Mat::zeros(1, 256, CV_32FC1);
    histogram(img, hists);

    Mat img_bw;
    int val = 0;
    // val = threshold(img, img_bw, 0, 255, THRESH_OTSU);
    // std::cout << "Otsu obtained value (opencv): " << val << std::endl;

    Mat vars = Mat::zeros(1, 256, CV_32FC1);
    val = Otsu(img, img_bw, hists, vars);
    std::cout << "Otsu obtained value: " << val << std::endl;

    imshow("img_bw", img_bw);

    /* Display the histogram */
    int hist_w = 512, hist_h = 512;
    int bin_w = cvRound((double)hist_w / 256);
    Mat histsNorm;
    Mat varsNorm;
    Mat histogram(hist_h, hist_w, CV_8UC3, Scalar(0));
    normalize(hists, histsNorm, 0, histogram.rows, NORM_MINMAX, -1, Mat());
    normalize(vars, varsNorm, 0, histogram.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < 256; i++) {
        line(histogram, Point(bin_w * (i - 1), hist_h - cvRound(histsNorm.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(histsNorm.at<float>(i))),
            Scalar(255, 0, 0), 1, 8, 0);
        line(histogram, Point(bin_w * (i - 1), hist_h - cvRound(varsNorm.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(varsNorm.at<float>(i))),
            Scalar(0, 0, 255), 1, 8, 0);
    }
    line(histogram, Point(bin_w * (val), hist_h), Point(bin_w * (val), 0), Scalar(0, 255, 0), 1, 8, 0);

    imshow("histogram", histogram);

    int k = waitKey(0); // Wait for a keystroke in the window
    if (k == 's') {
        imwrite(imageName + "-" + std::to_string(val) + "-save.png", img_bw);
        imwrite(imageName + "-histogram.png", histogram);
    }
    return 0;
}
