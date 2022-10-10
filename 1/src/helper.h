#if !defined(HELPER_H)
#define HELPER_H

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;

/* HELPERS FOR OPENCV TYPES GOD HELP ME PLEASE */

// CV_8U -> unsigned char (min = 0, max = 255)
// CV_32F -> float

std::string type2str(int type)
{
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:
        r = "8U";
        break;
    case CV_8S:
        r = "8S";
        break;
    case CV_16U:
        r = "16U";
        break;
    case CV_16S:
        r = "16S";
        break;
    case CV_32S:
        r = "32S";
        break;
    case CV_32F:
        r = "32F";
        break;
    case CV_64F:
        r = "64F";
        break;
    default:
        r = "User";
        break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

void check_data(Mat m, std::string name)
{
    double min, max;
    cv::minMaxLoc(m, &min, &max);

    std::cout << "------- " << name << " -------" << std::endl;
    std::cout << type2str(m.type()) << std::endl;
    std::cout << min << " <-> " << max << std::endl;
    std::cout << m.channels() << std::endl;
    std::cout << "------------------------" << std::endl;
}

#endif // HELPER_H
