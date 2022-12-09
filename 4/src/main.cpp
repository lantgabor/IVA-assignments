#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
// #include <opencv2/tracking.hpp>
#include <time.h>

#include "helper.h"

using namespace cv;
using namespace std;

void DOF(Mat& frame_g, Mat& frame_before_g)
{
    /*  Dense Optical flow
            Two-Frame Motion Estimation Based on Polynomial Expansion" by Gunnar Farneback
    */
    Mat flow(frame_before_g.size(), CV_32FC2);
    calcOpticalFlowFarneback(frame_before_g,
        frame_g,
        flow,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0);
    // visualization
    Mat flow_parts[2];
    split(flow, flow_parts);
    Mat magnitude, angle, magn_norm;
    cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
    normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
    angle *= ((1.f / 360.f) * (180.f / 255.f));
    //build hsv image
    Mat _hsv[3], hsv, hsv8, bgr;
    _hsv[0] = angle;
    _hsv[1] = Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;
    merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cvtColor(hsv8, bgr, COLOR_HSV2BGR);
    imshow("Dense optical flow", bgr);
}

/* src: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html */
int main(int argc, char const* argv[])
{
    // LOAD VIDEO
    cv::String keys = "{@input |../data/halak1.mpg|input video}"
                      "{@numPts |12|number of pts}"
                      "{@cornerQuality |0.01|minimum quality of corner below which everyone is rejected}"
                      "{@minDist |100|min euclidean distance}"
                      "{@windowSize |32|window size}"
                      "{@maxLevel |4|maximal pyramid level}";

    cv::CommandLineParser parser(argc, argv, keys);
    String videoName = parser.get<String>("@input");
    std::string image_path = samples::findFile(videoName);
    VideoCapture capture(image_path);

    if (!capture.isOpened()) {
        std::cout << "Error reading video: " << image_path << std::endl;
        return 1;
    }

    // Create some random colors
    vector<Scalar> colors;
    RNG rng;
    for (int i = 0; i < parser.get<int>("@numPts"); i++) {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r, g, b));
    }

    Mat frame_before, frame_before_g;
    vector<Point2f> p0, p1;

    // Take first frame and find corners in it
    capture >> frame_before;
    cvtColor(frame_before, frame_before_g, COLOR_BGR2GRAY);

    /*  Shi-Tomasi Corner Detector 
        https://medium.com/pixel-wise/detect-those-corners-aba0f034078b

        Similar to Harris Corner Detector
        1, Weighted sum multiplied by the intensity difference for all pixels in a window,
        using Taylor Series expansion we can obtain E(u,v).
        2, Scoring func:  R = min(lambda_1, lambda_2)
    */
    goodFeaturesToTrack(frame_before_g,
        p0,
        parser.get<int>("@numPts") /* N strongest corners in the image */,
        parser.get<float>("@cornerQuality") /* minimum quality of corner below which everyone is rejected */,
        parser.get<int>("@minDist") /* minimum euclidean distance between corners */,
        Mat() /* Output, vector of conrer qualities */,
        7 /* Size of an average block for computing a derivative covariation matrix over each
pixel neighborhood */,
        false,
        0.04);

    // Create a mask image for drawing purposes
    Mat mask = Mat::zeros(frame_before.size(), frame_before.type());

    // Start time
    double speed = 5; // capture.get(CAP_PROP_FPS);
    time_t start, end;
    time(&start);

    int windowSize = parser.get<int>("@windowSize");

    while (true) {
        Mat frame, frame_g;
        capture >> frame;

        // exit on end
        if (frame.empty())
            break;

        cvtColor(frame, frame_g, COLOR_BGR2GRAY);

        // DOF(frame_g, frame_before_g);

        /*  Calculate Lucas-Kanade Optical Flow 
        */
        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(frame_before_g /*prev img*/,
            frame_g /*next img*/,
            p0 /*prev pts*/,
            p1 /*next pts*/,
            status,
            err,
            Size(windowSize, windowSize) /* window size*/,
            parser.get<int>("@maxLevel") /* maximal pyramid level number */,
            criteria);

        vector<Point2f> good_new;
        for (uint i = 0; i < p0.size(); i++) {
            // Select good points
            if (status[i] == 1) {
                good_new.push_back(p1[i]);
                // draw the tracks
                line(mask, p1[i], p0[i], colors[i], 2);
                circle(frame, p1[i], 5, colors[i], -1);
            }
        }
        Mat img;
        add(frame, mask, img);

        // Time elapsed
        time(&end);
        double seconds = difftime(end, start);

        putText(img, format("Time: %1.1fs", seconds), Point(50, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 170, 50), 1);
        putText(img, format("numPts: %d", parser.get<int>("@numPts")), Point(50, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 170, 50), 1);
        putText(img, format("cornerQuality: %1.2f", parser.get<float>("@cornerQuality")), Point(50, 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 170, 50), 1);
        putText(img, format("minDist: %dpx", parser.get<int>("@minDist")), Point(50, 80), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 170, 50), 1);
        putText(img, format("windowSize: %d", parser.get<int>("@windowSize")), Point(50, 100), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 170, 50), 1);
        putText(img, format("maxLevel: %d", parser.get<int>("@maxLevel")), Point(50, 120), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 170, 50), 1);

        // Display frame.
        imshow("Optical flow", img);

        // Exit if ESC pressed.
        int k = waitKey(speed);
        if (k == 27) {
            break;
        }

        // Now update the previous frame and previous points
        frame_before_g = frame_g.clone();
        p0 = good_new;
    }

    return 0;
}
