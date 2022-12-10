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

Mat DOF(Mat& frame_g, Mat& frame_before_g, float scale, int windowSize, int maxLevel, int iterations)
{
    /*  
        Dense Optical flow
        Two-Frame Motion Estimation Based on Polynomial Expansion" by Gunnar Farneback
    */
    Mat flow(frame_before_g.size(), CV_32FC2);
    calcOpticalFlowFarneback(frame_before_g,
        frame_g,
        flow,
        scale /* Pyramid scale */,
        maxLevel /* number of pyramid layers including the initial image */,
        windowSize /* Window size */,
        iterations /* Iterations per pyramid level */,
        15 /* Poly pixel neighbourhood */,
        1.2 /* Poly sigma */,
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

    return bgr;
}

Mat KLT(vector<Scalar>& colors, Mat& frame, Mat& frame_g, Mat& frame_before_g, Mat& mask, vector<Point2f>& p0, vector<Point2f>& p1, int windowSize, int maxLevel)
{
    /*
        Calculate Lucas-Kanade Optical Flow 
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
        maxLevel /* maximal pyramid level number */,
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
    p0 = good_new;

    Mat img;
    add(frame, mask, img);

    return img;
}

/* src: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html */
int main(int argc, char const* argv[])
{
    // LOAD VIDEO
    cv::String keys = "{@input |../data/KLT_TRACK_r3.mpg|input video}"
                      "{@detectionType |0|choose between `0` for `klt` or `1` for `dof`}"
                      "{@numPts |16|number of pts}"
                      "{@cornerQuality |0.1|minimum quality of corner below which everyone is rejected}"
                      "{@minDist |64|min euclidean distance}"
                      "{@windowSize |64|window size}"
                      "{@maxLevel |3|maximal pyramid level}"
                      "{@iterations |3|Iterations per pyramid level (DOF only)}"
                      "{@scale |0.1|Pyramid scale (DOF only)}";

    cv::CommandLineParser parser(argc, argv, keys);
    String videoName = parser.get<String>("@input");
    std::string image_path = samples::findFile(videoName);
    VideoCapture capture(image_path);

    if (!capture.isOpened()) {
        std::cout << "Error reading video: " << image_path << std::endl;
        return 1;
    }

    // Random colors for tracking
    vector<Scalar> colors;
    RNG rng;
    for (int i = 0; i < parser.get<int>("@numPts"); i++) {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r, g, b));
    }

    Mat frame_before, frame_before_g;

    // Take first frame and find corners in it
    capture >> frame_before;
    cvtColor(frame_before, frame_before_g, COLOR_BGR2GRAY);

    vector<Point2f> p0, p1;
    if (parser.get<int>("@detectionType") == 0) {

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
            7 /* Size of an average block for computing a derivative covariation matrix over each pixel neighborhood */,
            false,
            0.04);
    }

    // Crate mask to draw points on
    Mat mask = Mat::zeros(frame_before.size(), frame_before.type());

    // Start time
    int speed = 1; // capture.get(CAP_PROP_FPS);
    // time_t start, end;
    // time(&start);

    Mat out;
    while (true) {
        Mat frame, frame_g;

        capture >> frame;

        // Exit on video end
        if (frame.empty()) {
            imwrite(videoName + "_s.png", out);
            break;
        }

        cvtColor(frame, frame_g, COLOR_BGR2GRAY);

        // Time elapsed
        // time(&end);
        // double seconds = difftime(end, start);

        if (parser.get<int>("@detectionType") == 0) {

            /* KLT OPTICAL FLOW */
            out = KLT(colors, frame, frame_g, frame_before_g, mask, p0, p1, parser.get<int>("@windowSize"), parser.get<int>("@maxLevel"));

            // putText(klt, format("Time: %1.1fs", seconds), Point(50, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 170, 50), 1);
            putText(out, format("numPts: %d", parser.get<int>("@numPts")), Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 170, 50), 1);
            putText(out, format("cornerQuality: %1.2f", parser.get<float>("@cornerQuality")), Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 170, 50), 1);
            putText(out, format("minDist: %dpx", parser.get<int>("@minDist")), Point(20, 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 170, 50), 1);
            putText(out, format("windowSize: %d", parser.get<int>("@windowSize")), Point(20, 80), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 170, 50), 1);
            putText(out, format("maxLevel: %d", parser.get<int>("@maxLevel")), Point(20, 100), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 170, 50), 1);
            // Display frames
            imshow("Optical flow", out);
        }
        else if (parser.get<int>("@detectionType") == 1) {
            /* DENSE OPTICAL FLOW */
            out = DOF(frame_g, frame_before_g, parser.get<float>("@scale"), parser.get<int>("@windowSize"), parser.get<int>("@maxLevel"), parser.get<int>("@iterations"));

            putText(out, format("scale: %1.1f", parser.get<float>("@scale")), Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 170, 50), 1);
            putText(out, format("windowSize: %d", parser.get<int>("@windowSize")), Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 170, 50), 1);
            putText(out, format("maxLevel: %d", parser.get<int>("@maxLevel")), Point(20, 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 170, 50), 1);
            putText(out, format("iterations: %d", parser.get<int>("@iterations")), Point(20, 80), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 170, 50), 1);
            // // Display frames
            imshow("Dense optical flow", out);
        }

        // EWxit on ESC
        int k = waitKey(speed);
        if (k == 27) {
            break;
        }
        if (k == 's') {
            imwrite(videoName + "_s.png", out);
            break;
        }

        // Update the frame
        frame_before_g = frame_g.clone();
    }

    return 0;
}
