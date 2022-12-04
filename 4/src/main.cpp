#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <time.h>

#include "helper.h"

using namespace cv;
using namespace std;

int main(int argc, char const* argv[])
{
    // LOAD IMAGE
    cv::String keys = "{@input |../data/halak1.mpg|input video}";

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
    for (int i = 0; i < 100; i++) {
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
    goodFeaturesToTrack(frame_before_g, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);

    // Create a mask image for drawing purposes
    Mat mask = Mat::zeros(frame_before.size(), frame_before.type());

    // Start time
    double fps = capture.get(CAP_PROP_FPS);
    time_t start, end;
    time(&start);
    while (true) {
        Mat frame, frame_g;
        capture >> frame;

        // exit on end
        if (frame.empty())
            break;

        cvtColor(frame, frame_g, COLOR_BGR2GRAY);

        // calculate optical flow
        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(frame_before_g, frame_g, p0, p1, status, err, Size(15, 15), 2, criteria);

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

        putText(img, format("FPS: %1.1f", fps), Point(50, 30), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
        putText(img, format("Time: %1.1fs", seconds), Point(50, 70), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

        // Display frame.
        imshow("Optical flow", img);

        // Exit if ESC pressed.
        int k = waitKey(fps);
        if (k == 27) {
            break;
        }

        // Now update the previous frame and previous points
        frame_before_g = frame_g.clone();
        p0 = good_new;
    }

    return 0;
}
