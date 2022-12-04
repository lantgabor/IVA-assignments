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

    double fps = capture.get(CAP_PROP_FPS);

    Mat frame;

    // Start time
    time_t start, end;
    time(&start);
    while (capture.read(frame)) {
        time(&end);
        // Time elapsed
        double seconds = difftime(end, start);

        putText(frame, format("FPS: %1.1f", fps), Point(50, 30), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
        putText(frame, format("Time: %1.1fs", seconds), Point(50, 70), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
        // Display frame.
        imshow("Tracking", frame);
        if (frame.empty())
            break;
        // Exit if ESC pressed.
        int k = waitKey(fps);
        if (k == 27) {
            break;
        }
    }

    return 0;
}
