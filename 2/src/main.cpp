#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main(int argc, char const* argv[])
{
    // LOAD IMAGE
    cv::CommandLineParser parser(argc, argv, "{@input |../data/phobos.png|input image}");
    String imageName = parser.get<String>("@input");
    std::string image_path
        = samples::findFile(imageName);
    Mat img = imread(image_path, 0);
    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    imshow("original", img);

    int k = waitKey(0); // Wait for a keystroke in the window

    return 0;
}
