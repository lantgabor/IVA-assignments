#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;

Mat convolve(Mat X, Mat Y)
{
    int H = floor((Y.rows - 1) / 2);
    int W = floor((Y.cols - 1) / 2);

    Mat out = Mat::zeros(X.rows, X.cols, CV_32F);

    // over the input image
    for (int i = H; i < X.rows - H; i++) {
        for (int j = W; j < X.cols - W; j++) {

            // over the filter
            float sum = 0;

            for (int k = -H; k < H + 1; k++) {
                for (int l = -W; l < W + 1; l++) {
                    float a = X.at<float>(i + k, j + l);
                    float b = Y.at<float>(H + k, W + l);
                    sum += (a * b);
                }
            }
            out.at<float>(i, j) = sum / 12;
        }
    }

    return out;
}

int main(int argc, char** argv)
{
    // LOAD IMAGES
    std::string image_path = samples::findFile("../data/lidi2.jpeg");
    Mat img = imread(image_path);
    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    imshow("original", img);

    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);

    Mat img_blur;
    GaussianBlur(img_gray, img_blur, Size(3, 3), 0);

    // Mat sobelx, sobely, sobelxy;
    // Sobel(img_blur, sobelx, CV_64F, 1, 0, 5);
    // Sobel(img_blur, sobely, CV_64F, 0, 1, 5);
    // Sobel(img_blur, sobelxy, CV_64F, 1, 1, 5);
    // imshow("Sobel XY using Sobel() function", sobelxy);
    // imshow("Sobel Y", sobely);
    // imshow("Sobel X", sobelx);

    // Mat edges;
    // Canny(img_blur, edges, 100, 200, 3, false);
    // imshow("Canny edge detection", edges);

    // Prewitt horizontal
    float p_kernelY[9] = { 1, 1, 1, 0, 0, 0, -1, -1, -1 };
    cv::Mat prewittY = cv::Mat(3, 3, CV_32F, p_kernelY);
    std::cout << prewittY << std::endl;

    // apply filter
    cv::Mat My;
    // My = convolve(img_blur, prewitt);
    cv::filter2D(img_blur, My, img_blur.depth(), prewittY);
    imshow("My", My);

    // Prewitt vertical
    float p_kernelX[9] = { 1, 0, -1, 1, 0, -1, 1, 0, -1 };
    cv::Mat prewittX = cv::Mat(3, 3, CV_32F, p_kernelX);
    std::cout << prewittX << std::endl;

    // apply filter
    cv::Mat Mx;
    cv::filter2D(img_blur, Mx, img_blur.depth(), prewittX);
    imshow("Mx", Mx);

    My.convertTo(My, CV_32F);
    Mx.convertTo(Mx, CV_32F);

    cv::Mat prewitt_out(Mx.size(), CV_32F);

    cv::magnitude(Mx, My, prewitt_out);

    imshow("prewitt_out", prewitt_out / 255);

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