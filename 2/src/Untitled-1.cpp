// Requires following includes
#include <vector>
#include <cmath>
#include <utility>
#include <Magick++.h>

/**
* Calculate the otsu threshold of given image.
*
* This function will automatically convert the input image to to grayscale when
* it is not provided as such.
*
* Returns the calculated threshold between 0 and QuantumRange
*
* Based on paper "A C++ Implementation of Otsu’s Image Segmentation Method"
* by Juan Pablo Balarini, Sergio Nesmachnow (2016)
* http://www.ipol.im/pub/art/2016/158/
*/
int otsuThreshold(const Magick::Image& image)
{
    int n = QuantumRange;
    long totalPixels = image.columns() * image.rows();

    // Prepare histogram image
    Magick::Image histImage(image);
    if (histImage.colorSpace() != Magick::GRAYColorspace) {
        histImage.colorSpace(Magick::GRAYColorspace);
    }

    // Find number of occurrences per color
    // The histogram should stretch all bins
    std::vector<long> histogram(n + 1, 0);

    Magick::Pixels view(histImage);
    const Magick::PixelPacket* px = view.getConst(0, 0, image.columns(), image.rows());

    for (long i = 0; i < totalPixels; i++) {
        histogram[std::round(px->red)]++;
        *px++;
    }

    // Compute threshold
    // Init variables
    int threshold = 0;
    double sum = 0;
    double sumB = 0;
    long q1 = 0;
    long q2 = 0;
    double varMax = 0;

    // Auxiliary value for computing m2
    for (int i = 0; i <= n; i++) {
        sum += (double)i * histogram[i];
    }

    for (int i = 0; i <= n; i++) {
        // Update q1
        q1 += histogram[i];
        if (q1 == 0) {
            continue;
        }

        // Update q2
        q2 = totalPixels - q1;
        if (q2 == 0) {
            break;
        }

        // Update m1 and m2
        sumB += (double)i * histogram[i];
        double m1 = sumB / q1;
        double m2 = (sum - sumB) / q2;

        // Update the between class variance
        double varBetween = (double)q1 * (double)q2 * (m1 - m2) * (m1 - m2);

        // Update the threshold if necessary
        if (varBetween > varMax) {
            varMax = varBetween;
            threshold = i;
        }
    }

    return threshold;
}