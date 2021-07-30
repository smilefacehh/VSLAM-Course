#include "math_util.h"
#include <assert.h>

namespace mango {
double ssd(const cv::Mat& p1, const cv::Mat& p2)
{
    assert(p1.size() == p2.size());

    double sum = 0;
    
    for(int r = 0; r < p1.rows; r++)
    {
        for(int c = 0; c < p1.cols; c++)
        {
            double d = p1.at<float>(r, c) - p2.at<float>(r, c);
            sum += d * d;
        }
    }
    return sum;
}
}