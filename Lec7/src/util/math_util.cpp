#include "math_util.h"
#include <assert.h>

namespace mango {

cv::Mat polyfit(const std::vector<double>& x, const std::vector<double>& y, int o)
{
	int n = x.size();
	// 参数个数
	int p = o + 1;

    cv::Mat U(n, p, CV_64F);
    cv::Mat Y(n, 1, CV_64F);

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < p; j++)
        {
            U.at<double>(i, j) = pow(x[i], j);
        }

        Y.at<double>(i, 0) = y[i];
    }

    cv::Mat K(p, 1, CV_64F);
    K = (U.t() * U).inv() * U.t() * Y;

    return K;
}
}