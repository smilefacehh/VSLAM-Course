#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

namespace mango {

/**
 * 计算SSD误差
 * 注：模板函数的实现要放到头文件中
*/
template <typename T>
double ssd(const std::vector<T>& v1, const std::vector<T>& v2)
{
    assert(v1.size() == v2.size());

    double sum = 0;

    for(int i = 0; i < v1.size(); i++)
    {
        double d = v1[i] - v2[i];
        sum += d * d;
    }

    return sum;
}

/**
 * Sum of Squared Differences
*/
template <typename T>
double ssd(const cv::Mat& p1, const cv::Mat& p2)
{
    double sum = 0;
    for(int r = 0; r < p1.rows; r++)
    {
        for(int c = 0; c < p1.cols; c++)
        {
            // 这里类型一定不能错
            double d = p1.at<T>(r, c) - p2.at<T>(r, c);
            sum += d * d;
        }
    }
    return sum;
}

/**
 * 多项式拟合
 * @param x x
 * @param y y
 * @param o 多项式阶数
*/
cv::Mat polyfit(const std::vector<double>& x, const std::vector<double>& y, int o);

}