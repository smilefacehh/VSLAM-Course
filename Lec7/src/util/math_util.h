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
double ssd(const cv::Mat& p1, const cv::Mat& p2);

/**
 * 多项式拟合
 * @param x x
 * @param y y
 * @param o 多项式阶数
*/
cv::Mat polyfit(const std::vector<double>& x, const std::vector<double>& y, int o);

}