#pragma once

#include <vector>

#include <Eigen/Eigen>

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
 * @return  (n,1) 多项式系数，顺序为c b a
*/
cv::Mat polyfit(const std::vector<double>& x, const std::vector<double>& y, int o);

/**
 * 多项式拟合
 * @param pt 2D点
 * @param o  多项式阶数
 * @return  (n,1) 多项式系数，顺序为c b a
*/
Eigen::VectorXf polyfit(const Eigen::Matrix2Xf& pt, int o);

/**
 * 多项式给定系数、x计算y
 * @param x          x
 * @param poly_param 多项式系数
 * @return           y
*/
Eigen::VectorXf polyVal(const Eigen::VectorXf& x, const Eigen::VectorXf& poly_param);

/**
 * 反对称矩阵3x3
 * @param vec (3,1) 向量
 * @return    (3,3) 反对称矩阵
*/
template <typename T>
Eigen::Matrix<T, 3, 3> skewSymmetricMatrix3(const Eigen::Matrix<T, 3, 1>& vec)
{
    Eigen::Matrix<T, 3, 3> skew_symmetric_matrix;
    skew_symmetric_matrix << 0, -vec(2,0), vec(1,0),
                             vec(2,0), 0, -vec(0,0),
                             -vec(1,0), vec(0,0), 0;
    
    return skew_symmetric_matrix;
}

/**
 * 向量克罗内克积
 * @param vec1 (m,1) 向量一
 * @param vec1 (n,1) 向量二
 * @return     (mn,1) 结果向量
*/
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> kronecker(const Eigen::Matrix<T, Eigen::Dynamic, 1>& vec1, const Eigen::Matrix<T, Eigen::Dynamic, 1>& vec2)
{
    int m = vec1.rows(), n = vec2.rows();

    Eigen::Matrix<T, Eigen::Dynamic, 1> kronecker_prod;
    kronecker_prod.resize(m * n, 1);

    int idx = 0;
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            kronecker_prod(idx++) = vec1(i) * vec2(j); 
        }
    }
    
    return kronecker_prod;
}

/**
 * 矩阵克罗内克积
*/
cv::Mat kronecker(const cv::Mat& A, const cv::Mat& B);

/**
 * 从[0,n)连续整数中随机取k个不同的数字
*/
std::vector<int> randomN(int n, int k);
}