#include "math_util.h"

#include <assert.h>
#include <random>
#include <ctime>
#include <set>

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

/**
 * 多项式拟合
 * @param pt 2D点
 * @param o  多项式阶数
 * @return  (n,1) 多项式系数，顺序为c b a
*/
Eigen::VectorXf polyfit(const Eigen::Matrix2Xf& pt, int o)
{
    int n = pt.cols();
	// 参数个数
	int p = o + 1;

    Eigen::MatrixXf U, Y;
    U.resize(n, p);
    Y.resize(n, 1);

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < p; j++)
        {
            U(i, j) = pow(pt(0, i), j);
        }

        Y(i) = pt(1, i);
    }

    Eigen::VectorXf K;
    K = (U.transpose() * U).inverse() * U.transpose() * Y;

    return K;
}

/**
 * 多项式给定系数、x计算y
 * @param x          x
 * @param poly_param 多项式系数
 * @return           y
*/
Eigen::VectorXf polyVal(const Eigen::VectorXf& x, const Eigen::VectorXf& poly_param)
{
    int pt_N = x.rows();
    Eigen::VectorXf y;
    y.resize(pt_N);

    for(int i = 0; i < pt_N; i++)
    {
        y(i) = poly_param(2) * x(i) * x(i) + poly_param(1) * x(i) + poly_param(0);
    }
     
    return y;
}

/**
 * 从[0,n)连续整数中随机取k个不同的数字
*/
std::vector<int> randomN(int n, int k)
{
    assert(k <= n);

    std::default_random_engine e(time(0));
    std::uniform_int_distribution<int> u(0, n);

    std::vector<int> num;
    std::set<int> num_set;
    for(int i = 0; i < k; i++)
    {
        int r = u(e);
        while(num_set.find(r) != num_set.end())
        {
            r = (r + 1) % n;
        }
        num_set.insert(r);
        num.push_back(r);
    }

    return num;
}
}