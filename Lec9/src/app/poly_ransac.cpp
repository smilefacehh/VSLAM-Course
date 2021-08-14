#include <iostream>
#include <vector>

#include <eigen3/Eigen/Core>

#include "util/math_util.h"


/**
 * 二次曲线ransac拟合
 * @param pt                (2,N) 数据点
 * @param err_th            inlier阈值
 * @param guess_iters       (3,M) 每次迭代的估计结果，二次曲线的参数
 * @param num_inlier_iters  (M,1) 每次迭代的inlier数量 
*/
void ransac(const Eigen::Matrix2Xf& pt, const float& err_th, Eigen::MatrixXf& guess_iters, Eigen::MatrixXf& num_inlier_iters)
{
    int num_iterations = 10;
    int pt_N = pt.cols();

    guess_iters.resize(3, num_iterations);
    num_inlier_iters.resize(num_iterations, 1);

    int max_num_inliers = 0;
    Eigen::VectorXf best_guess = Eigen::Vector3f::Zero();

    // 迭代
    for(int i = 0; i < num_iterations; i++)
    {
        // 随机3对点
        std::vector<int> rnd_index = mango::randomN(pt_N, 3);
        Eigen::Matrix<float, 2, 3> sample_pt;
        for(int j = 0; j < rnd_index.size(); j++)
        {
            sample_pt(0, j) = pt(0, rnd_index[j]);
            sample_pt(1, j) = pt(1, rnd_index[j]);
        }

        // 拟合多项式曲线，并计算inlier
        Eigen::VectorXf poly_param = mango::polyfit(sample_pt, 2);
        Eigen::VectorXf poly_y = mango::polyVal(pt.row(0).transpose(), poly_param);
        Eigen::VectorXf error_y = poly_y - pt.row(1).transpose();
        Eigen::Matrix2Xf inliers;
        for(int j = 0; j < error_y.rows(); j++)
        {
            if(error_y(j) <= err_th + 1e-5)
            {
                // 动态添加元素到Eigen::Matrix
                inliers.conservativeResize(inliers.rows(), inliers.cols() + 1);
                inliers.col(inliers.cols() - 1) = pt.col(j);
            }
        }

        // 用inlier重新估计参数
        int num_inliers = inliers.cols();
        if(num_inliers > max_num_inliers)
        {
            best_guess = mango::polyfit(inliers, 2);
            max_num_inliers = num_inliers;
        }

        // 添加到历史数据中
        guess_iters.col(i) = best_guess;
        num_inlier_iters(i) = max_num_inliers;
    }
}

int main(int argc, char** argv)
{
    // y = 2x^2 + 3x - 1
    Eigen::Matrix<float, 2, 10> pt;
    // pt << -4, -3, -2, -1,  0, 1,  2,  3,  4,  5,
    //       19,  8,  1, -2, -1, 4, 13, 26, 43, 64;
    pt << -4, -3, -2, -1,  0, 1,  2,  3,  4,  5,
          17,  8,  3, -4, -2, 4, 14, 26, 42, 63;
    Eigen::MatrixXf guess_iters;
    Eigen::MatrixXf num_inlier_iters;
    // 这个误差阈值在这里很重要，如果不合适，拟合的结果不太对
    ransac(pt, 0.2, guess_iters, num_inlier_iters);
    
    std::cout << guess_iters << std::endl;
    std::cout << num_inlier_iters.transpose() << std::endl;
          
    return 0;
}