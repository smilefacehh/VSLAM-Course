#pragma once

#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>

namespace mango
{
class KanadeLucasTomasi
{
public:
    KanadeLucasTomasi();
    ~KanadeLucasTomasi();

    /**
     * 在第2帧图像中跟踪第一帧图像点(x,y)，通过取patch的方式，迭代优化仿射变换矩阵参数，得到最终的仿射变换参数
    */
    Eigen::Matrix<float, 2, 3> trackPointKLT(const cv::Mat& img_ref, const cv::Mat& img_query, float x, float y, float patch_radius, int num_iters);
};
}