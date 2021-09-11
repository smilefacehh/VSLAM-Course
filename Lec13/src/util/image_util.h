#pragma once

#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>

namespace mango
{
/**
 * 合并图像
 * @param imgs 图像
 * @param w    图像缩放为指定大小，宽
 * @param h    图像缩放为指定大小，高
*/
cv::Mat mergeImage(const std::vector<cv::Mat>& imgs, int w, int h);

/**
 * 灰度图映射成彩色图，显示用
 * @param img 灰度图像
*/
cv::Mat gray2color(const cv::Mat& img);

/**
 * 图像仿射变换，旋转、平移、缩放
*/
cv::Mat warpImage(const cv::Mat& img, const Eigen::Matrix<float, 2, 3>& warpTransform);

/**
 * (2,3)仿射变换矩阵，近似
*/
Eigen::Matrix<float, 2, 3> getWarp(float dx, float dy, float alpha, float lambda);

/**
 * 取点（x,y）处仿射变换后的patch
*/
cv::Mat getWarpedPatch(const cv::Mat& img, const Eigen::Matrix<float, 2, 3>& warpTransform, float x, float y, float patch_radius);

}