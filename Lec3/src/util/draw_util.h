#pragma once

#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <eigen3/Eigen/Core>

#include "../base/pose.hpp"
#include "../base/camera.h"
#include "../base/point.hpp"

namespace mango {

enum class DrawType : u_int8_t
{
    NONETYPE,
    CIRCLE,
    POINT,
    X,
    RECT
};

/**
 * 绘制相机位姿，坐标轴形式
 * @param img    图像
 * @param pose   相机位姿
 * @param camera 相机内参
*/
cv::Mat drawPose(const cv::Mat& img, const Pose3D& pose, const CameraPtr& camera);

/**
 * 绘制点，类型包括：圆圈、实心点、叉叉、正方形
 * @param img  图像
 * @param pxs  像素点
 * @param type 类型
*/
cv::Mat drawPoint(const cv::Mat& img, const Eigen::Matrix<double, Eigen::Dynamic, 2>& pxs, DrawType type, const cv::Scalar& color);

cv::Mat drawPoint(const cv::Mat& img, const std::vector<mango::Point2D>& pxs, DrawType type, const cv::Scalar& color);
}