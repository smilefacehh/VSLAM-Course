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

/**
 * 绘制特征点
 * @param img   图像
 * @param kps   特征点
 * @param color 颜色
*/
cv::Mat drawKeyPoint(const cv::Mat& img, const std::vector<cv::KeyPoint>& kps, const cv::Scalar& color);

template <class PointType>
cv::Mat drawPoint(const cv::Mat& img, const std::vector<PointType>& pxs, DrawType type, const cv::Scalar& color)
{
    cv::Mat img_result(img.rows, img.cols, CV_8UC3);
    if(img.channels() == 1)
    {
        std::vector<cv::Mat> channels(3, img);
        cv::merge(channels, img_result);
    }
    else
    {
        img.copyTo(img_result);
    }
    for(int i = 0; i < pxs.size(); i++)
    {
        double x = pxs[i].x, y = pxs[i].y;
        if(x < 0 || x >= img.cols || y < 0 || y >= img.rows)
        {
            continue;
        }

        if(type == DrawType::CIRCLE)
        {
            cv::circle(img_result, cv::Point(x, y), 4, color);
        }
        else if(type == DrawType::POINT)
        {
            cv::circle(img_result, cv::Point(x, y), 2, color, -1);
        }
        else if(type == DrawType::X)
        {
            double x_l = x - 3, x_r = x + 3, y_l = y - 3, y_r = y + 3;
            if(x_l < 0) x_l = 0;
            if(x_r >= img.cols) x_r = img.cols;
            if(y_l < 0) y_l = 0;
            if(y_l >= img.rows) y_l = img.rows;

            cv::line(img_result, cv::Point(x_l, y_l), cv::Point(x_r, y_r), color);
            cv::line(img_result, cv::Point(x_l, y_r), cv::Point(x_r, y_l), color);
        }
        else if(type == DrawType::RECT)
        {
            double x_l = x - 5, x_r = x + 5, y_l = y - 5, y_r = y + 5;
            if(x_l < 0) x_l = 0;
            if(x_r >= img.cols) x_r = img.cols;
            if(y_l < 0) y_l = 0;
            if(y_l >= img.rows) y_l = img.rows;

            cv::rectangle(img_result, cv::Rect(x_l, y_l, x_r - x_l, y_r - y_l), color, 2);
        }
    }

    return img_result;
}
}