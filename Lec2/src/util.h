#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "point.hpp"
#include "camera.h"
#include "pose.hpp"

namespace mango {

/**
 * 双线性插值
 * 四个顶点，左上角值val1；插值点距离上边界距离为d1；均顺时针排序
*/
template<typename TVal, typename TDist>
TVal bilinear(const TVal& val1, const TVal& val2, const TVal& val3, const TVal& val4, const TDist& d1, const TDist& d2, const TDist& d3, const TDist& d4);

/**
 * 施加畸变，注意是在归一化相机坐标系下进行的
 * @param point  原始点坐标，归一化相机平面坐标，z归一化为1
 * @param camera 相机
 * @return       畸变点坐标，归一化相机平面坐标，z归一化为1
*/
mango::Point2D distort(const mango::Point2D& point, const mango::CameraPtr& camera);

/**
 * 图像畸变矫正
 * @param img    畸变图像
 * @param camera 相机
 * @return       返回校正后的图像
*/
cv::Mat undistortImage(const cv::Mat& img, const mango::CameraPtr& camera);


/**
 * 相机点投影到像素坐标系
 * @param pt_camera 相机坐标点
 * @param camera    相机参数
*/
mango::Point2D project(const mango::Point3D& pt_camera, const mango::CameraPtr& camera);

/**
 * 世界点投影到像素坐标系
 * @param pt_world 世界坐标点
 * @param camera   相机参数
 * @param pose     相机位姿
*/
mango::Point2D project(const mango::Point3D& pt_world, const mango::CameraPtr& camera, const mango::Pose3D& pose);

/**
 * 像素点反投影到相机坐标系
 * @param p      像素点
 * @param camera 相机参数
*/
mango::Point3D unproject(const mango::Point2D& p, const mango::CameraPtr& camera);

/**
 * 像素点反投影到世界坐标系
 * @param p      像素点
 * @param camera 相机参数
 * @param pose   相机位姿
*/
mango::Point3D unproject(const mango::Point2D& p, const mango::CameraPtr& camera, const mango::Pose3D& pose);

/**
 * 位姿变换
 * @param pt   参考点
 * @param pose 施加变换
*/
mango::Point3D transform(const mango::Point3D& pt, const mango::Pose3D& pose);

/**
 * 合并图像
 * @param imgs 图像
 * @param w    图像缩放为指定大小，宽
 * @param h    图像缩放为指定大小，高
*/
cv::Mat mergeImage(const std::vector<cv::Mat>& imgs, int w, int h);
}