#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>

#include "../base/point.hpp"
#include "../base/camera.h"
#include "../base/pose.hpp"

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
 * 计算重投影误差
 * @param pxs    像素点
 * @param pts    世界点
 * @param camera 相机内参
 * @param pose   相机pose
*/
double reprojectionError(const std::vector<mango::Point2D>& pxs, const std::vector<mango::Point3D>& pts, const mango::CameraPtr& camera, const mango::Pose3D& pose);

}