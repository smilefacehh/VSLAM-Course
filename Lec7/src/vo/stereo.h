#ifndef STEREO_H_
#define STEREO_H_

// 头文件顺序：pcl, opencv
#include <eigen3/Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>


/**
 * Stereo双目
 * 
 * 1.输入已经校正之后的双目图像，查找匹配点，输出视差图，并行滑窗匹配，SSD
 * 2.剔除外点
 * 3.三角化计算3D点
 * 4.利用外参，累积点云，展示
*/

namespace mango
{
class Stereo
{
public:
    Stereo();
    ~Stereo();

    /**
     * 对齐的双目图像计算视差
     * @param left_img      左图
     * @param right_img     右图
     * @param patch_radius  半径，2r+1
     * @param min_disp      最小视差
     * @param max_disp      最大视差
    */
    cv::Mat match(const cv::Mat& left_img, const cv::Mat& right_img, float patch_radius, float min_disp, float max_disp);
    
    /**
     * 视差图转换成点云
     * @param disparity 视差图
     * @param img       原始左图
     * @param depth     深度图
     * @param K         相机内参
     * @param baseline  基线
    */
    pcl::PointCloud<pcl::PointXYZRGB> disparity2pointcloud(const cv::Mat& disparity, const cv::Mat& img, cv::Mat& depth, const Eigen::Matrix3d& K, double baseline);
};
}
#endif