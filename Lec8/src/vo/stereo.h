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

    /**
     * 线性三角化
     * @param p1 (3,N) 图像一中的像素点齐次坐标
     * @param p2 (3,N) 图像二中的像素点齐次坐标
     * @param M1 (3,4) 图像一的投影矩阵，M=K[R|t]
     * @param M2 (3,4) 图像二的投影矩阵，M=K[R|t]
     * @return   (4,N) 3D点的齐次坐标
    */
    Eigen::Matrix<float, 4, Eigen::Dynamic> linearTriangulation(const Eigen::Matrix<float, 3, Eigen::Dynamic>& p1,
                                                                const Eigen::Matrix<float, 3, Eigen::Dynamic>& p2,
                                                                const Eigen::Matrix<float, 3, 4>& M1,
                                                                const Eigen::Matrix<float, 3, 4>& M2);


    /**
     * 八点法计算基础矩阵
     * @param p1 (3,N) 图像一中的像素点齐次坐标
     * @param p2 (3,N) 图像二中的像素点齐次坐标
     * @return   (3,3) 基础矩阵
    */
    Eigen::Matrix3f fundamentalEightPoint(const Eigen::Matrix<float, 3, Eigen::Dynamic>& p1, const Eigen::Matrix<float, 3, Eigen::Dynamic>& p2);


    /**
     * 归一化八点法计算基础矩阵
     * @param p1 (3,N) 图像一中的像素点齐次坐标
     * @param p2 (3,N) 图像二中的像素点齐次坐标
     * @return   (3,3) 基础矩阵
    */
    Eigen::Matrix3f fundamentalEightPointNormalized(const Eigen::Matrix<float, 3, Eigen::Dynamic>& p1, const Eigen::Matrix<float, 3, Eigen::Dynamic>& p2);

    /**
     * 像素点归一化
     * @param pt           (3,N) 像素坐标
     * @param pt_normalized (3,N) 归一化像素坐标
     * @param transform    (3,3) 变换矩阵
    */
    void normalizePoint(const Eigen::Matrix<float, 3, Eigen::Dynamic>& pt, Eigen::Matrix<float, 3, Eigen::Dynamic>& pt_normalized, Eigen::Matrix3f& transform);

    /**
     * 计算本质矩阵
     * @param p1 (3,N) 图像一中的像素点齐次坐标
     * @param p2 (3,N) 图像二中的像素点齐次坐标
     * @param K1 (3,3) 相机内参一
     * @param K2 (3,3) 相机内参二
     * @return   (3,3) 本质矩阵
    */
    Eigen::Matrix3f estimateEssentialMatrix(const Eigen::Matrix<float, 3, Eigen::Dynamic>& p1,
                                            const Eigen::Matrix<float, 3, Eigen::Dynamic>& p2,
                                            const Eigen::Matrix3f& K1,
                                            const Eigen::Matrix3f& K2);

    /**
     * 点到极线距离之和
     * @param F  (3,3) 基础矩阵
     * @param p1 (3,N) 图像一中的像素点齐次坐标
     * @param p2 (3,N) 图像二中的像素点齐次坐标
     * @return         距离之和
    */
    double distPoint2EpipolarLine(const Eigen::Matrix3f& F, const Eigen::Matrix<float, 3, Eigen::Dynamic>& p1, const Eigen::Matrix<float, 3, Eigen::Dynamic>& p2);

    /**
     * 本质矩阵分解
     * @param rotation (6,3) 两个(3,3)的旋转矩阵
     * @param u3       (3,1) 平移
    */
    void decomposeEssentialMatrix(const Eigen::Matrix3f& E, Eigen::Matrix<float, 6, 3>& rotation, Eigen::Vector3f& u3);

    /**
     * 通过三角化选择正确的位姿结果
     * @param rotation  (6,3) 两个(3,3)的旋转矩阵
     * @param u3        (3,1) 平移
     * @param p1        (3,N) 图像一中的像素点齐次坐标
     * @param p2        (3,N) 图像二中的像素点齐次坐标
     * @param K1        (3,3) 相机内参一
     * @param K2        (3,3) 相机内参二
     * @return          (3,4) [R|t]
    */
    Eigen::Matrix<float, 3, 4> disambiguateRelativePose(const Eigen::Matrix<float, 6, 3>& rotation, 
                                                        const Eigen::Vector3f& u3,
                                                        const Eigen::Matrix<float, 3, Eigen::Dynamic>& p1,
                                                        const Eigen::Matrix<float, 3, Eigen::Dynamic>& p2,
                                                        const Eigen::Matrix3f& K1,
                                                        const Eigen::Matrix3f& K2);
};
}
#endif