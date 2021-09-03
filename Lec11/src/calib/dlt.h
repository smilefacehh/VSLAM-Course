#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/Eigen>

#include "../base/pose.hpp"

namespace mango {

class DLT
{
public:
    DLT(const Eigen::Matrix<double, Eigen::Dynamic, 3>& world_pts, const Eigen::Matrix<double, Eigen::Dynamic, 2>& pxs);
    DLT(const Eigen::Matrix<double, Eigen::Dynamic, 3>& world_pts, const Eigen::Matrix<double, Eigen::Dynamic, 2>& pxs, const Eigen::Matrix3d& camera_K);
    ~DLT();

    /**
     * 执行接口
    */
    void run();

    /**
     * 计算pose
     * @param world_pts 世界点集合
     * @param pxs       投影像素点集合
    */
    inline Pose3D getPose()
    {
        return pose_;
    }

    /**
     * 计算内参
    */
    inline Eigen::Matrix3d getCameraK()
    {
        return camera_K_;
    }

private:
    /**
     * 构建矩阵Q
    */
    Eigen::Matrix<double, Eigen::Dynamic, 12> makeQ();

    /**
     * 计算矩阵M
    */
    Eigen::Matrix<double, 3, 4> getM(const Eigen::Matrix<double, Eigen::Dynamic, 12> Q);
    
    /**
     * M分解得到R、t
    */
    Pose3D poseFromM(const Eigen::Matrix<double, 3, 4> M);

    /**
     * M分解得到R、t、K
    */
    void decompM(const Eigen::Matrix<double, 3, 4> M, Pose3D& pose, Eigen::Matrix3d& K);


    Eigen::Matrix<double, Eigen::Dynamic, 3> world_pts_;    // 世界点
    Eigen::Matrix<double, Eigen::Dynamic, 2> pxs_;          // 投影像素点
    Eigen::Matrix3d camera_K_;                              // 相机内参
    Pose3D pose_;                                           // 外参R、t
    int pt_num_;                                            // 点数
};

}