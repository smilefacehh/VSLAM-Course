#include "util.h"
#include <eigen3/Eigen/Core>

namespace mango
{
/**
 * 双线性插值
 * 四个顶点，左上角值val1；插值点距离上边界距离为d1；均顺时针排序
*/
template<typename TVal, typename TDist>
TVal bilinear(const TVal& val1, const TVal& val2, const TVal& val3, const TVal& val4, const TDist& d1, const TDist& d2, const TDist& d3, const TDist& d4)
{
    return val1 * d2 * d3 + val2 * d3 * d4 + val3 * d1 * d4 + val4 * d1 * d2;
}

/**
 * 施加畸变，注意是在归一化相机坐标系下进行的
 * @param point  原始点坐标，归一化相机平面坐标，z归一化为1
 * @param camera 相机
 * @return       畸变点坐标，归一化相机平面坐标，z归一化为1
*/
mango::Point2D distort(const mango::Point2D& point, const mango::CameraPtr& camera)
{
    double u = point.x, v = point.y;
    double r2 = (u - camera->cx) * (u - camera->cx) + (v - camera->cy) * (v - camera->cy);
    double r4 = r2 * r2;
    double coef = 1 + camera->k1 * r2 + camera->k2 * r4;
    double ud = coef * (u - camera->cx) + camera->cx;
    double vd = coef * (v - camera->cy) + camera->cy;
    return mango::Point2D(ud, vd);
}

/**
 * 图像畸变矫正
 * @param img    畸变图像
 * @param camera 相机
 * @return       返回校正后的图像
*/
cv::Mat undistortImage(const cv::Mat& img, const mango::CameraPtr& camera)
{
    // std::cout << camera << std::endl;
    int w = img.cols, h = img.rows;
    cv::Mat undistort_img(h, w, CV_8UC1);

    for(int i = 0; i < w; i++)
    {
        for(int j = 0; j < h; j++)
        {
            // 注意，归一化平面点坐标，水平轴u，竖直轴v；参数ij顺序
            mango::Point2D p = distort(mango::Point2D(i, j), camera);
            int u = p.x, v = p.y;
            if(u >= 0 && v >= 0 && u < w && v < h)
            {
                uchar px = img.at<uchar>(v, u);
                if(u + 1 < w && v + 1 < h)
                {
                    px = bilinear<uchar, double>(img.at<uchar>(v, u), 
                                                img.at<uchar>(v, u + 1), 
                                                img.at<uchar>(v + 1, u), 
                                                img.at<uchar>(v + 1, u + 1), 
                                                p.y - v,
                                                1 - (p.x - u),
                                                1 - (p.y - v),
                                                p.x - u);
                }
                undistort_img.at<uchar>(j, i) = px;
            }
        }
    }
    return undistort_img;
}

/**
 * 相机点投影到像素坐标系
 * @param pt_camera 相机坐标点
 * @param camera    相机参数
*/
mango::Point2D project(const mango::Point3D& pt_camera, const mango::CameraPtr& camera)
{
    mango::Point2D p;
    p.x = camera->fx * pt_camera.x / pt_camera.z + camera->cx;
    p.y = camera->fy * pt_camera.y / pt_camera.z + camera->cy;
    return p;
}

/**
 * 世界点投影到像素坐标系
 * @param pt_world 世界坐标点
 * @param camera   相机参数
 * @param pose     相机位姿
*/
mango::Point2D project(const mango::Point3D& pt_world, const mango::CameraPtr& camera, const mango::Pose3D& pose)
{
    mango::Point3D pt_camera = transform(pt_world, pose);
    mango::Point2D p = project(pt_camera, camera);
    return p;
}

/**
 * 像素点反投影到相机坐标系
 * @param p      像素点
 * @param camera 相机参数
*/
mango::Point3D unproject(const mango::Point2D& p, const mango::CameraPtr& camera)
{
    return mango::Point3D(0, 0, 0);
}

/**
 * 像素点反投影到世界坐标系
 * @param p      像素点
 * @param camera 相机参数
 * @param pose   相机位姿
*/
mango::Point3D unproject(const mango::Point2D& p, const mango::CameraPtr& camera, const mango::Pose3D& pose)
{
    return mango::Point3D(0, 0, 0);
}

/**
 * 位姿变换
 * @param pt   参考点
 * @param pose 施加变换
*/
mango::Point3D transform(const mango::Point3D& pt, const mango::Pose3D& pose)
{
    Eigen::Vector4d p_src;
    p_src << pt.x,
             pt.y,
             pt.z,
             1;
    Eigen::Vector4d p_dst = pose.Rt() *  p_src;
    mango::Point3D p(p_dst(0), p_dst(1), p_dst(2));

    return p;
}


/**
 * 计算重投影误差
 * @param pxs    像素点
 * @param pts    世界点
 * @param camera 相机内参
 * @param pose   相机pose
*/
double reprojectionError(const std::vector<mango::Point2D>& pxs, const std::vector<mango::Point3D>& pts, const mango::CameraPtr& camera, const mango::Pose3D& pose)
{
    return 0;
}

}