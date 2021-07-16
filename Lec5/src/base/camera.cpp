#include "camera.h"

#include <opencv2/core/core.hpp>

namespace mango {

Camera::Camera(int id, const std::string& name)
    : id_(id), name_(name), fx(0.0), fy(0.0), cx(0.0), cy(0.0), k1(0.0), k2(0.0)
{

}

Camera::~Camera(){}

void Camera::loadConfig(const std::string& file)
{
    cv::FileStorage fs(file, cv::FileStorage::READ);
    
    if(!fs.isOpened())
    {
        return;
    }
    cv::FileNode n = fs["camera"]["intrinsics"];
    fx = static_cast<double>(n["fx"]);
    fy = static_cast<double>(n["fy"]);
    cx = static_cast<double>(n["cx"]);
    cy = static_cast<double>(n["cy"]);

    n = fs["camera"]["distort"];
    k1 = static_cast<double>(n["k1"]);
    k2 = static_cast<double>(n["k2"]);

    // 内参矩阵
    K_ << fx,  0, cx,
           0, fy, cy,
           0,  0,  1;
}

}