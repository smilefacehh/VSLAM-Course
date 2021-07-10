#pragma once

#include <memory>
#include <opencv2/opencv.hpp>

#include "pose.hpp"

namespace mango {

class Image
{
public:
    Image() = default;
    
    Image(const cv::Mat& image)
    {
        image_ = image;
    }

    Image(const cv::Mat& image, const mango::Pose3D& pose)
    {
        image_ = image;
        pose_ = pose;
    }

    ~Image();

    Image& operator=(const Image& rhs)
    {
        image_ = rhs.image_;
        pose_ = rhs.pose_;
        return *this;
    }

    Image(const Image& rhs)
    {
        image_ = rhs.image_;
        pose_ = rhs.pose_;
    }

    /**
     * 获取图像数据
    */
    inline cv::Mat& image()
    {
        return image_;
    }

    /**
     * 获取对应相机位姿
    */
    inline const mango::Pose3D& pose()
    {
        return pose_;
    }
private:
    cv::Mat image_;
    mango::Pose3D pose_;
};

using ImagePtr = std::shared_ptr<Image>;
}