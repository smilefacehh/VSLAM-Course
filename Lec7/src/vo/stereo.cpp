#include "stereo.h"
#include "util/math_util.h"
#include <limits.h>

namespace mango
{
Stereo::Stereo() {}
Stereo::~Stereo() {}

cv::Mat Stereo::match(const cv::Mat& left_img, const cv::Mat& right_img, float patch_radius, float min_disp, float max_disp)
{
    cv::Mat disp_img = cv::Mat::zeros(left_img.size(), CV_8UC1);
    
    cv::Mat left_patch, right_patch;

    // 遍历左图像素点
    for(int r = patch_radius; r < left_img.rows - patch_radius; r++)
    {
        for(int c = max_disp + patch_radius; r < left_img.cols - patch_radius; c++)
        {
            double min_ssd_val = std::numeric_limits<double>::max();
            double disp_val = 0;

            left_patch = left_img.rowRange(r-patch_radius, r+patch_radius).colRange(c-patch_radius, c+patch_radius);
            // 遍历右图扫描线，范围
            for(int i = min_disp; i <= max_disp; i++)
            {
                right_patch = right_img.rowRange(r-patch_radius, r+patch_radius).colRange(c-patch_radius-i, c+patch_radius-i);
                double ssd_val = mango::ssd(left_patch, right_patch);
                if(ssd_val < min_ssd_val)
                {
                    min_ssd_val = ssd_val;
                    disp_val = i;
                }
            }
            disp_img.at<uchar>(r, c) = disp_val;
        }
    }

    return disp_img;
}

void Stereo::triangulate() {}
}